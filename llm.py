"""LLM access for the TraceML breakpoint experiments, via the CMU LiteLLM proxy.

Usage:
    from llm import CFG, call_model, parse_json
    txt = call_model("You are ...", "user prompt")                 # generator model from config.yaml
    txt = call_model(system, user, model=CFG["model"]["judge"])    # any model name from the proxy
    obj = parse_json(txt)

Smoke test (needs LITELLM_API_KEY in .env):
    python llm.py                      # one short call to the generator model
    python llm.py --model wine-claude-haiku-4-5 --prompt "Say hi"
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

API_BASE = CFG["litellm"]["api_base"]
PREFIX   = CFG["litellm"].get("prefix", "openai/")


def api_key():
    k = os.environ.get("LITELLM_API_KEY", "").strip()
    if not k:
        raise RuntimeError("LITELLM_API_KEY is empty. Put your key in .env (LITELLM_API_KEY=...) next to config.yaml.")
    return k


def qualify(model):
    """'wine-claude-opus-4-6' -> 'openai/wine-claude-opus-4-6' (the proxy speaks the OpenAI protocol)."""
    return model if model.startswith(PREFIX) else PREFIX + model


def call_model(system, user, model=None, max_tokens=None, temperature=None, messages=None, **kw):
    """One chat completion; returns the assistant text. Retries on transient errors.

    Pass `messages` to supply a full conversation instead of system+user.
    """
    import litellm
    litellm.suppress_debug_info = True
    m   = CFG["model"]
    msgs = messages or [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last = None
    for attempt in range(m.get("max_retries", 3)):
        try:
            r = litellm.completion(
                api_key=api_key(), base_url=API_BASE, model=qualify(model or m["generator"]),
                messages=msgs,
                max_tokens=max_tokens if max_tokens is not None else m["max_tokens"],
                temperature=temperature if temperature is not None else m["temperature"],
                timeout=m.get("timeout_s", 120), **kw)
            return r.choices[0].message.content
        except Exception as e:  # rate limit / timeout / 5xx: back off and retry
            last = e
            if "LITELLM_API_KEY is empty" in str(e) or "AuthenticationError" in type(e).__name__:
                raise
            time.sleep(2 ** attempt)
    raise last


def parse_json(txt):
    """Tolerant JSON parse of a model reply (strips fences, falls back to the first {...} block)."""
    txt = re.sub(r"^```(?:json)?|```$", "", (txt or "").strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"_raw": txt}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="proxy model name; default = model.generator in config.yaml")
    ap.add_argument("--prompt", default="Reply with the single word: pong")
    ap.add_argument("--list", action="store_true", help="print configured models and exit")
    a = ap.parse_args()
    if a.list:
        print("api_base :", API_BASE)
        print("generator:", CFG["model"]["generator"])
        print("judge    :", CFG["model"]["judge"])
        print("available:", ", ".join(CFG["litellm"]["available_models"]))
        sys.exit(0)
    t0 = time.time()
    out = call_model("You are a helpful assistant.", a.prompt, model=a.model, max_tokens=50, temperature=0)
    print(f"[{a.model or CFG['model']['generator']}, {time.time()-t0:.1f}s] {out}")


# ----------------------------------------------------------------------------------------------- batch helper
def pmap_jsonl(items, fn, path, key=lambda r: r["step_id"], workers=8, desc="", overwrite=False):
    """Run fn(item) -> dict in a thread pool and append each result as one JSON line to `path`.
    Items whose key is already present in the file are skipped, so a run can be resumed. Returns all records."""
    import concurrent.futures as cf
    path = Path(path)
    done = {}
    if path.exists() and not overwrite:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l); done[key(r)] = r
    todo = [it for it in items if key(it) not in done]
    print(f"{desc}: {len(done)} done, {len(todo)} to run, {workers} workers", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex, open(path, "w" if overwrite else "a", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(fn, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            done[key(rec)] = rec
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    return [done[key(it)] for it in items if key(it) in done]
