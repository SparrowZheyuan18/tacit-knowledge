"""Reconstruct the expert's rationale for each key step (results/steps.jsonl -> results/rationales.jsonl).

For every step: one independent reconstruction per model in rationale.models (no hindsight), then one
consensus call that sees the samples plus the step's outcome and the following steps, rates their agreement,
checks plausibility against what happened next, and writes a merged rationale. A daily-quota error from the
gateway aborts the run (records already written are kept; rerun to resume).

    python rationale.py                 # all key steps (resumable)
    python rationale.py --limit 10      # pilot
    python rationale.py --show 3        # print 3 finished records
"""
import argparse, datetime, json
from pathlib import Path

from data import CFG, WORK, STEPS_JSONL, load_steps, task_desc, render_history, render_code, fmt_score
from llm import call_model, parse_json, pmap_jsonl

RATIONALES_JSONL = WORK / "rationales.jsonl"
R = CFG["rationale"]

SAMPLE_SYSTEM = """You are reconstructing the reasoning of an experienced Kaggle competitor at one point in the edit history of their notebook.
You see: the competition, the earlier edits with their public leaderboard results, the full code before this edit, and the edit itself (code diff plus a short annotation). The author's own reasoning was not recorded; reconstruct it from this evidence only. Notebook outputs (validation logs, plots, printed values) are not available; when the edit must have relied on something not visible here, say so.
Respond ONLY with JSON:
{
  "situation": ["<facts visible in the history, scores, or code that bear on this edit>"],
  "problem": "<what the author is addressing: a bottleneck, a risk, a hypothesis to test, or a routine need>",
  "decision": "<the edit in one sentence>",
  "reasoning": "<the inference from situation to decision, as the author would have made it>",
  "alternatives": "<other moves available at this point and why this one was chosen over them>",
  "expected_effect": "<what the author expected to happen to CV / leaderboard>",
  "step_type": "deliberate" | "exploratory" | "routine" | "fix",
  "evidence": "<the specific parts of the history, code, or diff that support this reconstruction>",
  "confidence": 0 | 1 | 2 | 3
}
step_type: deliberate = a choice motivated by something visible in the state; exploratory = trying an option without a specific trigger; routine = standard practice applied regardless of state; fix = correcting an error.
confidence: 3 = the motivation is clearly visible in the evidence; 2 = plausible with partial evidence; 1 = weak evidence; 0 = a guess."""

CONSENSUS_SYSTEM = """You are checking several independent reconstructions of why a Kaggle competitor made one edit.
You see the edit, the reconstructions, and (as hindsight the author did not have) the leaderboard outcome and the next edits.
Tasks: (1) rate how much the reconstructions agree on the author's motivation; (2) judge whether the majority reconstruction is consistent with what happened next; (3) write one merged reconstruction that keeps only what the evidence supports.
Respond ONLY with JSON:
{
  "agreement": 0 | 1 | 2 | 3,
  "disagreements": "<where the reconstructions differ, or 'none'>",
  "plausible": true | false,
  "plausibility_note": "<does the hindsight support or contradict the reconstructed motivation?>",
  "consensus": {
    "situation": ["..."], "problem": "...", "decision": "...", "reasoning": "...", "alternatives": "...",
    "expected_effect": "...", "step_type": "deliberate" | "exploratory" | "routine" | "fix", "evidence": "...", "confidence": 0 | 1 | 2 | 3
  }
}
agreement: 3 = same problem and same reasoning; 2 = same motivation, details differ; 1 = different motivations; 0 = contradictory.
In the merged reconstruction, `situation`, `reasoning` and `evidence` may use only information available before the edit (history, scores so far, code, diff); hindsight goes only in plausibility_note."""


def step_header(s):
    sc = s["score"]
    return (f"# This edit: v{s['v_old']} -> v{s['v_new']}  (size: {s['labels']['magnitude']}; labels: {', '.join(s['labels']['coarse_actions'])})\n"
            f"Annotated goal: {s['labels']['goal']}\nAnnotated change: {s['labels']['change']}\n"
            f"Leaderboard before this edit: {fmt_score(sc['old'])}; best so far: {fmt_score(sc['best_before'])} "
            f"({'higher' if sc['direction'] == 'max' else 'lower'} is better)")


def sample_prompt(s):
    return f"""# Competition
{task_desc(s['comp'])}

# Earlier edits (oldest first), with public leaderboard results
{render_history(s['history'])}

# Code before this edit (v{s['v_old']})
```python
{render_code(s['code_ref']['kernel'], s['code_ref']['version'])}
```

{step_header(s)}
```diff
{s['diff']}
```

Reconstruct the author's reasoning for this edit."""


def hindsight(s):
    o = s["outcome"]
    nxt = "\n".join(f"- v{x['v_old']}->v{x['v_new']}: {x['goal']} [LB {fmt_score(x['score'])}]" for x in o["next_steps"]) or "(none)"
    return f"Leaderboard after this edit: {fmt_score(o['score_new'])} ({o['score_effect']})\nNext edits:\n{nxt}"


def consensus_prompt(s, samples):
    recs = "\n\n".join(f"## Reconstruction {i+1}\n{json.dumps({k: v for k, v in x.items() if k != 'model'}, ensure_ascii=False, indent=1)}" for i, x in enumerate(samples))
    return f"""{step_header(s)}
```diff
{s['diff'][:6000]}
```

# Reconstructions
{recs}

# Hindsight (the author did not have this)
{hindsight(s)}

Rate agreement and plausibility, then write the merged reconstruction."""


class QuotaExhausted(RuntimeError):
    """The gateway's daily token quota for a model is used up; the run is aborted rather than recorded as errors."""


def _call(system, user, model, temperature):
    try:
        return parse_json(call_model(system, user, model=model, temperature=temperature))
    except Exception as e:
        if "tokens per day" in str(e):
            raise QuotaExhausted(f"{model}: {str(e)[:200]}")
        return {"_error": repr(e)}


def sample_models():
    """One model per sample: rationale.models (a list) if given, else rationale.model repeated n_samples times."""
    return R.get("models") or [R["model"]] * R["n_samples"]


def annotate(s):
    p = sample_prompt(s)
    samples = [dict(_call(SAMPLE_SYSTEM, p, m, R["temperature"]), model=m) for m in sample_models()]
    ok = [x for x in samples if "_error" not in x and "_raw" not in x]
    cons = _call(CONSENSUS_SYSTEM, consensus_prompt(s, ok), R["consensus_model"], 0) if ok else {}
    return dict(step_id=s["step_id"], key_id=s["key_id"], comp=s["comp"], v_old=s["v_old"], v_new=s["v_new"],
                goal=s["labels"]["goal"], samples=samples, agreement=cons.get("agreement"),
                disagreements=cons.get("disagreements"), plausible=cons.get("plausible"),
                plausibility_note=cons.get("plausibility_note"), consensus=cons.get("consensus") or {},
                models=dict(samples=sample_models(), consensus=R["consensus_model"]), prompt_chars=len(p),
                ts=datetime.datetime.now().isoformat(timespec="seconds"))


def load_rationales(path=RATIONALES_JSONL):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if Path(path).exists() else []


def show(recs):
    for r in recs:
        c = r["consensus"]
        print(f"\n=== {r['step_id']}  [{r['comp']}]  agreement={r['agreement']}  plausible={r['plausible']}  type={c.get('step_type')}  conf={c.get('confidence')}")
        print(f"goal (label): {r['goal']}")
        print(f"problem:   {c.get('problem')}\ndecision:  {c.get('decision')}\nreasoning: {c.get('reasoning')}")
        print(f"alternatives: {c.get('alternatives')}\nevidence: {c.get('evidence')}")
        print(f"disagreements: {r['disagreements']}\nplausibility: {r['plausibility_note']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N steps")
    ap.add_argument("--sample", type=int, default=None, help="N evenly spaced steps (pilot across competitions)")
    ap.add_argument("--workers", type=int, default=R["workers"])
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    if a.show:
        show(load_rationales()[-a.show:])
    else:
        steps = load_steps(STEPS_JSONL)
        if a.sample:
            steps = steps[:: max(1, len(steps) // a.sample)][: a.sample]
        steps = steps[: a.limit]
        recs = pmap_jsonl(steps, annotate, RATIONALES_JSONL, workers=a.workers, desc="rationale")
        import collections
        print("agreement:", collections.Counter(r["agreement"] for r in recs))
        print("step_type:", collections.Counter(r["consensus"].get("step_type") for r in recs))
        print("plausible:", collections.Counter(r["plausible"] for r in recs))
