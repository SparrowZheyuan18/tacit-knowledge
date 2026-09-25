"""Tacit-knowledge extraction from step rationales (results/rationales.jsonl).

An item is a conditional rule: {id, name, context (observable situation), action (what to do, and otherwise)}.
Two methods build a list:
  evolve   : start from seed_tk.yaml; rationales are processed in batches; for each batch the model matches
             steps to items and proposes new items / revisions / merges, which are applied before the next
             batch; a final consolidation pass removes duplicates.
  bottomup : each rationale yields 0-2 candidate rules; candidates are merged hierarchically into a list.
Then `match` assigns every step to the items of a list (support), and `compare` reports coverage, support
and the correspondence between the two lists.

    python tk.py evolve [--resume]     python tk.py bottomup; python tk.py finalize
    python tk.py match --method evolve|bottomup     python tk.py compare     python tk.py show --method evolve
"""
import argparse, collections, datetime, json, random
from pathlib import Path

import yaml

from data import CFG, WORK, ROOT, load_steps, STEPS_JSONL
from rationale import load_rationales
from llm import call_model, parse_json, pmap_jsonl

T = CFG["tk"]
LIST_PATH = lambda m: WORK / f"tk_{m}.json"
MATCH_PATH = lambda m: WORK / f"tk_match_{m}.jsonl"

ITEM_RULES = """An item is a judgement rule, not an action category: `context` names observable conditions (in the code, data, scores, history or remaining budget) under which the rule applies, and `action` says what to do when the context holds and what to do otherwise. Items must be general across competitions but specific enough to be wrong in some situations; plain best practice with no condition ("use cross-validation", "try an ensemble") is not an item."""


# ----------------------------------------------------------------------------------------------- inputs
def rationale_steps():
    """Steps whose consensus rationale qualifies for extraction, as compact dicts."""
    steps = {s["step_id"]: s for s in load_steps(STEPS_JSONL)}
    out = []
    for r in load_rationales():
        c = r.get("consensus") or {}
        if r["step_id"] not in steps or not c or c.get("step_type") not in T["step_types"] or (c.get("confidence") or 0) < T["min_confidence"]:
            continue
        s = steps[r["step_id"]]
        o = s["outcome"]
        outcome = f"LB {o['score_new']:.5f} ({o['score_effect']})" if o["score_new"] is not None else "not submitted"
        out.append(dict(step_id=r["step_id"], comp=r["comp"], step_type=c["step_type"], confidence=c["confidence"],
                        situation=c.get("situation"), problem=c.get("problem"), decision=c.get("decision"),
                        reasoning=c.get("reasoning"), alternatives=c.get("alternatives"),
                        expected_effect=c.get("expected_effect"), outcome=outcome))
    return out


def render_steps(batch):
    keys = ["comp", "step_type", "situation", "problem", "decision", "reasoning", "alternatives", "expected_effect", "outcome"]
    return "\n\n".join(f"## step {b['step_id']}\n" + "\n".join(f"{k}: {json.dumps(b[k], ensure_ascii=False) if isinstance(b[k], list) else b[k]}" for k in keys)
                       for b in batch)


def render_items(items, active_only=True):
    return "\n".join(f"- [{i['id']}] {i['name']}\n  context: {i['context']}\n  action: {i['action']}"
                     for i in items if i.get("active", True) or not active_only) or "(empty list)"


def batches(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def save_list(method, items, meta):
    json.dump(dict(method=method, meta=meta, items=items), open(LIST_PATH(method), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def load_list(method):
    return json.load(open(LIST_PATH(method), encoding="utf-8"))


# ----------------------------------------------------------------------------------------------- method 1: evolve
EVOLVE_SYSTEM = f"""You maintain a list of tacit-knowledge items used by experienced Kaggle competitors. {ITEM_RULES}
You receive the current list and a batch of reconstructed rationales for edits made by top competitors. For each rationale, decide which items the author's judgement instantiates: match only when the item's context visibly held in the situation and the author's decision is the item's action. Where a rationale shows a judgement that no item captures, propose a new item. Where an item nearly fits but is too narrow or too broad, propose a revision. Where two items express the same rule, propose a merge. Do not propose an item from a single step unless the judgement is clearly stated; prefer revising an existing item over adding a near-duplicate.
Respond ONLY with JSON:
{{
  "matches": [{{"step_id": "...", "tk_ids": ["..."], "note": "<one clause>"}}],
  "new_items": [{{"name": "<snake_case>", "context": "...", "action": "...", "step_ids": ["..."], "why_new": "..."}}],
  "revisions": [{{"id": "...", "context": "...", "action": "...", "reason": "..."}}],
  "merges": [{{"ids": ["...", "..."], "name": "...", "context": "...", "action": "...", "reason": "..."}}]
}}
Include one entry in "matches" for every step (tk_ids may be empty). Use the ids exactly as given; new items have no id yet."""

CONSOLIDATE_SYSTEM = f"""You are cleaning up a list of tacit-knowledge items. {ITEM_RULES}
Merge items that express the same rule, split an item that bundles two unrelated rules, and rewrite vague contexts or actions so each item is a testable conditional. Keep every distinct rule. Respond ONLY with JSON:
{{"items": [{{"name": "<snake_case>", "context": "...", "action": "...", "from_ids": ["..."]}}]}}
`from_ids` lists the input ids each output item is built from (one or several)."""


def next_id(items):
    return f"TK{1 + max([int(i['id'][2:]) for i in items] + [0]):02d}"


def apply_ops(items, ops, batch_no):
    """Apply new_items / revisions / merges from one evolve call to the list in place. Returns the ops log."""
    by_id = {i["id"]: i for i in items}
    log = []
    for m in ops.get("merges") or []:
        ids = [x for x in m.get("ids", []) if x in by_id and by_id[x].get("active", True)]
        if len(ids) < 2:
            continue
        keep = by_id[ids[0]]
        keep["history"].append(dict(batch=batch_no, op="merge", absorbed=ids[1:], old=dict(name=keep["name"], context=keep["context"], action=keep["action"]), reason=m.get("reason")))
        keep.update(name=m.get("name") or keep["name"], context=m["context"], action=m["action"])
        for x in ids[1:]:
            by_id[x].update(active=False, merged_into=keep["id"])
        log.append(dict(op="merge", ids=ids, into=keep["id"]))
    for rv in ops.get("revisions") or []:
        it = by_id.get(rv.get("id"))
        if not it or not it.get("active", True) or not rv.get("context") or not rv.get("action"):
            continue
        it["history"].append(dict(batch=batch_no, op="revise", old=dict(context=it["context"], action=it["action"]), reason=rv.get("reason")))
        it.update(context=rv["context"], action=rv["action"])
        log.append(dict(op="revise", id=it["id"]))
    for n in ops.get("new_items") or []:
        if not n.get("context") or not n.get("action"):
            continue
        it = dict(id=next_id(items), name=n.get("name", "unnamed"), context=n["context"], action=n["action"],
                  source="evolve", active=True, created_batch=batch_no, seed_steps=n.get("step_ids", []),
                  history=[dict(batch=batch_no, op="new", reason=n.get("why_new"))])
        items.append(it); by_id[it["id"]] = it
        log.append(dict(op="new", id=it["id"], name=it["name"]))
    return log


def evolve(resume=False):
    """With resume=True the ops logged in tk_evolve_log.jsonl are replayed onto the seed list and processing
    continues at the next batch (the batch order is fixed by tk.seed)."""
    seed = yaml.safe_load(open(ROOT / "seed_tk.yaml", encoding="utf-8"))
    items = [dict(id=s["id"], name=s["name"], context=s["context"], action=s["action"], source="seed", active=True, history=[]) for s in seed]
    steps = rationale_steps()
    random.Random(T["seed"]).shuffle(steps)
    bs = batches(steps, T["batch_size"])
    log_path, start = WORK / "tk_evolve_log.jsonl", 1
    if resume and log_path.exists():
        for rec in (json.loads(l) for l in open(log_path, encoding="utf-8")):
            apply_ops(items, rec["raw_ops"], rec["batch"]); start = rec["batch"] + 1
    log_f = open(log_path, "a" if resume else "w", encoding="utf-8")
    print(f"evolve: {len(steps)} steps in {len(bs)} batches, {len(seed)} seed items; starting at batch {start} with {sum(i['active'] for i in items)} active items", flush=True)
    for bi, b in enumerate(bs, 1):
        if bi < start:
            continue
        user = f"# Current items\n{render_items(items)}\n\n# Rationales\n{render_steps(b)}\n\nMatch, then propose new items, revisions and merges."
        ops = parse_json(call_model(EVOLVE_SYSTEM, user, model=T["model"], temperature=0, max_tokens=8000))
        log = apply_ops(items, ops, bi)
        log_f.write(json.dumps(dict(batch=bi, model=T["model"], matches=ops.get("matches"), ops=log, raw_ops={k: ops.get(k) for k in ("new_items", "revisions", "merges")}), ensure_ascii=False) + "\n"); log_f.flush()
        n_act = sum(i["active"] for i in items)
        print(f"  batch {bi}/{len(bs)}: {len(log)} ops ({collections.Counter(o['op'] for o in log)}), active items {n_act}", flush=True)
    # consolidation
    active = [i for i in items if i["active"]]
    out = parse_json(call_model(CONSOLIDATE_SYSTEM, f"# Items\n{render_items(active)}\n\nConsolidate.", model=T["model"], temperature=0, max_tokens=12000))
    final = []
    for k, it in enumerate(out.get("items") or [], 1):
        srcs = [x for x in it.get("from_ids", []) if x in {i["id"] for i in active}]
        final.append(dict(id=f"TK{k:02d}", name=it.get("name", "unnamed"), context=it["context"], action=it["action"],
                          from_ids=srcs, source="seed" if all(s[2:].isdigit() and int(s[2:]) <= len(seed) for s in srcs) and srcs else "evolve"))
    save_list("evolve", final, dict(n_steps=len(steps), n_batches=len(bs), pre_consolidation=items, model=T["model"],
                                    ts=datetime.datetime.now().isoformat(timespec="seconds")))
    print(f"evolve done: {len(active)} active items before consolidation -> {len(final)} final items -> {LIST_PATH('evolve')}")


# ----------------------------------------------------------------------------------------------- method 2: bottom-up
EXTRACT_SYSTEM = f"""You read reconstructed rationales of edits made by top Kaggle competitors and write down the tacit knowledge each one shows. {ITEM_RULES}
For each rationale, output 0-2 candidate items: only where the rationale shows a judgement (a condition the author observed, and the choice it led to). Output none for routine steps, for trial-and-error with no stated trigger, and for competition-specific facts. Respond ONLY with JSON:
{{"candidates": [{{"step_id": "...", "name": "<snake_case>", "context": "...", "action": "..."}}]}}"""

MERGE_SYSTEM = f"""You are grouping candidate tacit-knowledge items that were extracted independently from many edits. {ITEM_RULES}
Group candidates that express the same rule (same triggering condition and same kind of response, even if worded differently or observed in different competitions) and write one item per group, generalised across its members but no more general than they support. Keep singletons as their own items. Every candidate must appear in exactly one group. Respond ONLY with JSON:
{{"items": [{{"name": "<snake_case>", "context": "...", "action": "...", "members": ["<candidate id>", ...]}}]}}"""


def extract_batch(b):
    out = parse_json(call_model(EXTRACT_SYSTEM, f"# Rationales\n{render_steps(b)}\n\nExtract candidates.", model=T["model"], temperature=0, max_tokens=6000))
    ids = {x["step_id"] for x in b}
    return dict(batch_id=b[0]["step_id"], candidates=[c for c in out.get("candidates") or [] if c.get("step_id") in ids and c.get("context") and c.get("action")])


def merge_round(cands, chunk):
    """One hierarchical merge round: chunks of candidates -> merged items whose members are candidate ids."""
    def one(ch):
        text = "\n".join(f"- [{c['cid']}] {c['name']}\n  context: {c['context']}\n  action: {c['action']}" for c in ch)
        out = parse_json(call_model(MERGE_SYSTEM, f"# Candidates\n{text}\n\nGroup and merge.", model=T["model"], temperature=0, max_tokens=24000))
        valid = {c["cid"]: c for c in ch}
        seen, items = set(), []
        for it in out.get("items") or []:
            mem = [m for m in it.get("members", []) if m in valid and m not in seen]
            if not mem or not it.get("context") or not it.get("action"):
                continue
            seen |= set(mem)
            items.append(dict(name=it.get("name", "unnamed"), context=it["context"], action=it["action"],
                              members=sorted({s for m in mem for s in valid[m]["members"]})))
        items += [dict(name=c["name"], context=c["context"], action=c["action"], members=c["members"]) for c in ch if c["cid"] not in seen]
        return items
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(T["workers"]) as ex:
        merged = [it for res in ex.map(one, batches(cands, chunk)) for it in res]
    return [dict(cid=f"c{k}", **it) for k, it in enumerate(merged, 1)]


def bottomup():
    steps = rationale_steps()
    bs = batches(steps, 10)
    recs = pmap_jsonl(bs, extract_batch, WORK / "tk_candidates.jsonl", key=lambda r: r["batch_id"] if "batch_id" in r else r[0]["step_id"],
                      workers=T["workers"], desc="bottomup extract")
    cands = [dict(cid=f"c{k}", name=c["name"], context=c["context"], action=c["action"], members=[c["step_id"]])
             for k, c in enumerate((c for r in recs for c in r["candidates"]), 1)]
    print(f"bottomup: {len(steps)} steps -> {len(cands)} candidates", flush=True)
    chunk, rounds = T["merge_chunk"], []
    cur = cands
    random.Random(T["seed"]).shuffle(cur)
    while True:  # rounds of chunked merging; stop after a single-chunk round, when a round barely reduces the list, or after 8 rounds
        nxt = merge_round(cur, chunk)
        rounds.append(dict(n_in=len(cur), n_out=len(nxt)))
        print(f"  merge round {len(rounds)}: {len(cur)} -> {len(nxt)}", flush=True)
        single, n_in = len(cur) <= chunk, len(cur)
        cur = nxt
        if single or len(rounds) >= 8 or len(nxt) >= 0.95 * n_in:
            break
        random.Random(len(rounds)).shuffle(cur)
    _save_bottomup(cur, dict(n_steps=len(steps), n_candidates=len(cands), rounds=rounds, candidates=cands, model=T["model"]))
    print(f"bottomup done: {len(cur)} items -> {LIST_PATH('bottomup')}; run `python tk.py finalize` to consolidate across chunks")


def _save_bottomup(cur, meta):
    final = [dict(id=f"TK{k:02d}", name=it["name"], context=it["context"], action=it["action"], source="bottomup",
                  members=it["members"], n_members=len(it["members"])) for k, it in enumerate(sorted(cur, key=lambda x: -len(x["members"])), 1)]
    meta["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_list("bottomup", final, meta)


def finalize(chunk=120):
    """Consolidate the bottom-up list across chunks: rounds with large chunks until the list fits one call,
    then one pass over the whole list."""
    d = load_list("bottomup")
    cur = [dict(cid=f"c{k}", name=i["name"], context=i["context"], action=i["action"], members=i["members"]) for k, i in enumerate(d["items"], 1)]
    rounds = d["meta"].get("rounds", [])
    while len(cur) > chunk:
        nxt = merge_round(cur, chunk); rounds.append(dict(n_in=len(cur), n_out=len(nxt), chunk=chunk, model=T["model"]))
        print(f"  finalize round: {len(cur)} -> {len(nxt)}", flush=True)
        if len(nxt) >= 0.97 * len(cur):
            cur = nxt; break
        cur = nxt; random.Random(len(rounds)).shuffle(cur)
    nxt = merge_round(cur, len(cur)); rounds.append(dict(n_in=len(cur), n_out=len(nxt), chunk=len(cur), model=T["model"]))
    print(f"  finalize whole-list pass: {len(cur)} -> {len(nxt)}", flush=True)
    d["meta"]["rounds"] = rounds; d["meta"]["finalize_model"] = T["model"]
    _save_bottomup(nxt, d["meta"])
    print(f"finalize done: {len(nxt)} items -> {LIST_PATH('bottomup')}")


# ----------------------------------------------------------------------------------------------- matching + comparison
MATCH_SYSTEM = f"""You assign reconstructed rationales of Kaggle edits to tacit-knowledge items. {ITEM_RULES}
For each rationale, list the items whose context visibly held in the situation and whose action is what the author did. Be strict; an empty list is a valid answer. Respond ONLY with JSON:
{{"matches": [{{"step_id": "...", "tk_ids": ["..."], "confidence": 0 | 1 | 2 | 3}}]}}
confidence: 3 = context and action both clearly present; 2 = action clear, context partly visible; 1 = weak."""


def match(method):
    items = load_list(method)["items"]
    steps = rationale_steps()
    ids = {i["id"] for i in items}
    def one(b):
        out = parse_json(call_model(MATCH_SYSTEM, f"# Items\n{render_items(items)}\n\n# Rationales\n{render_steps(b)}\n\nMatch.", model=T["match_model"], temperature=0, max_tokens=4000))
        got = {m.get("step_id"): m for m in out.get("matches") or []}
        return dict(batch_id=b[0]["step_id"], matches=[dict(step_id=s["step_id"], comp=s["comp"], tk_ids=[x for x in (got.get(s["step_id"], {}).get("tk_ids") or []) if x in ids],
                                                         confidence=got.get(s["step_id"], {}).get("confidence")) for s in b])
    recs = pmap_jsonl(batches(steps, T["batch_size"]), one, MATCH_PATH(method), key=lambda r: r["batch_id"] if "batch_id" in r else r[0]["step_id"],
                      workers=T["workers"], desc=f"match {method}", overwrite=True)
    ms = [m for r in recs for m in r["matches"]]
    support = collections.Counter(t for m in ms for t in m["tk_ids"])
    for i in items:
        i["support"] = support.get(i["id"], 0)
        i["support_comps"] = sorted({m["comp"] for m in ms if i["id"] in m["tk_ids"]})
    d = load_list(method); d["items"] = items; d["meta"]["match_model"] = T["match_model"]
    save_list(method, items, d["meta"])
    cov = sum(bool(m["tk_ids"]) for m in ms) / max(len(ms), 1)
    print(f"match {method}: {len(ms)} steps, coverage {cov:.2f}, items with support>={T['min_support']}: {sum(i['support'] >= T['min_support'] for i in items)}/{len(items)}")


ALIGN_SYSTEM = """Two lists of tacit-knowledge items (conditional rules) were produced by different methods. For each item in list B, name the item in list A that expresses the same rule (same condition and same response), or null if none does. Respond ONLY with JSON:
{"pairs": [{"b_id": "...", "a_id": "..." | null, "relation": "same" | "overlap" | "none"}]}"""


def data_summary():
    """Funnel from raw TraceML rows to extraction inputs, plus rationale-stage statistics."""
    import pandas as pd
    from data import load_actions, TRANSITIONS_CSV, P, ROOT
    raw = pd.read_parquet(ROOT / P["action_parquet"].format(split=P["split"]))
    act = load_actions()
    tr = pd.read_csv(TRANSITIONS_CSV, dtype={"key_id": str})
    sel = pd.read_csv(WORK / "selection.csv", dtype={"key_id": str})
    steps = load_steps(STEPS_JSONL)
    rats = [r for r in load_rationales() if r["step_id"] in {s["step_id"] for s in steps}]
    sc = CFG["selection"]
    ty = collections.Counter(r["consensus"].get("step_type") for r in rats)
    ag = collections.Counter(r["agreement"] for r in rats)
    pl = collections.Counter(r["plausible"] for r in rats)
    q = rationale_steps()
    return "\n".join([
        "## data",
        f"- split `{P['split']}`: {len(raw):,} action rows; {int((~raw.is_agent).sum()):,} human; {len(act):,} human version transitions in {act.key_id.nunique()} trajectories, {act.comp.nunique()} competitions",
        f"- selection: top {sc['top_k']} trajectories per competition by best public score; >= {sc['min_transitions']} transitions, >= {sc['min_scored']} scored versions, source for >= {sc['min_source_frac']:.0%} of versions",
        f"- selected: {len(sel)} trajectories, {sel.author_username.nunique()} authors, {sel.comp.nunique()} competitions, {len(tr):,} transitions; author tiers {{k: int(v) for k, v in sel.tier.fillna('None').value_counts().items()}}",
        f"- key-step funnel: {len(tr):,} transitions -> {int(tr.has_source.sum()):,} with source -> {int(tr.before_best.sum()):,} at/before best version -> {int((tr.before_best & tr.retained).sum()):,} retained -> {int(tr.is_key.sum()):,} substantive (key) -> {len(steps):,} after cap of {sc['max_key_steps_per_trajectory']} per trajectory",
        f"- rationales: {len(rats)} steps; step types {dict(ty)}; agreement between reconstructions {dict(ag)}; plausible given hindsight {dict(pl)}",
        f"- extraction inputs: {len(q)} steps with type in {T['step_types']} and confidence >= {T['min_confidence']}; per competition {dict(collections.Counter(x['comp'] for x in q))}",
    ])


def compare():
    A, B = load_list("evolve"), load_list("bottomup")
    out = parse_json(call_model(ALIGN_SYSTEM, f"# List A (evolve)\n{render_items(A['items'])}\n\n# List B (bottomup)\n{render_items(B['items'])}\n\nAlign.",
                                model=T["model"], temperature=0, max_tokens=6000))
    pairs = out.get("pairs") or []
    lines = ["# Tacit-knowledge extraction report", "", data_summary(), ""]
    for name, L in (("evolve", A), ("bottomup", B)):
        items = L["items"]; ms = [m for r in (json.loads(l) for l in open(MATCH_PATH(name), encoding="utf-8")) for m in r["matches"]]
        cov = sum(bool(m["tk_ids"]) for m in ms) / max(len(ms), 1)
        sup = [i.get("support", 0) for i in items]
        lines += [f"## {name}: {len(items)} items", f"- steps matched to >=1 item: {cov:.2f} of {len(ms)}",
                  f"- items with support >= {T['min_support']}: {sum(s >= T['min_support'] for s in sup)}; median support {sorted(sup)[len(sup)//2] if sup else 0}; max {max(sup) if sup else 0}",
                  f"- items with support from >= 2 competitions: {sum(len(i.get('support_comps', [])) >= 2 for i in items)}", ""]
        for i in sorted(items, key=lambda x: -x.get("support", 0)):
            lines += [f"### {i['id']} {i['name']}  (support {i.get('support', 0)}, comps {len(i.get('support_comps', []))}, source {i.get('source')})",
                      f"- context: {i['context']}", f"- action: {i['action']}", ""]
    same = [p for p in pairs if p.get("relation") == "same"]; ov = [p for p in pairs if p.get("relation") == "overlap"]
    lines += ["## correspondence (bottomup -> evolve)", f"- same rule: {len(same)}; overlapping: {len(ov)}; no counterpart: {len(pairs) - len(same) - len(ov)} of {len(B['items'])}", ""]
    lines += [f"- {p['b_id']} -> {p['a_id']} ({p['relation']})" for p in pairs if p.get("relation") != "none"]
    (WORK / "report.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump(pairs, open(WORK / "tk_alignment.json", "w"), indent=1)
    print("\n".join(lines[:12])); print(f"... full report: {WORK/'report.md'}")


def show(method):
    L = load_list(method)
    for i in sorted(L["items"], key=lambda x: -x.get("support", 0)):
        print(f"[{i['id']}] {i['name']}  support={i.get('support', '?')} comps={len(i.get('support_comps', []))} source={i.get('source')}\n  context: {i['context']}\n  action: {i['action']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evolve", "bottomup", "finalize", "match", "compare", "show", "inputs"])
    ap.add_argument("--method", default="evolve")
    ap.add_argument("--resume", action="store_true", help="evolve: continue from tk_evolve_log.jsonl")
    a = ap.parse_args()
    if a.cmd == "inputs":
        s = rationale_steps(); print(f"{len(s)} qualifying steps:", collections.Counter(x["comp"] for x in s), collections.Counter(x["step_type"] for x in s))
    else:
        dict(evolve=lambda: evolve(a.resume), bottomup=bottomup, finalize=finalize, match=lambda: match(a.method), compare=compare, show=lambda: show(a.method))[a.cmd]()
