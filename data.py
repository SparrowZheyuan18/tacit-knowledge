"""TraceML data layer: load the release, select expert trajectories, find key steps, and write one
structured record per key step (with the full prior history) for rationale annotation.

    python data.py            # build results/transitions.csv + results/steps.jsonl, print statistics
    python data.py --stats    # only print statistics from the files already written

Selection (config.yaml -> selection): per competition, human trajectories with >= min_transitions version
transitions, >= min_scored scored versions and notebook source for >= min_source_frac of their versions, ranked by
their best public score (direction-aware), top_k kept.
Key step: a transition at or before the trajectory's best-scoring version whose added code lines still exist
in that best version (retention >= 1 line) and whose action labels are not only housekeeping/infra. At most
max_key_steps_per_trajectory key steps are kept per trajectory (evenly spaced), so one long notebook cannot
dominate the sample.
"""
import argparse, datetime, difflib, functools, glob, json, os, re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
P = CFG["paths"]
WORK = ROOT / P["work_dir"]
TRANSITIONS_CSV = WORK / "transitions.csv"
STEPS_JSONL = WORK / "steps.jsonl"

COMPS = json.load(open(ROOT / P["competitions"], encoding="utf-8"))
TASK_DESC = yaml.safe_load(open(ROOT / "task_descriptions.yaml", encoding="utf-8"))
TRIVIAL_ACTIONS = {"housekeeping", "infra"}


# ----------------------------------------------------------------------------------------------- loading
def direction(comp):
    """'max' or 'min' from competitions.json."""
    return "min" if COMPS.get(comp, {}).get("score_direction") == "lower" else "max"


def task_desc(comp):
    m = COMPS.get(comp, {})
    return TASK_DESC.get(comp) or f"{m.get('name', comp)} (Kaggle). Metric: {m.get('metric', 'unknown')}, {m.get('score_direction', '')} is better."


def load_actions(split=None, humans_only=True):
    """Version-edge transitions of the split, sorted by trajectory and version."""
    df = pd.read_parquet(ROOT / P["action_parquet"].format(split=split or P["split"]))
    df["key_id"] = df["key_id"].astype(str)
    if humans_only:
        df = df[~df["is_agent"]]
    df = df[df["edge_kind"].fillna("version") == "version"]
    return df.sort_values(["comp", "key_id", "v_old"]).reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def notebook_index():
    """{(kernel_id, version_number): path} over data/TraceML/trajectories_human/human/<kernel>/versions/vNNN.ipynb.
    The directory walk over ~167k files is slow on network storage, so the index is cached in results/."""
    cache = WORK / "notebook_index.json"
    if cache.exists():
        return {(k, int(v)): p for k, v, p in json.load(open(cache))}
    idx = {}
    for p in glob.glob(str(ROOT / P["notebook_dir"] / "**" / "*.ipynb"), recursive=True):
        p = Path(p)
        m = re.match(r"^v?(\d+)$", p.stem)
        kernel = p.parent.parent.name if p.parent.name == "versions" else p.parent.name
        if m and kernel.isdigit():
            idx[(kernel, int(m.group(1)))] = str(p)
    WORK.mkdir(exist_ok=True)
    json.dump([[k, v, p] for (k, v), p in idx.items()], open(cache, "w"))
    return idx


def nb_to_source(path, keep_markdown=True):
    """Notebook -> plain text: code cells verbatim, markdown cells as # comments, cells separated by a marker."""
    try:
        nb = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return f"# [failed to read notebook: {e}]"
    parts = []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "code":
            parts.append(src)
        elif keep_markdown and c.get("cell_type") == "markdown":
            parts.append("\n".join("# " + l for l in src.splitlines()))
    return "\n\n# ---CELL---\n\n".join(parts)


@functools.lru_cache(maxsize=4096)
def source(key_id, version):
    """Source text of one version, or None if the notebook is missing."""
    p = notebook_index().get((str(key_id), int(version)))
    return nb_to_source(p) if p else None


def unified_diff(a, b, la, lb, max_lines, n=2):
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), la, lb, lineterm="", n=n))
    total = len(d)
    if total > max_lines:
        d = d[:max_lines] + [f"... [{total - max_lines} more diff lines omitted]"]
    return ("\n".join(d) if d else "(no code change)"), total


def _jsonlist(x, key=None):
    """Parse a JSON-encoded label list; with `key`, keep that field of each dict element."""
    if not isinstance(x, str):
        return []
    try:
        v = json.loads(x)
    except Exception:
        return [x]
    return [e.get(key, e) if isinstance(e, dict) else e for e in v] if key else v


def _iso(ts):
    return None if pd.isna(ts) else datetime.datetime.fromtimestamp(float(ts), datetime.UTC).strftime("%Y-%m-%dT%H:%M")


def _f(x, cast=float):
    """NaN/None -> None, else cast."""
    return None if x is None or pd.isna(x) else cast(x)


def fmt_score(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.5f}"


# ----------------------------------------------------------------------------------------------- selection
def trajectory_table(act):
    g = act.groupby(["comp", "key_id"]).agg(tier=("group", "first"), n_trans=("v_old", "size"),
                                            n_scored=("score_new", lambda x: x.notna().sum()),
                                            smax=("score_new", "max"), smin=("score_new", "min"),
                                            first_v=("v_old", "min"), last_v=("v_new", "max")).reset_index()
    g["best"] = [r.smax if direction(r.comp) == "max" else r.smin for r in g.itertuples()]
    return g.drop(columns=["smax", "smin"])


def select_trajectories(act, sel=None):
    """Top-k trajectories per competition by best public score, subject to size filters. Adds rank + author."""
    sel = sel or CFG["selection"]
    t = trajectory_table(act)
    t = t[(t.n_trans >= sel["min_transitions"]) & (t.n_scored >= sel["min_scored"]) & t.best.notna()].copy()
    idx = notebook_index()
    t["source_frac"] = [sum((k, v) in idx for v in range(int(a), int(b) + 1)) / (int(b) - int(a) + 1)
                        for k, a, b in zip(t.key_id, t.first_v, t.last_v)]
    t = t[t.source_frac >= sel["min_source_frac"]]
    t["_key"] = [r.best if direction(r.comp) == "max" else -r.best for r in t.itertuples()]
    t = t.sort_values(["comp", "_key"], ascending=[True, False])
    t["rank"] = t.groupby("comp").cumcount() + 1
    t = t[t["rank"] <= sel["top_k"]].drop(columns="_key")
    k = pd.read_parquet(ROOT / P["data_root"] / "extras" / "kernels.parquet")[["kernel_id", "author_username"]]
    k["kernel_id"] = k.kernel_id.astype(str)
    return t.merge(k, left_on="key_id", right_on="kernel_id", how="left").drop(columns="kernel_id").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- key steps
def trajectory(act, key_id):
    return act[act.key_id == str(key_id)].sort_values("v_old").reset_index(drop=True)


def best_version_row(t):
    """Transition whose v_new is the best-scoring version (direction-aware; ties -> earliest)."""
    scored = t[t.score_new.notna()]
    if not len(scored):
        return t.iloc[-1]
    asc = direction(t.comp.iloc[0]) == "min"
    return scored.sort_values(["score_new", "v_new"], ascending=[asc, True]).iloc[0]


def _norm_lines(src):
    out = []
    for l in (src or "").splitlines():
        l = re.sub(r"#.*$", "", l).strip()
        if len(l) >= 8:
            out.append(l)
    return out


def annotate_trajectory(t):
    """Per-transition signals for one trajectory: score deltas, retention in the best version, key flag."""
    t = t.copy()
    d = direction(t.comp.iloc[0])
    best = best_version_row(t)
    ref = set(_norm_lines(source(best.key_id, best.v_new)))
    better = (lambda a, b: a > b) if d == "max" else (lambda a, b: a < b)

    rows, prev_scored, best_so_far = [], None, None
    first = t.iloc[0]
    if not pd.isna(first.score_old):
        prev_scored = best_so_far = float(first.score_old)
    for r in t.itertuples():
        a, b = source(r.key_id, r.v_old), source(r.key_id, r.v_new)
        has_src = a is not None and b is not None
        added = retained = None
        if has_src and ref:
            add = [l[1:] for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
                   if l.startswith("+") and not l.startswith("+++")]
            add = set(_norm_lines("\n".join(add)))
            added, retained = len(add), len(add & ref)
        delta = None if pd.isna(r.score_new) or prev_scored is None else float(r.score_new) - prev_scored
        new_best = (not pd.isna(r.score_new)) and (best_so_far is None or better(r.score_new, best_so_far))
        coarse = set(_jsonlist(r.coarse_actions))
        rows.append(dict(has_source=has_src, added_lines=added, retained_lines=retained,
                         delta_vs_prev=delta, best_before=best_so_far, new_best=bool(new_best),
                         before_best=bool(r.v_new <= best.v_new),
                         substantive=bool(coarse - TRIVIAL_ACTIONS)))
        if not pd.isna(r.score_new):
            prev_scored = float(r.score_new)
            best_so_far = prev_scored if best_so_far is None or better(prev_scored, best_so_far) else best_so_far
    sig = pd.DataFrame(rows, index=t.index)
    t = pd.concat([t, sig], axis=1)
    t["retained"] = t.retained_lines.fillna(0) >= 1
    t["is_key"] = t.before_best & t.retained & t.substantive
    t["best_v_new"] = int(best.v_new)
    return t


def history_entries(t, upto_v_old):
    """Structured list of the transitions before v_old: what was tried and what the leaderboard said."""
    out = []
    for r in t[t.v_new <= upto_v_old].itertuples():
        out.append(dict(v_old=int(r.v_old), v_new=int(r.v_new), time=_iso(r.ctime_new),
                        coarse_actions=_jsonlist(r.coarse_actions), magnitude=r.magnitude,
                        goal=r.goal_nl, change=r.diff_summary,
                        score=_f(r.score_new), delta_vs_prev=_f(r.delta_vs_prev), new_best=bool(r.new_best)))
    return out


def step_record(t, r, sel_row):
    """One key-step record: labels, scores, retention, prior history, the diff, and (hindsight) outcome."""
    ctx = CFG["context"]
    a, b = source(r.key_id, r.v_old), source(r.key_id, r.v_new)
    diff, n_diff = unified_diff(a or "", b or "", f"v{r.v_old}", f"v{r.v_new}", ctx["diff_max_lines"])
    later = t[t.v_old > r.v_old].head(3)
    return dict(
        step_id=f"{r.key_id}:{r.v_old}->{r.v_new}", key_id=str(r.key_id), comp=r.comp, tier=r.group,
        author=sel_row.author_username, traj_rank=int(sel_row["rank"]),
        v_old=int(r.v_old), v_new=int(r.v_new), time_old=_iso(r.ctime_old), time_new=_iso(r.ctime_new),
        labels=dict(coarse_actions=_jsonlist(r.coarse_actions), fine_actions=_jsonlist(r.fine_actions, "action"),
                    intents=_jsonlist(r.intents, "intent"), magnitude=r.magnitude, score_effect=r.score_effect,
                    goal=r.goal_nl, change=r.diff_summary),
        score=dict(direction=direction(r.comp), old=_f(r.score_old), new=_f(r.score_new),
                   delta_vs_prev=_f(r.delta_vs_prev), best_before=_f(r.best_before), new_best=bool(r.new_best)),
        retention=dict(added_lines=_f(r.added_lines, int), retained_lines=_f(r.retained_lines, int), best_version=int(r.best_v_new)),
        history=history_entries(t, r.v_old),
        diff=diff, diff_lines=n_diff,
        code_ref=dict(kernel=str(r.key_id), version=int(r.v_old)),
        outcome=dict(score_new=_f(r.score_new), score_effect=r.score_effect,
                     next_steps=[dict(v_old=int(x.v_old), v_new=int(x.v_new), goal=x.goal_nl, score=_f(x.score_new))
                                 for x in later.itertuples()]),
    )


# ----------------------------------------------------------------------------------------------- rendering for prompts
def render_history(hist, max_chars=None):
    """Compact text of the prior transitions for a prompt; drops the oldest entries if too long."""
    max_chars = max_chars or CFG["context"]["history_max_chars"]
    lines = []
    for h in hist:
        s = f"- v{h['v_old']}->v{h['v_new']} [{', '.join(h['coarse_actions']) or '-'}; {h['magnitude']}] {h['goal']}"
        if h["change"] and h["change"] != h["goal"]:
            s += f" | change: {h['change']}"
        if h["score"] is not None:
            s += f" | LB {h['score']:.5f}"
            if h["delta_vs_prev"] is not None:
                s += f" ({h['delta_vs_prev']:+.5f} vs previous submission)"
            if h["new_best"]:
                s += " NEW BEST"
        else:
            s += " | not submitted"
        lines.append(s)
    dropped = 0
    while len(lines) > 1 and sum(len(l) + 1 for l in lines) > max_chars:
        lines.pop(0); dropped += 1
    if dropped:
        lines.insert(0, f"(the earliest {dropped} transitions are omitted for length)")
    return "\n".join(lines) if lines else "(this is the first version)"


def render_code(key_id, version, max_chars=None):
    max_chars = max_chars or CFG["context"]["code_max_chars"]
    s = source(key_id, version)
    if s is None:
        return "(source not available)"
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n# ... [truncated, {len(s) - max_chars} chars omitted]"
    return s


# ----------------------------------------------------------------------------------------------- build + stats
def load_steps(path=STEPS_JSONL, key_only=True):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build():
    WORK.mkdir(exist_ok=True)
    act = load_actions()
    sel = select_trajectories(act)
    ann, steps = [], []
    for s in sel.itertuples():
        t = annotate_trajectory(trajectory(act, s.key_id))
        ann.append(t)
        keys = t[t.is_key]
        cap = CFG["selection"].get("max_key_steps_per_trajectory")
        if cap and len(keys) > cap:  # long trajectories: keep `cap` key steps evenly spaced over the trajectory
            keys = keys.iloc[sorted(set(np.linspace(0, len(keys) - 1, cap).round().astype(int)))]
        steps += [step_record(t, r, sel.loc[s.Index]) for r in keys.itertuples()]
    ann = pd.concat([x for x in ann if len(x)])
    cols = ["comp", "key_id", "group", "v_old", "v_new", "magnitude", "score_effect", "coarse_actions", "goal_nl",
            "score_old", "score_new", "delta_vs_prev", "new_best", "has_source", "added_lines", "retained_lines",
            "before_best", "retained", "substantive", "is_key", "best_v_new"]
    ann[cols].to_csv(TRANSITIONS_CSV, index=False)
    sel.to_csv(WORK / "selection.csv", index=False)
    with open(STEPS_JSONL, "w", encoding="utf-8") as f:
        for s in steps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {WORK/'selection.csv'} ({len(sel)} trajectories), {TRANSITIONS_CSV} ({len(ann)} transitions), {STEPS_JSONL} ({len(steps)} key steps)")
    stats(act, sel, ann)


def stats(act=None, sel=None, ann=None):
    act = act if act is not None else load_actions()
    sel = sel if sel is not None else pd.read_csv(WORK / "selection.csv", dtype={"key_id": str})
    ann = ann if ann is not None else pd.read_csv(TRANSITIONS_CSV, dtype={"key_id": str})
    raw = pd.read_parquet(ROOT / P["action_parquet"].format(split=P["split"]))
    s = CFG["selection"]
    print(f"\n== split '{P['split']}': raw action rows {len(raw):,} | human {(~raw.is_agent).sum():,} | human version edges {len(act):,} "
          f"| trajectories {act.key_id.nunique():,} | competitions {act.comp.nunique()}")
    print(f"== selection: top_k={s['top_k']} per comp by best public score, min_transitions={s['min_transitions']}, min_scored={s['min_scored']}, min_source_frac={s['min_source_frac']}")
    print(f"   selected trajectories {len(sel)} | authors {sel.author_username.nunique()} | comps {sel.comp.nunique()} | transitions {len(ann):,}")
    print(f"   tiers of selected: {sel.tier.fillna('None').value_counts().to_dict()}")
    print("\n   per competition:")
    per = ann.groupby("comp").agg(traj=("key_id", "nunique"), transitions=("v_old", "size"),
                                  with_source=("has_source", "sum"), before_best=("before_best", "sum"),
                                  retained=("retained", "sum"), key=("is_key", "sum"))
    print(per.to_string())
    print(f"\n   funnel: transitions {len(ann):,} -> with source {int(ann.has_source.sum()):,} -> at/before best version "
          f"{int(ann.before_best.sum()):,} -> retained >=1 line {int((ann.before_best & ann.retained).sum()):,} "
          f"-> and substantive (not only housekeeping/infra) = key {int(ann.is_key.sum()):,}")
    k = ann[ann.is_key]
    print(f"   key steps: magnitude {k.magnitude.value_counts().to_dict()} | score_effect {k.score_effect.value_counts().to_dict()} | new_best {int(k.new_best.sum())}")
    print(f"   key steps per trajectory: median {k.groupby('key_id').size().median():.0f}, max {k.groupby('key_id').size().max()}")
    n_written = sum(1 for _ in open(STEPS_JSONL, encoding="utf-8"))
    print(f"   written to steps.jsonl after the per-trajectory cap ({s.get('max_key_steps_per_trajectory')}): {n_written}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    stats() if a.stats else build()
