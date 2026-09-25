# tacit-knowledge

Extract tacit knowledge (conditional rules: *in situation X, do Y*) from expert trajectories in the
[TraceML](https://huggingface.co/datasets/jerryyan/TraceML) release of Kaggle notebook version histories.

## Pipeline

```
data.py       select trajectories, find key steps, build results/steps.jsonl (one record per key step)
rationale.py  reconstruct the expert's reasoning for each key step  -> results/rationales.jsonl
tk.py         extract tacit-knowledge items from the rationales     -> results/tk_{evolve,bottomup}.json, results/report.md
```

```
python scripts/download_traceml.py     # once: ~9 GB into data/TraceML
python data.py                         # ~5 min (first run also builds results/notebook_index.json)
python rationale.py --workers 12       # LLM calls; resumable
python tk.py evolve; python tk.py bottomup
python tk.py match --method evolve; python tk.py match --method bottomup
python tk.py compare                   # writes results/report.md
```

Settings live in `config.yaml`; the API key in `.env` (`LITELLM_API_KEY`). Competition descriptions given to the
model are in `task_descriptions.yaml`; the seed rule list for the evolve method in `seed_tk.yaml`.

## Method

**Trajectory selection** (`data.py`). Human trajectories of the `paired` split (7 competitions). Per competition,
trajectories with >= 5 version transitions, >= 3 scored versions and notebook source for >= 90% of versions are
ranked by their best public-leaderboard score (direction from `manifests/competitions.json`) and the top 10 kept.
Author tier is not used: it correlates only weakly with notebook score.

**Key steps.** A transition is a key step if (a) it lies at or before the trajectory's best-scoring version,
(b) at least one code line it added still exists in that best version (retention), and (c) its action labels are
not only housekeeping/infra. At most 30 key steps per trajectory are kept, evenly spaced. Each record carries the
step's labels, scores (delta vs. previous submission, best so far, new-best flag), retention counts, the full
structured history of earlier transitions (what was tried, what the leaderboard said), the diff, a reference to
the code before the edit, and the outcome (kept separate as hindsight).

**Rationale reconstruction** (`rationale.py`). For each key step, one reconstruction per model in
`rationale.models` (competition description, history, code before the edit, diff; no hindsight), then a consensus
call that sees the reconstructions plus the outcome and next edits, rates agreement (0-3), judges plausibility,
and writes a merged rationale whose evidence may cite only pre-edit information.

**Tacit-knowledge extraction** (`tk.py`), on steps whose consensus type is deliberate or exploratory:
- *evolve*: start from `seed_tk.yaml`; rationales are processed in shuffled batches of 15; each batch matches steps
  to items and proposes new items, revisions and merges, applied before the next batch; a final consolidation pass.
- *bottomup*: 0-2 candidate rules per rationale; candidates merged hierarchically (LLM grouping in chunks of 40,
  reshuffled between rounds) into a list.
- *match*: every step is assigned to the items of a list by a separate strict matching pass (support counts).
- *compare*: coverage, support, and the correspondence between the two lists -> `results/report.md`.
