"""Download the TraceML release (jerryyan/TraceML) into data/TraceML and extract the human notebooks.

Skips trajectories_experiment_run/ and trajectories_toolkit_demo/ (agent runs, ~13k small files); add them later if needed.
"""
import os, sys, tarfile, glob, time
from huggingface_hub import snapshot_download, hf_hub_download

REPO = "jerryyan/TraceML"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TraceML")
ROOT = os.path.abspath(ROOT)
os.makedirs(ROOT, exist_ok=True)

t0 = time.time()
print("1) small files ->", ROOT, flush=True)
snapshot_download(REPO, repo_type="dataset", local_dir=ROOT,
                  allow_patterns=["data/**", "manifests/**", "extras/**", "code/**",
                                  "README.md", "DATASHEET.md", "LICENSE", "croissant.json"])
print(f"   done in {time.time()-t0:.0f}s", flush=True)

NB_ROOT = os.path.join(ROOT, "trajectories_human")
if glob.glob(os.path.join(NB_ROOT, "**", "*.ipynb"), recursive=True):
    print("2) notebooks already extracted, skipping", flush=True)
else:
    print("2) trajectories_human.tar.gz (2.96 GB)", flush=True)
    tar_path = hf_hub_download(REPO, "trajectories_human.tar.gz", repo_type="dataset", local_dir=ROOT)
    print(f"   downloaded in {time.time()-t0:.0f}s -> {tar_path}; extracting", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(NB_ROOT, filter="data")
    n = len(glob.glob(os.path.join(NB_ROOT, "**", "*.ipynb"), recursive=True))
    print(f"   extracted {n:,} notebooks in {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
