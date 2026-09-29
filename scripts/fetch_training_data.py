"""Download the v2 training data into the current directory (safe to re-run).

    python scripts/fetch_training_data.py

- ./synrs3d/<folder>/{opt,gt_nDSM,gt_ss_mask}: SynRS3D folders rich in high-rises
  and hilly terrain (the zips have no top-level folder, so each one is unpacked
  into its own directory; its name carries the GSD range the loader needs).
- ./gamus/{images,heights,classes}/train: every GAMUS train tile (existing val and
  test downloads are kept). Files already present are skipped.

Both run in parallel; failures are reported and skipped, never fatal, so the
training that follows always starts with whatever arrived.
"""

from __future__ import annotations

import shutil
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SYN_FOLDERS = ["terrain_g1_high_v1", "terrain_g05_mid_v1", "terrain_g05_high_v1", "grid_g05_high_v1"]
SYN_KEEP = ("opt/", "gt_nDSM/", "gt_ss_mask/")


def fetch_synrs3d(out: Path = Path("synrs3d")) -> None:
    out.mkdir(exist_ok=True)
    for stray in ("opt", "pre_opt", "gt_nDSM", "gt_ss_mask", "gt_cd_mask", "train.txt"):
        target = out / stray  # left by an earlier version that unpacked every zip into one folder
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for name in SYN_FOLDERS:
        dest = out / name
        if (dest / ".complete").exists():
            print("synrs3d: have", name, flush=True)
            continue
        try:
            z = hf_hub_download("JTRNEO/SynRS3D", f"SynRS3D/{name}.zip", repo_type="dataset",
                                local_dir="synrs3d_zips")
            print("synrs3d: extracting", name, flush=True)
            with zipfile.ZipFile(z) as f:
                f.extractall(dest, members=[m for m in f.namelist() if m.startswith(SYN_KEEP)])
            (dest / ".complete").touch()
            print("synrs3d: DONE", name, flush=True)
        except Exception as exc:  # keep going: training uses whatever arrived
            print("synrs3d: FAILED", name, exc, flush=True)


def fetch_gamus_train(out: Path = Path("gamus")) -> None:
    repo = "earthflow/GAMUS"
    paths = defaultdict(dict)
    for f in HfApi().list_repo_files(repo, repo_type="dataset"):
        parts = f.split("/")
        if (len(parts) == 3 and parts[0] in ("images", "heights", "classes")
                and parts[1] == "train" and f.endswith(".h5")):
            paths[parts[2].rsplit("_", 1)[0]][parts[0]] = f
    want = [p for v in paths.values() if len(v) == 3 for p in v.values()]
    print("gamus: train files", len(want), flush=True)

    def get(p: str) -> None:
        err = None
        for _ in range(3):
            try:
                hf_hub_download(repo, p, repo_type="dataset", local_dir=str(out))
                return
            except Exception as exc:
                err = exc
        print("gamus: FAILED", p, err, flush=True)

    with ThreadPoolExecutor(32) as ex:
        for n, _ in enumerate(ex.map(get, want)):
            if n % 1000 == 0:
                print("gamus:", n, "/", len(want), flush=True)
    print("gamus: DONE", flush=True)


if __name__ == "__main__":
    t = threading.Thread(target=fetch_gamus_train)
    t.start()
    fetch_synrs3d()
    t.join()
    print("ALL DONE", flush=True)
