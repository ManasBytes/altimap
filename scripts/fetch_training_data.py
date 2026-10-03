"""Download the v2 training data into the current directory (safe to re-run).

    python scripts/fetch_training_data.py

- ./synrs3d/<folder>/{opt,gt_nDSM,gt_ss_mask}: SynRS3D folders rich in high-rises
  and hilly terrain (the zips have no top-level folder, so each one is unpacked
  into its own directory; its name carries the GSD range the loader needs).
- ./gamus/{images,heights,classes}/{train,val,test}: requested GAMUS splits. Files
  already present are skipped.

Both run in parallel; failures are reported and skipped, never fatal, so the
training that follows always starts with whatever arrived.
"""

from __future__ import annotations

import shutil
import threading
import zipfile
import argparse
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


def fetch_gamus(out: Path = Path("gamus"), splits: tuple[str, ...] = ("train",)) -> None:
    """Download complete GAMUS triplets for the requested splits, skipping existing files."""
    repo = "earthflow/GAMUS"
    paths = {}
    for f in HfApi().list_repo_files(repo, repo_type="dataset"):
        parts = f.split("/")
        if (len(parts) == 3 and parts[0] in ("images", "heights", "classes")
                and parts[1] in splits and f.endswith(".h5")):
            key = (parts[1], parts[2].rsplit("_", 1)[0])
            paths.setdefault(key, {})[parts[0]] = f
    complete = [v for v in paths.values() if len(v) == 3]
    want = [p for v in complete for p in v.values()]
    print(f"gamus: splits={','.join(splits)} complete_tiles={len(complete)} "
          f"files={len(want)} incomplete_triplets={len(paths) - len(complete)}", flush=True)

    def get(p: str) -> None:
        # Avoid a Hub HEAD request for files already downloaded. This is both
        # faster on resume and prevents unauthenticated rate limiting.
        if (out / p).is_file():
            return
        err = None
        for _ in range(3):
            try:
                hf_hub_download(repo, p, repo_type="dataset", local_dir=str(out))
                return
            except Exception as exc:
                err = exc
        print("gamus: FAILED", p, err, flush=True)

    with ThreadPoolExecutor(32) as ex:
        for n, _ in enumerate(ex.map(get, want), 1):
            if n == 1 or n % 250 == 0 or n == len(want):
                print("gamus:", n, "/", len(want), flush=True)
    print("gamus: DONE", flush=True)


def fetch_gamus_train(out: Path = Path("gamus")) -> None:
    fetch_gamus(out, ("train",))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamus-only", action="store_true",
                        help="download GAMUS only; do not fetch SynRS3D")
    parser.add_argument("--gamus-root", type=Path, default=Path("gamus"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        default=("train",))
    args = parser.parse_args()
    if args.gamus_only:
        fetch_gamus(args.gamus_root, tuple(args.splits))
        raise SystemExit(0)
    t = threading.Thread(target=fetch_gamus_train)
    t.start()
    fetch_synrs3d()
    t.join()
    print("ALL DONE", flush=True)
