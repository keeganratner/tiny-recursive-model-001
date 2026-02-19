"""Download and organize ARC-format datasets for extended training.

Downloads:
  - ConceptARC (~160 tasks) from GitHub
  - Re-ARC (~1000+ tasks) from GitHub (uses generate.py to create tasks)

Merges all tasks into data/all_tasks/, creates a fixed 200-task validation
holdout (reproducible), and copies the remainder to data/train_all/.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --seed 42 --holdout-size 200
    python scripts/fetch_data.py --skip-download  # only rebuild splits

Directory layout created:
    data/
      all_tasks/       # all tasks from all sources (JSON files)
      train_all/       # all_tasks minus holdout (symlinks on Linux, copies on Windows)
      validation_holdout/  # fixed 200-task holdout set
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

project_root = Path(__file__).parent.parent
data_dir = project_root / "data"

# GitHub archive URLs (raw zip downloads – no auth required)
CONCEPTARC_URL = "https://github.com/victorvikram/ConceptARC/archive/refs/heads/main.zip"
REARC_URL = "https://github.com/michaelhodel/re-arc/archive/refs/heads/main.zip"


def download_zip(url: str, dest_dir: Path, label: str) -> Path:
    """Download a ZIP from url into dest_dir, return path to extracted folder."""
    zip_path = dest_dir / f"{label}.zip"
    if not zip_path.exists():
        print(f"Downloading {label}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Downloaded {zip_path.stat().st_size // 1024} KB")
    else:
        print(f"  {label}.zip already present, skipping download")

    extract_dir = dest_dir / label
    if not extract_dir.exists():
        print(f"  Extracting {label}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir / label)
    return extract_dir


def collect_json_tasks(source_dir: Path) -> list[Path]:
    """Recursively collect all .json files under source_dir."""
    return sorted(source_dir.rglob("*.json"))


def copy_task(src: Path, dest_dir: Path) -> bool:
    """Copy a JSON task file to dest_dir, skipping if task ID already exists.

    Returns True if copied, False if skipped (duplicate).
    """
    dest = dest_dir / src.name
    if dest.exists():
        return False
    # Quick sanity check: valid ARC task has 'train' and 'test' keys
    try:
        with open(src) as f:
            task = json.load(f)
        if "train" not in task or "test" not in task:
            return False
    except (json.JSONDecodeError, KeyError):
        return False
    shutil.copy2(src, dest)
    return True


def build_splits(all_tasks_dir: Path, train_dir: Path, holdout_dir: Path,
                 holdout_size: int, seed: int) -> tuple[int, int]:
    """Create train_all and validation_holdout directories from all_tasks.

    Uses a seeded random shuffle so the holdout split is reproducible.
    Returns (n_train, n_holdout).
    """
    all_files = sorted(all_tasks_dir.glob("*.json"))
    if len(all_files) < holdout_size:
        print(f"WARNING: only {len(all_files)} tasks available, "
              f"reducing holdout to {len(all_files) // 5}")
        holdout_size = len(all_files) // 5

    rng = random.Random(seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)

    holdout = set(f.name for f in shuffled[:holdout_size])
    train = [f for f in all_files if f.name not in holdout]

    # Clear and rebuild destination dirs
    for d in [train_dir, holdout_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for f in shuffled[:holdout_size]:
        shutil.copy2(f, holdout_dir / f.name)

    for f in train:
        shutil.copy2(f, train_dir / f.name)

    return len(train), len(shuffled[:holdout_size])


def fetch_conceptarc(tmp_dir: Path, all_tasks_dir: Path) -> int:
    """Download ConceptARC and copy tasks to all_tasks_dir. Returns count added."""
    extract_dir = download_zip(CONCEPTARC_URL, tmp_dir, "conceptarc")
    tasks = collect_json_tasks(extract_dir)
    added = 0
    for task_path in tasks:
        if copy_task(task_path, all_tasks_dir):
            added += 1
    return added


def fetch_rearc(tmp_dir: Path, all_tasks_dir: Path) -> int:
    """Download Re-ARC and copy tasks to all_tasks_dir. Returns count added.

    Re-ARC may contain a generate.py script; if tasks/ directory exists we
    use those directly, otherwise we attempt to run generate.py.
    """
    extract_dir = download_zip(REARC_URL, tmp_dir, "rearc")
    # Re-ARC stores tasks under re-arc-main/tasks/ or similar
    tasks = collect_json_tasks(extract_dir)
    added = 0
    for task_path in tasks:
        if copy_task(task_path, all_tasks_dir):
            added += 1

    # If Re-ARC has a generator, try running it for more tasks
    generator = None
    for candidate in extract_dir.rglob("generate.py"):
        generator = candidate
        break

    if generator and added == 0:
        print("  Re-ARC: no pre-generated tasks found, attempting generation...")
        gen_out = generator.parent / "generated"
        gen_out.mkdir(exist_ok=True)
        try:
            subprocess.run(
                [sys.executable, str(generator), "--out", str(gen_out)],
                check=True, timeout=300
            )
            for task_path in collect_json_tasks(gen_out):
                if copy_task(task_path, all_tasks_dir):
                    added += 1
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  Re-ARC generation failed: {e}")

    return added


def copy_existing_arc(all_tasks_dir: Path, include_evaluation: bool = True) -> int:
    """Copy existing ARC-AGI-1 tasks from data/training and optionally data/evaluation.

    Args:
        all_tasks_dir: Destination directory for all tasks
        include_evaluation: If True, also copies ARC-AGI-1 evaluation tasks.
                           Set to False (via --no-arc-eval) to keep the evaluation
                           split clean as a held-out test set.
    """
    added = 0
    splits = ["training", "evaluation"] if include_evaluation else ["training"]
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for task_path in sorted(split_dir.glob("*.json")):
            if copy_task(task_path, all_tasks_dir):
                added += 1
    return added


def main():
    parser = argparse.ArgumentParser(description="Fetch and organize ARC datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for holdout split (default: 42)")
    parser.add_argument("--holdout-size", type=int, default=200,
                        help="Number of tasks to hold out for validation (default: 200)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading extra datasets, only rebuild splits")
    parser.add_argument("--no-conceptarc", action="store_true",
                        help="Skip ConceptARC download")
    parser.add_argument("--no-rearc", action="store_true",
                        help="Skip Re-ARC download")
    parser.add_argument("--no-arc-eval", action="store_true",
                        help="Skip copying ARC-AGI-1 evaluation tasks into all_tasks/ "
                             "(keeps data/evaluation/ clean as a held-out test set). "
                             "Use this to match the paper's training setup: "
                             "train on ARC-train (400) + ConceptARC (~160) only.")
    args = parser.parse_args()

    tmp_dir = data_dir / "_downloads"
    all_tasks_dir = data_dir / "all_tasks"
    train_dir = data_dir / "train_all"
    holdout_dir = data_dir / "validation_holdout"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    all_tasks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fetching and organizing ARC datasets")
    print("=" * 60)

    # Step 1: Copy existing ARC-AGI-1 tasks
    print("\n[1/4] Copying existing ARC-AGI-1 tasks...")
    if args.no_arc_eval:
        print("  --no-arc-eval: skipping evaluation split (kept clean for testing)")
    arc_added = copy_existing_arc(all_tasks_dir, include_evaluation=not args.no_arc_eval)
    print(f"  Added {arc_added} ARC-AGI-1 tasks")

    if not args.skip_download:
        # Step 2: Download ConceptARC
        if not args.no_conceptarc:
            print("\n[2/4] Fetching ConceptARC...")
            try:
                concept_added = fetch_conceptarc(tmp_dir, all_tasks_dir)
                print(f"  Added {concept_added} ConceptARC tasks")
            except Exception as e:
                print(f"  WARNING: ConceptARC download failed: {e}")
                print("  Continuing without ConceptARC...")
        else:
            print("\n[2/4] ConceptARC: skipped (--no-conceptarc)")

        # Step 3: Download Re-ARC
        if not args.no_rearc:
            print("\n[3/4] Fetching Re-ARC...")
            try:
                rearc_added = fetch_rearc(tmp_dir, all_tasks_dir)
                print(f"  Added {rearc_added} Re-ARC tasks")
            except Exception as e:
                print(f"  WARNING: Re-ARC download failed: {e}")
                print("  Continuing without Re-ARC...")
        else:
            print("\n[3/4] Re-ARC: skipped (--no-rearc)")
    else:
        print("\n[2-3/4] Downloads skipped (--skip-download)")

    # Step 4: Build train_all / validation_holdout splits
    total_tasks = len(list(all_tasks_dir.glob("*.json")))
    print(f"\n[4/4] Building splits from {total_tasks} total tasks...")
    print(f"  Holdout size: {args.holdout_size}, seed: {args.seed}")

    if total_tasks == 0:
        print("ERROR: No tasks found in all_tasks/. Check data/training and data/evaluation exist.")
        sys.exit(1)

    n_train, n_holdout = build_splits(
        all_tasks_dir, train_dir, holdout_dir,
        holdout_size=args.holdout_size, seed=args.seed
    )

    print("\n" + "=" * 60)
    print("Dataset summary:")
    print(f"  Total tasks:          {total_tasks}")
    print(f"  Training tasks:       {n_train}  → data/train_all/")
    print(f"  Validation holdout:   {n_holdout} → data/validation_holdout/")
    print("=" * 60)
    print("\nNext steps:")
    print("  python -m pytest tests/test_long_run.py -v")
    print("  python scripts/train.py --in-context --deep-supervision --augment \\")
    print("    --batch-size 4 --train-split train_all --val-split evaluation \\")
    print("    --epochs 5000 --patience 50000 --save-every 250 --embed-lr 1e-4 \\")
    print("    --val-every 50 --bf16 --log-dir logs")


if __name__ == "__main__":
    main()
