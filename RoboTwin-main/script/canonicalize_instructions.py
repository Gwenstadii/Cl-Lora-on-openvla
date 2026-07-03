#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path


def atomic_write_json(path: Path, obj: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
        tmp_path = Path(f.name)

    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize RoboTwin per-episode instruction prompts."
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expect-episodes", type=int, default=50)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root) / args.task_name / args.task_config
    instruction_dir = root / "instructions"
    data_dir = root / "data"
    video_dir = root / "video"

    if not root.exists():
        raise FileNotFoundError(f"Task data directory not found: {root}")
    if not instruction_dir.exists():
        raise FileNotFoundError(f"Instruction directory not found: {instruction_dir}")

    instruction_files = sorted(instruction_dir.glob("episode*.json"))
    hdf5_files = sorted(data_dir.glob("*.hdf5")) + sorted(data_dir.glob("*.h5"))
    video_files = sorted(video_dir.glob("*.mp4"))

    print(f"Task directory: {root}")
    print(f"Instruction files: {len(instruction_files)}")
    print(f"HDF5/H5 files: {len(hdf5_files)}")
    print(f"Video files: {len(video_files)}")
    print(f"Canonical prompt: {args.prompt!r}")

    if len(instruction_files) != args.expect_episodes:
        raise RuntimeError(
            f"Expected {args.expect_episodes} instruction files, "
            f"got {len(instruction_files)}."
        )

    if len(hdf5_files) != args.expect_episodes:
        raise RuntimeError(
            f"Expected {args.expect_episodes} HDF5/H5 files, "
            f"got {len(hdf5_files)}."
        )

    if len(video_files) != args.expect_episodes:
        raise RuntimeError(
            f"Expected {args.expect_episodes} video files, "
            f"got {len(video_files)}."
        )

    before_seen = set()
    before_unseen = set()

    for path in instruction_files:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise TypeError(f"Instruction file is not a JSON object: {path}")

        before_seen.update(obj.get("seen", []))
        before_unseen.update(obj.get("unseen", []))

    print(f"Unique seen prompts before: {len(before_seen)}")
    print(f"Unique unseen prompts before: {len(before_unseen)}")

    if args.dry_run:
        print("Dry run only. No files were modified.")
        return

    for path in instruction_files:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        obj["seen"] = [args.prompt]
        obj["unseen"] = [args.prompt]

        atomic_write_json(path, obj)

    after_seen = set()
    after_unseen = set()

    for path in instruction_files:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        after_seen.update(obj.get("seen", []))
        after_unseen.update(obj.get("unseen", []))

    if after_seen != {args.prompt}:
        raise RuntimeError(f"Seen prompt verification failed: {after_seen}")
    if after_unseen != {args.prompt}:
        raise RuntimeError(f"Unseen prompt verification failed: {after_unseen}")

    print("Canonicalization complete.")
    print(f"Unique seen prompts after: {after_seen}")
    print(f"Unique unseen prompts after: {after_unseen}")


if __name__ == "__main__":
    main()