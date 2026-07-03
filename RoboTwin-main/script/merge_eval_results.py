import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Parallel eval result directory containing worker summaries.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summaries = sorted(run_dir.glob("worker_*/_summary.json"))
    if not summaries:
        summaries = sorted(run_dir.glob("_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No _summary.json files found under {run_dir}")

    rows = []
    for path in summaries:
        with open(path, "r", encoding="utf-8") as file:
            item = json.load(file)
        item["_summary_path"] = str(path)
        rows.append(item)

    total_successes = sum(int(row["successes"]) for row in rows)
    total_episodes = sum(int(row["episodes"]) for row in rows)
    success_rate = total_successes / total_episodes if total_episodes else 0.0

    merged = {
        "run_dir": str(run_dir),
        "num_workers": len(rows),
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": success_rate,
        "workers": rows,
    }

    with open(run_dir / "_merged_result.json", "w", encoding="utf-8") as file:
        json.dump(merged, file, indent=2)

    with open(run_dir / "_merged_result.txt", "w", encoding="utf-8") as file:
        file.write(f"Successes: {total_successes}/{total_episodes}\n")
        file.write(f"Success rate: {success_rate:.6f}\n")
        for row in rows:
            file.write(
                f"worker {row['eval_worker_id']}: "
                f"{row['successes']}/{row['episodes']} = {row['success_rate']:.6f}, "
                f"start_seed={row['start_seed']}, stride={row['seed_stride']}\n"
            )

    print(f"Merged success rate: {total_successes}/{total_episodes} = {success_rate:.4f}")
    print(f"Merged files written to: {run_dir}")


if __name__ == "__main__":
    main()
