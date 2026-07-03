#!/usr/bin/env python3
import argparse
from collections import Counter

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def to_list(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def as_int(x):
    if hasattr(x, "item"):
        x = x.item()
    return int(x)


def get_episode_lengths_from_index(dataset):
    index = getattr(dataset, "episode_data_index", None)
    if not isinstance(index, dict):
        return None

    if "from" not in index or "to" not in index:
        return None

    starts = [as_int(v) for v in to_list(index["from"])]
    ends = [as_int(v) for v in to_list(index["to"])]

    if len(starts) != len(ends):
        return None

    return [end - start for start, end in zip(starts, ends)]


def get_episode_lengths_by_iterating(dataset):
    counter = Counter()

    for i in range(len(dataset)):
        item = dataset[i]
        ep_idx = item.get("episode_index", None)
        if ep_idx is None:
            return None
        counter[as_int(ep_idx)] += 1

    return [counter[k] for k in sorted(counter)]


def main():
    parser = argparse.ArgumentParser(description="Count LeRobot/OpenPI training samples.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id)
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            args.action_key: [t / meta.fps for t in range(args.action_horizon)]
        },
    )

    total_training_samples = len(dataset)

    episode_lengths = get_episode_lengths_from_index(dataset)
    source = "dataset.episode_data_index"

    if episode_lengths is None:
        episode_lengths = get_episode_lengths_by_iterating(dataset)
        source = "iterating dataset episode_index"

    print("Summary")
    print(f"repo_id: {args.repo_id}")
    print(f"fps: {meta.fps}")
    print(f"action_horizon: {args.action_horizon}")
    print(f"action_key: {args.action_key}")
    print(f"training_samples_len_dataset: {total_training_samples}")

    if episode_lengths is not None:
        print(f"episodes: {len(episode_lengths)}")
        print(f"total_episode_frames: {sum(episode_lengths)}")
        print(f"min_episode_frames: {min(episode_lengths)}")
        print(f"max_episode_frames: {max(episode_lengths)}")
        print(f"mean_episode_frames: {sum(episode_lengths) / len(episode_lengths):.2f}")
        print(f"episode_length_source: {source}")

        if args.verbose:
            print("\nEpisode lengths")
            for i, n in enumerate(episode_lengths):
                print(f"episode_{i}: {n}")
    else:
        print("episodes: unknown")
        print("total_episode_frames: unknown")
        print("episode_length_source: unavailable")


if __name__ == "__main__":
    main()