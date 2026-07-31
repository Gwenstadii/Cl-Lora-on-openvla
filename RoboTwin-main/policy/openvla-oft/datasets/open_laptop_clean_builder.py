from typing import Iterator, Tuple, Any
import glob
import numpy as np
import h5py
import tensorflow_datasets as tfds
from datasets.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    for path in paths:
        with h5py.File(path, "r") as f:
            if not all(k in f for k in ["/relative_action","/head_camera_image",
                "/left_wrist_image","/right_wrist_image","/low_cam_image","/action","/seen"]):
                continue
            T = f["/action"].shape[0]
            actions = f["/action"][1:].astype(np.float32)
            head = f["/head_camera_image"][:T-1].astype(np.uint8)
            left = f["/left_wrist_image"][:T-1].astype(np.uint8)
            right = f["/right_wrist_image"][:T-1].astype(np.uint8)
            low = f["/low_cam_image"][:T-1].astype(np.uint8)
            states = f["/action"][:T-1].astype(np.float32)
            seen = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["/seen"][()]]
            T -= 1
            if not (head.shape[0]==left.shape[0]==right.shape[0]==low.shape[0]==T==states.shape[0]):
                continue
            steps = []
            for i in range(T):
                steps.append({
                    "observation": {"image": head[i], "left_wrist_image": left[i],
                        "right_wrist_image": right[i], "low_cam_image": low[i], "state": states[i]},
                    "action": actions[i], "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i==T-1 else 0.0),
                    "is_first": np.bool_(i==0), "is_last": np.bool_(i==T-1),
                    "is_terminal": np.bool_(i==T-1), "language_instruction": seen,
                })
            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}


class aloha_open_laptop_clean(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "RoboTwin open_laptop clean"}
    N_WORKERS = 1; MAX_PATHS_IN_MEMORY = 100; PARSE_FCN = _generate_examples

    def _info(self):
        return self.dataset_info_from_configs(features=tfds.features.FeaturesDict({
            "steps": tfds.features.Dataset({
                "observation": tfds.features.FeaturesDict({
                    "image": tfds.features.Image(shape=(256,256,3), dtype=np.uint8, encoding_format="jpeg"),
                    "left_wrist_image": tfds.features.Image(shape=(256,256,3), dtype=np.uint8, encoding_format="jpeg"),
                    "right_wrist_image": tfds.features.Image(shape=(256,256,3), dtype=np.uint8, encoding_format="jpeg"),
                    "low_cam_image": tfds.features.Image(shape=(256,256,3), dtype=np.uint8, encoding_format="jpeg"),
                    "state": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                }),
                "action": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                "discount": tfds.features.Scalar(dtype=np.float32),
                "reward": tfds.features.Scalar(dtype=np.float32),
                "is_first": tfds.features.Scalar(dtype=np.bool_),
                "is_last": tfds.features.Scalar(dtype=np.bool_),
                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                "language_instruction": tfds.features.Sequence(tfds.features.Text()),
            }),
            "episode_metadata": tfds.features.FeaturesDict({"file_path": tfds.features.Text()}),
        }))

    def _split_paths(self):
        base = "../../data/open_laptop/processed_openvla"
        return {"train": glob.glob(f"{base}/train/*.hdf5"), "val": glob.glob(f"{base}/val/*.hdf5")}


if __name__ == "__main__":
    aloha_open_laptop_clean().download_and_prepare()
