from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from scripts import build_robotwin_prototype_buffer_v2 as builder


def test_quaternion_rotation_vector_uses_shortest_wxyz_rotation() -> None:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    quarter_turn_z = np.asarray(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)], dtype=np.float32
    )
    rotation = builder._quaternion_rotation_vector(identity, quarter_turn_z)
    np.testing.assert_allclose(rotation, [0.0, 0.0, math.pi / 2.0], atol=1e-5)
    np.testing.assert_allclose(
        builder._quaternion_rotation_vector(identity, -quarter_turn_z), rotation, atol=1e-5
    )


def test_raw_episode_alignment_matches_process_data_one_frame_shift(tmp_path) -> None:
    path = tmp_path / "episode0.hdf5"
    frames = 6
    left_arm = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6) / 100.0
    right_arm = left_arm + 1.0
    left_gripper = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    right_gripper = 1.0 - left_gripper
    pose = np.zeros((frames, 7), dtype=np.float32)
    pose[:, 3] = 1.0
    with h5py.File(path, "w") as h5:
        joint = h5.create_group("joint_action")
        joint.create_dataset("left_arm", data=left_arm)
        joint.create_dataset("right_arm", data=right_arm)
        joint.create_dataset("left_gripper", data=left_gripper)
        joint.create_dataset("right_gripper", data=right_gripper)
        endpose = h5.create_group("endpose")
        endpose.create_dataset("left_endpose", data=pose)
        endpose.create_dataset("right_endpose", data=pose)
        endpose.create_dataset("left_gripper", data=left_gripper)
        endpose.create_dataset("right_gripper", data=right_gripper)

    raw = builder._load_raw_episode(path, "wxyz")
    builder._validate_raw_alignment(raw, raw.joints[:-1], atol=1e-6, rtol=1e-6)
    builder._validate_raw_alignment(raw, raw.joints[:-2], atol=1e-6, rtol=1e-6)


def test_kinematic_segmentation_detects_persistent_translation_phase() -> None:
    raw_frames = 36
    poses = np.zeros((raw_frames, 2, 7), dtype=np.float32)
    poses[..., 3] = 1.0
    poses[10:25, 0, 0] = np.linspace(0.0, 0.28, 15, dtype=np.float32)
    poses[25:, 0, 0] = 0.28
    raw = builder.RawEpisode(
        path=Path("synthetic.hdf5"),
        joints=np.zeros((raw_frames, 14), dtype=np.float32),
        poses=poses,
        grippers=np.zeros((raw_frames, 2), dtype=np.float32),
    )
    signals = builder._compute_motion_signals(
        raw,
        raw_frames - 1,
        start_offset=0,
        window=5,
        translation_threshold=0.03,
        rotation_threshold=0.05,
        gripper_threshold=0.1,
        descriptor_clip=5.0,
    )
    assert np.any(signals.modes[:, 0] == 1)
    segments = builder._build_kinematic_segments(
        signals,
        persistence_frames=3,
        min_boundary_gap=3,
        min_segment_len=3,
        max_segment_len=0,
    )
    assert segments[0][0] == 0
    assert segments[-1][1] == raw_frames - 1
    assert all(first[1] == second[0] for first, second in zip(segments, segments[1:], strict=True))
    assert len(segments) >= 2


def test_temporal_k_center_improves_set_coverage() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.70, 0.70],
            [0.0, 1.0],
            [-0.70, 0.70],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = builder._select_coverage_representatives(
        features,
        top_k=3,
        temporal_min_gap=1,
        outlier_mad_scale=0.0,
    )
    assert len(result.selected_indices) == 3
    assert len(set(result.selected_indices)) == 3
    assert result.coverage_error < result.prototype_error
    assert all(gain >= -1e-6 for gain in result.coverage_gains)
