#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 to LeRobot Pipeline: Clean + Convert
LeRobot 数据集格式参考 Pi0.5 版本
"""

import os
import cv2
import json
import glob
import argparse
import shutil
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
from datasets import Dataset, Features, Image as ImageFeature, Value, Sequence
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


# ============================================================================
# SECTION 1: Data Cleaning Functions
# ============================================================================


def parse_ee_pose(ee_pose):
    """
    自动识别ee_pose维度并提取组件
    Returns: pos (T,3), rot (T,3 or 4), gripper (T,)
    """
    dim = ee_pose.shape[1]
    pos = ee_pose[:, :3]
    # print(ee_pose.shape)
    if dim == 7:
        # [x, y, z, rx, ry, rz, gripper]
        return pos, ee_pose[:, 3:6], ee_pose[:, 6]
    else:
        # [x, y, z, qx, qy, qz, qw, gripper]
        return pos, ee_pose[:, 3:7], ee_pose[:, 7]


def analyze_episode_motion(ee_pose, window_size=5):
    T = len(ee_pose)
    motion_score = np.zeros(T)
    pos, rot, gripper = parse_ee_pose(ee_pose)

    for i in range(T):
        start, end = max(0, i - window_size), min(T, i + window_size + 1)
        pos_var = np.var(pos[start:end], axis=0).sum()
        rot_var = np.var(rot[start:end], axis=0).sum()
        gripper_var = np.var(gripper[start:end])
        motion_score[i] = pos_var + rot_var + gripper_var * 10
    return motion_score


def detect_static_segments_advanced(
    ee_pose,
    gripper_threshold=0.0005,
    pos_threshold=0.001,
    rot_threshold=0.01,
    min_static_frames=10,
    motion_score_threshold=0.0001,
):
    T = len(ee_pose)
    if T == 0:
        return np.array([], dtype=bool)

    motion_score = analyze_episode_motion(ee_pose)
    pos, rot, gripper = parse_ee_pose(ee_pose)

    pos_delta = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    rot_delta = np.linalg.norm(np.diff(rot, axis=0), axis=1)
    gripper_delta = np.abs(np.diff(gripper, axis=0))

    is_static = np.concatenate(
        [
            [False],
            (pos_delta < pos_threshold)
            & (rot_delta < rot_threshold)
            & (gripper_delta < gripper_threshold),
        ]
    )
    low_motion = motion_score < motion_score_threshold
    is_abnormal = is_static & low_motion

    mask = np.ones(T, dtype=bool)
    i = 0
    while i < T:
        if is_abnormal[i]:
            j = i
            while j < T and is_abnormal[j]:
                j += 1
            if (j - i) >= min_static_frames:
                if not (
                    (i > 0 and motion_score[i - 1] > motion_score_threshold * 2)
                    and (j < T and motion_score[j] > motion_score_threshold * 2)
                ):
                    mask[i:j] = False
            i = j
        else:
            i += 1
    return mask


def filter_hdf5_file(
    input_path,
    output_path,
    cleaning_params: Dict,
    fps=10.0,
):
    """过滤单个HDF5文件"""
    try:
        with h5py.File(input_path, "r") as f_in:
            ee_pose = f_in["observations"]["ee_pose"][:]
            original_length = len(ee_pose)

            # 检测并生成mask
            mask = detect_static_segments_advanced(
                ee_pose,
                gripper_threshold=cleaning_params["gripper_threshold"],
                pos_threshold=cleaning_params["pos_threshold"],
                rot_threshold=cleaning_params["rot_threshold"],
                min_static_frames=cleaning_params["min_static_frames"],
            )

            filtered_length = np.sum(mask)

            if filtered_length < cleaning_params["min_episode_length"]:
                return False, original_length, 0

            # 创建输出文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with h5py.File(output_path, "w") as f_out:
                # 1. 重新生成连续的timestamp
                new_timestamp = np.arange(filtered_length, dtype=np.float32) / fps
                f_out.create_dataset("timestamp", data=new_timestamp)

                # 2. 过滤其他top-level数据
                for key in ["stage", "joint_action"]:
                    if key in f_in:
                        data = f_in[key][:]
                        f_out.create_dataset(key, data=data[mask])

                # 3. 过滤observations组
                obs_group = f_out.create_group("observations")
                f_in_obs = f_in["observations"]

                for key in f_in_obs.keys():
                    if key == "images":
                        img_group = obs_group.create_group("images")
                        for img_key in f_in_obs["images"].keys():
                            img_data = f_in_obs["images"][img_key][:]
                            img_group.create_dataset(img_key, data=img_data[mask])
                    elif key == "robot_base_pose_in_world":
                        data = f_in_obs[key][:]
                        obs_group.create_dataset(key, data=data[mask])
                    else:
                        data = f_in_obs[key][:]
                        obs_group.create_dataset(key, data=data[mask])

            return True, original_length, filtered_length

    except Exception as e:
        print(f"    [ERROR] {str(e)}")
        return False, 0, 0


def clean_hdf5_dataset(
    input_path: str, output_path: str, cleaning_params: Dict, fps: float
):
    """清洗单个数据集的所有HDF5文件"""

    # 查找所有HDF5文件
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        patterns = ["**/*.h5", "**/*.hdf5"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(input_path, p), recursive=True))
        files = sorted(files)

    if not files:
        print(f"[WARNING] No HDF5 files found in: {input_path}")
        return 0, 0, 0

    print(f"\n{'=' * 80}")
    print(f"Cleaning Dataset: {input_path}")
    print(f"{'=' * 80}")
    print(f"Found {len(files)} HDF5 files")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}\n")

    stats = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "original_frames": 0,
        "filtered_frames": 0,
    }

    for i, file_path in enumerate(tqdm(files, desc="Cleaning")):
        rel_path = os.path.relpath(file_path, input_path)
        out_file = os.path.join(output_path, rel_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        success, orig_len, filt_len = filter_hdf5_file(
            file_path,
            out_file,
            cleaning_params,
            fps=fps,
        )

        if success:
            if filt_len > 0:
                stats["success"] += 1
                stats["original_frames"] += orig_len
                stats["filtered_frames"] += filt_len
            else:
                stats["skipped"] += 1
        else:
            stats["error"] += 1

    print(f"\n{'=' * 80}")
    print("Cleaning Summary")
    print(f"{'=' * 80}")
    print(f"Successfully cleaned: {stats['success']}")
    print(f"Skipped (too short):  {stats['skipped']}")
    print(f"Errors:               {stats['error']}")
    print(f"Original frames:      {stats['original_frames']}")
    print(f"Filtered frames:      {stats['filtered_frames']}")
    if stats["original_frames"] > 0:
        kept_ratio = stats["filtered_frames"] / stats["original_frames"] * 100
        print(f"Kept ratio:           {kept_ratio:.1f}%")
    print(f"{'=' * 80}\n")

    return stats["success"], stats["filtered_frames"], stats["error"]


# ============================================================================
# SECTION 2: LeRobot Conversion Functions
# ============================================================================


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    """Wrap angle delta to [-π, π]."""
    return (delta + np.pi) % (2 * np.pi) - np.pi


def get_euler_from_pose(pose_row: np.ndarray) -> np.ndarray:
    """
    根据维度自动转换旋转表示为欧拉角。
    Args:
        pose_row: 7位 [x,y,z,rx,ry,rz,g] 或 8位 [x,y,z,qx,qy,qz,qw,g]
    """
    dim = len(pose_row)
    if dim == 7:
        return R.from_rotvec(pose_row[3:6]).as_euler("xyz", degrees=False)
    elif dim == 8:
        return R.from_quat(pose_row[3:7]).as_euler("xyz", degrees=False)
    else:
        raise ValueError(f"Unsupported ee_pose dimension: {dim}")


def compute_actions_from_ee_pose(ee_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ee_pose trajectory to LeRobot format, auto-detecting 7D or 8D input.
    """
    T = len(ee_pose)
    dim = ee_pose.shape[1]

    # 1. 自动提取旋转并转为欧拉角
    euler_angles = np.array([get_euler_from_pose(ee_pose[i]) for i in range(T)])

    # 2. 确定夹爪索引
    gripper_idx = 6 if dim == 7 else 7
    gripper_open_threshold = 0.04

    # 3. Build state (7D)
    states = np.zeros((T, 7), dtype=np.float32)
    states[:, :3] = ee_pose[:, :3]
    states[:, 3:6] = euler_angles
    states[:, 6] = ee_pose[:, gripper_idx]

    # 4. Build actions (7D)
    actions = np.zeros((T, 7), dtype=np.float32)

    for t in range(T - 1):
        actions[t, :3] = ee_pose[t + 1, :3] - ee_pose[t, :3]
        delta_rot = euler_angles[t + 1] - euler_angles[t]
        actions[t, 3:6] = wrap_angle_delta(delta_rot)
        actions[t, 6] = (
            1.0 if ee_pose[t + 1, gripper_idx] >= gripper_open_threshold else 0.0
        )

    if T > 1:
        actions[-1] = actions[-2]

    return states, actions


def load_hdf5(hdf5_path: str):
    """Load HDF5 file with flexible structure support."""
    with h5py.File(hdf5_path, "r") as f:
        if "observations" not in f or "ee_pose" not in f["observations"]:
            raise KeyError(f"Cannot find observations/ee_pose in {hdf5_path}")
        ee_pose = f["observations"]["ee_pose"][:]

        # Load images
        front_imgs = None
        wrist_imgs = None
        left_imgs = None

        if "observations" in f and "images" in f["observations"]:
            imgs = f["observations"]["images"]
            # image       ← camera_left_color  (正前方全景, 对应原始 1.mp4)
            # wrist_image ← camera_wrist_color  (腕部俯视,   对应原始 0.mp4)
            # camera_front_color 是侧后相机 (2.mp4), 不使用
            if "camera_left_color" in imgs:
                front_imgs = imgs["camera_left_color"][:]
            elif "camera_front_color" in imgs:
                front_imgs = imgs["camera_front_color"][:]
            if "camera_wrist_color" in imgs:
                wrist_imgs = imgs["camera_wrist_color"][:]
            if front_imgs is None:
                raise KeyError(f"Cannot find camera_left_color in {hdf5_path}")
        # wrist camera is optional — missing ones will be filled with black frames

        # Load timestamps
        timestamps = None
        if "timestamp" in f:
            timestamps = f["timestamp"][:]
        elif "observations" in f and "timestamp" in f["observations"]:
            timestamps = f["observations"]["timestamp"][:]

    # Truncate to minimum length across available streams
    T = len(ee_pose)
    if front_imgs is not None:
        T = min(T, len(front_imgs))
    if wrist_imgs is not None:
        T = min(T, len(wrist_imgs))
    if timestamps is not None:
        T = min(T, len(timestamps))

    front_imgs = front_imgs[:T]
    # Fill missing optional cameras with black frames matching front camera shape
    black = np.zeros_like(front_imgs)
    wrist_imgs = wrist_imgs[:T] if wrist_imgs is not None else black

    return (
        ee_pose[:T],
        front_imgs,
        wrist_imgs,
        timestamps[:T] if timestamps is not None else None,
    )


def build_episode_data(
    ee_pose: np.ndarray,
    front_imgs: np.ndarray,
    wrist_imgs: np.ndarray,
    timestamps: Optional[np.ndarray],
    episode_index: int,
    fps: float,
    image_height: int,
    image_width: int,
    task_index: int = 0,
    frame_offset: int = 0,
) -> dict:
    """Build episode data as dict for HuggingFace datasets in LeRobot v2.1 format."""

    T = len(ee_pose)
    states, actions = compute_actions_from_ee_pose(ee_pose)

    if timestamps is None:
        timestamps = np.arange(T, dtype=np.float32) / float(fps)
    else:
        timestamps = (timestamps - timestamps[0]).astype(np.float32)

    num_workers = min(8, mp.cpu_count())

    def resize_batch(imgs, h, w):
        def resize_single(img):
            resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            return Image.fromarray(resized)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(resize_single, imgs))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            "front": executor.submit(
                resize_batch, front_imgs, image_height, image_width
            ),
            "wrist": executor.submit(
                resize_batch, wrist_imgs, image_height, image_width
            ),
        }
        front_resized = futures["front"].result()
        wrist_resized = futures["wrist"].result()

    data = {
        "image": front_resized,
        "wrist_image": wrist_resized,
        "state": states.tolist(),
        "actions": actions.tolist(),
        "timestamp": timestamps.tolist(),
        "frame_index": np.arange(T, dtype=np.int64).tolist(),
        "episode_index": np.full(T, episode_index, dtype=np.int64).tolist(),
        "index": np.arange(frame_offset, frame_offset + T, dtype=np.int64).tolist(),
        "task_index": np.full(T, task_index, dtype=np.int64).tolist(),
    }

    return data


def _image_channel_stats(pil_images: list) -> dict:
    """Compute per-channel min/max/mean/std over a list of PIL images, normalized to [0,1]."""
    n = len(pil_images)
    C = 3
    ch_min = np.full(C, np.inf, dtype=np.float64)
    ch_max = np.full(C, -np.inf, dtype=np.float64)
    ch_sum = np.zeros(C, dtype=np.float64)
    ch_sum_sq = np.zeros(C, dtype=np.float64)

    for img in pil_images:
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)
        for c in range(C):
            ch = arr[:, :, c]
            ch_min[c] = min(ch_min[c], float(ch.min()))
            ch_max[c] = max(ch_max[c], float(ch.max()))
            ch_sum[c] += float(ch.mean())
            ch_sum_sq[c] += float((ch**2).mean())

    ch_mean = ch_sum / n
    ch_std = np.sqrt(np.maximum(ch_sum_sq / n - ch_mean**2, 0.0))

    def nested(arr):
        return [[[float(v)]] for v in arr]

    return {
        "min": nested(ch_min),
        "max": nested(ch_max),
        "mean": nested(ch_mean),
        "std": nested(ch_std),
        "count": [n],
    }


def compute_episode_stats(data: dict, episode_index: int) -> dict:
    """Compute episodes_stats record for one episode."""
    n = len(data["timestamp"])

    def vec_stats(key):
        arr = np.array(data[key], dtype=np.float64)  # (N, D)
        return {
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "count": [n],
        }

    def scalar_stats(key):
        arr = np.array(data[key], dtype=np.float64)
        return {
            "min": [float(arr.min())],
            "max": [float(arr.max())],
            "mean": [float(arr.mean())],
            "std": [float(arr.std())],
            "count": [n],
        }

    img_stats = _image_channel_stats(data["image"])
    wrist_stats = _image_channel_stats(data["wrist_image"])

    return {
        "episode_index": episode_index,
        "stats": {
            "observation.images.image": img_stats,
            "observation.images.wrist_image": wrist_stats,
            "image": img_stats,
            "wrist_image": wrist_stats,
            "state": vec_stats("state"),
            "actions": vec_stats("actions"),
            "timestamp": scalar_stats("timestamp"),
            "frame_index": scalar_stats("frame_index"),
            "episode_index": scalar_stats("episode_index"),
            "index": scalar_stats("index"),
            "task_index": scalar_stats("task_index"),
        },
    }


def append_episode_stats(meta_dir: str, stats_record: dict):
    os.makedirs(meta_dir, exist_ok=True)
    path = os.path.join(meta_dir, "episodes_stats.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats_record) + "\n")


def save_episode_with_datasets(data: dict, out_path: str):
    """Save episode using HuggingFace datasets library."""
    features = Features(
        {
            "image": ImageFeature(),
            "wrist_image": ImageFeature(),
            "state": Sequence(Value("float32"), length=7),
            "actions": Sequence(Value("float32"), length=7),
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
        }
    )

    dataset = Dataset.from_dict(data, features=features)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset.to_parquet(out_path)


def _encode_video_from_pil(
    frames: list,
    path: str,
    fps: float,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
):
    """Encode a list of PIL images to MP4 using ffmpeg."""
    import subprocess

    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].size[1], frames[0].size[0]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-vcodec",
        codec,
        "-pix_fmt",
        pix_fmt,
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.array(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def write_tasks_jsonl(meta_dir: str, all_tasks: List[str]):
    os.makedirs(meta_dir, exist_ok=True)
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")
    with open(tasks_path, "w", encoding="utf-8") as f:
        for task_idx, task_text in enumerate(all_tasks):
            f.write(
                json.dumps(
                    {"task_index": task_idx, "task": task_text}, ensure_ascii=False
                )
                + "\n"
            )


def append_episode_meta(meta_dir: str, episode_index: int, length: int, task_text: str):
    os.makedirs(meta_dir, exist_ok=True)
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    rec = {"episode_index": episode_index, "tasks": [task_text], "length": length}
    with open(episodes_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_info_json(
    meta_dir: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: float,
    chunk_size: int,
    image_height: int,
    image_width: int,
):
    video_info = {
        "video.height": int(image_height),
        "video.width": int(image_width),
        "video.codec": "libx264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": float(fps),
        "video.channels": 3,
        "has_audio": False,
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_panda",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "total_videos": int(total_episodes * 2),
        "total_chunks": int((total_episodes + chunk_size - 1) // chunk_size),
        "chunks_size": int(chunk_size),
        "fps": float(fps),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": {
                "names": ["channel", "height", "width"],
                "dtype": "video",
                "shape": [3, int(image_height), int(image_width)],
                "info": video_info,
            },
            "observation.images.wrist_image": {
                "names": ["channel", "height", "width"],
                "dtype": "video",
                "shape": [3, int(image_height), int(image_width)],
                "info": video_info,
            },
            "image": {
                "dtype": "image",
                "shape": [int(image_height), int(image_width), 3],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": [int(image_height), int(image_width), 3],
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": [7],
                "names": ["ee_pose_and_gripper_width"],
            },
            "actions": {
                "dtype": "float32",
                "shape": [7],
                "names": ["delta_ee_pose_and_gripper_action"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def convert_cleaned_dataset(
    cleaned_path: str,
    output_root: str,
    task_text: str,
    task_index: int,
    episode_offset: int,
    fps: float,
    image_height: int,
    image_width: int,
    chunk_size: int,
    episode_offset_frames: int = 0,
) -> Tuple[int, int, List[str]]:
    """
    Convert a single cleaned dataset to LeRobot format.
    Returns: (num_episodes, num_frames, errors)
    """

    # Find all cleaned HDF5 files
    if os.path.isfile(cleaned_path):
        files = [cleaned_path]
    else:
        patterns = ["**/*.h5", "**/*.hdf5"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(cleaned_path, p), recursive=True))
        files = sorted(files)

    if not files:
        print(f"[WARNING] No cleaned HDF5 files found in: {cleaned_path}")
        return 0, 0, []

    print(f"\n{'=' * 80}")
    print(f"Converting Dataset: {cleaned_path}")
    print(f"{'=' * 80}")
    print(f"Task: {task_text}")
    print(f"Files: {len(files)}")
    print(f"{'=' * 80}\n")

    data_root = os.path.join(output_root, "data")
    meta_root = os.path.join(output_root, "meta")

    total_frames = 0
    errors = []

    # Clear episodes_stats.jsonl for this run (only if starting from offset 0)
    if episode_offset == 0:
        stats_path = os.path.join(meta_root, "episodes_stats.jsonl")
        if os.path.exists(stats_path):
            os.remove(stats_path)

    for local_idx, h5p in enumerate(tqdm(files, desc="Converting")):
        try:
            ep_idx = episode_offset + local_idx
            chunk_id = ep_idx // chunk_size
            chunk_dir = os.path.join(data_root, f"chunk-{chunk_id:03d}")
            os.makedirs(chunk_dir, exist_ok=True)

            # Load data
            ee_pose, front_imgs, wrist_imgs, timestamps = load_hdf5(h5p)

            # frame_offset tracks the global frame index start for this episode
            frame_offset = episode_offset_frames + total_frames

            data = build_episode_data(
                ee_pose=ee_pose,
                front_imgs=front_imgs,
                wrist_imgs=wrist_imgs,
                timestamps=timestamps,
                episode_index=ep_idx,
                fps=fps,
                image_height=image_height,
                image_width=image_width,
                task_index=task_index,
                frame_offset=frame_offset,
            )

            episode_length = len(data["timestamp"])

            # Save parquet
            parquet_path = os.path.join(chunk_dir, f"episode_{ep_idx:06d}.parquet")
            save_episode_with_datasets(data, parquet_path)

            # Compute and append episodes_stats.jsonl
            stats_record = compute_episode_stats(data, ep_idx)
            append_episode_stats(meta_root, stats_record)

            # Save videos
            # observation.images.image       ← front camera  (data["image"],       camera_left_color)
            # observation.images.wrist_image ← wrist camera  (data["wrist_image"], camera_wrist_color)
            video_chunk_dir = os.path.join(
                output_root, "videos", f"chunk-{chunk_id:03d}"
            )
            _encode_video_from_pil(
                data["image"],
                os.path.join(
                    video_chunk_dir,
                    f"observation.images.image/episode_{ep_idx:06d}.mp4",
                ),
                fps=fps,
            )
            _encode_video_from_pil(
                data["wrist_image"],
                os.path.join(
                    video_chunk_dir,
                    f"observation.images.wrist_image/episode_{ep_idx:06d}.mp4",
                ),
                fps=fps,
            )

            # Append to episodes.jsonl
            append_episode_meta(
                meta_root, ep_idx, length=episode_length, task_text=task_text
            )

            total_frames += episode_length

        except Exception as e:
            error_msg = f"Episode {ep_idx} ({os.path.basename(h5p)}): {str(e)}"
            errors.append(error_msg)
            print(f"\n  [✗] ERROR: {error_msg}")

    return len(files) - len(errors), total_frames, errors


# ============================================================================
# SECTION 3: Main Pipeline
# ============================================================================


def run_pipeline(
    config_path: str, skip_cleaning: bool = False, skip_conversion: bool = False
):
    """运行完整的清洗+转换流程"""

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("\n" + "=" * 80)
    print("HDF5 to LeRobot Pipeline")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Skip cleaning: {skip_cleaning}")
    print(f"Skip conversion: {skip_conversion}")
    print("=" * 80 + "\n")

    # Extract parameters
    output_root = config["output_root"]
    fps = config.get("fps", 15)
    image_height = config.get("image_height", 480)
    image_width = config.get("image_width", 640)
    chunk_size = config.get("chunk_size", 1000)
    datasets = config["datasets"]

    # TODO: Cleaning parameters
    cleaning_params = config.get(
        "cleaning",
        {
            "gripper_threshold": 0.0005,
            "pos_threshold": 0.0005,
            "rot_threshold": 0.005,
            "min_static_frames": 5,
            "min_episode_length": 20,
        },
    )

    # Create temp directory for cleaned data
    temp_clean_root = config.get(
        "temp_clean_dir", os.path.join(output_root, "_temp_cleaned")
    )

    # Collect all unique tasks
    all_tasks = []
    for ds in datasets:
        task = ds["task"]
        if task not in all_tasks:
            all_tasks.append(task)

    print(f"Unique tasks: {len(all_tasks)}")
    for i, task in enumerate(all_tasks):
        print(f"  Task {i}: {task}")
    print()

    # Prepare output directories
    os.makedirs(output_root, exist_ok=True)
    data_root = os.path.join(output_root, "data")
    meta_root = os.path.join(output_root, "meta")
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)

    # Write tasks.jsonl
    write_tasks_jsonl(meta_root, all_tasks)

    # Clear episodes.jsonl
    episodes_jsonl = os.path.join(meta_root, "episodes.jsonl")
    if os.path.exists(episodes_jsonl):
        os.remove(episodes_jsonl)

    # Process each dataset
    total_episodes = 0
    total_frames = 0
    all_errors = []

    for ds_idx, ds_config in enumerate(datasets):
        input_path = ds_config["path"]
        task_text = ds_config["task"]
        task_index = all_tasks.index(task_text)

        print(f"\n{'#' * 80}")
        print(f"Dataset {ds_idx + 1}/{len(datasets)}")
        print(f"{'#' * 80}")
        print(f"Input: {input_path}")
        print(f"Task: {task_text}")
        print(f"{'#' * 80}\n")

        # Step 1: Clean data
        if not skip_cleaning:
            cleaned_path = os.path.join(temp_clean_root, f"dataset_{ds_idx}")
            num_cleaned, num_frames_cleaned, num_errors = clean_hdf5_dataset(
                input_path=input_path,
                output_path=cleaned_path,
                cleaning_params=cleaning_params,
                fps=fps,
            )

            if num_cleaned == 0:
                print(f"[WARNING] No files successfully cleaned for dataset {ds_idx}")
                continue
        else:
            cleaned_path = input_path
            print(f"[INFO] Skipping cleaning, using raw data from: {input_path}")

        # Step 2: Convert to LeRobot format
        if not skip_conversion:
            num_episodes, num_frames, errors = convert_cleaned_dataset(
                cleaned_path=cleaned_path,
                output_root=output_root,
                task_text=task_text,
                task_index=task_index,
                episode_offset=total_episodes,
                fps=fps,
                image_height=image_height,
                image_width=image_width,
                chunk_size=chunk_size,
                episode_offset_frames=total_frames,
            )

            total_episodes += num_episodes
            total_frames += num_frames
            all_errors.extend(errors)

    # Write info.json
    if not skip_conversion:
        write_info_json(
            meta_root,
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=len(all_tasks),
            fps=fps,
            chunk_size=chunk_size,
            image_height=image_height,
            image_width=image_width,
        )

    # Clean up temp directory
    if not skip_cleaning and not config.get("keep_temp_cleaned", False):
        if os.path.exists(temp_clean_root):
            print(f"\n[INFO] Cleaning up temporary directory: {temp_clean_root}")
            shutil.rmtree(temp_clean_root)

    # Final summary
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"Output root: {output_root}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Errors: {len(all_errors)}")

    if all_errors:
        print(f"\n[!] {len(all_errors)} errors encountered:")
        for err in all_errors[:5]:
            print(f"  - {err}")
        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="HDF5 to LeRobot Pipeline (Clean + Convert)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace1/zhijun/RLinf/real_franka/config/pick_and_place_config.json",
        help="Path to pipeline config JSON",
    )
    parser.add_argument(
        "--skip-cleaning", action="store_true", help="Skip cleaning step"
    )
    parser.add_argument(
        "--skip-conversion", action="store_true", help="Skip conversion step"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        return

    run_pipeline(
        args.config,
        skip_cleaning=args.skip_cleaning,
        skip_conversion=args.skip_conversion,
    )


if __name__ == "__main__":
    main()
