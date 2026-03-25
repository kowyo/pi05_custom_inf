#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple LeRobot v2.1 datasets into one.

Usage:
    python merge_datasets.py \
        --datasets /path/to/real /path/to/mimicgen \
        --output /path/to/merged \
        --chunk-size 1000
"""

import os
import json
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# Helpers
# ============================================================================


def read_meta(dataset_path: str):
    """Read info.json, tasks.jsonl, episodes.jsonl from a dataset."""
    meta_dir = os.path.join(dataset_path, "meta")

    with open(os.path.join(meta_dir, "info.json"), encoding="utf-8") as f:
        info = json.load(f)

    tasks = []
    with open(os.path.join(meta_dir, "tasks.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    episodes = {}
    with open(os.path.join(meta_dir, "episodes.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                episodes[rec["episode_index"]] = rec

    return info, tasks, episodes


def get_sorted_parquets(dataset_path: str):
    """Return all episode parquet paths sorted by episode index."""
    files = glob.glob(os.path.join(dataset_path, "data", "**", "episode_*.parquet"), recursive=True)
    files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
    return files


def read_episode_stats(dataset_path: str) -> dict:
    """Read episodes_stats.jsonl; returns dict keyed by episode_index, or {} if missing."""
    stats_path = os.path.join(dataset_path, "meta", "episodes_stats.jsonl")
    if not os.path.exists(stats_path):
        return {}
    stats = {}
    with open(stats_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                stats[rec["episode_index"]] = rec
    return stats


def patch_stats_record(rec: dict, new_ep_idx: int, new_frame_offset: int, T: int,
                        task_remap: dict) -> dict:
    """
    Return a copy of a stats record with episode_index, index, and task_index fields updated.

    - episode_index stats: all values become the single new episode index
    - index stats: min = new_frame_offset, max = new_frame_offset + T - 1,
                   mean = new_frame_offset + (T-1)/2, std = std of uniform range
    - task_index stats: remapped via task_remap (all values the same new task index)
    """
    import copy
    rec = copy.deepcopy(rec)
    rec["episode_index"] = new_ep_idx

    s = rec.get("stats", {})

    # Patch episode_index stats
    if "episode_index" in s:
        s["episode_index"] = {
            "min": [float(new_ep_idx)],
            "max": [float(new_ep_idx)],
            "mean": [float(new_ep_idx)],
            "std": [0.0],
            "count": [T],
        }

    # Patch index stats
    if "index" in s:
        idx_min = float(new_frame_offset)
        idx_max = float(new_frame_offset + T - 1)
        idx_mean = (idx_min + idx_max) / 2.0
        idx_std = float(np.std(np.arange(new_frame_offset, new_frame_offset + T, dtype=np.float64)))
        s["index"] = {
            "min": [idx_min],
            "max": [idx_max],
            "mean": [idx_mean],
            "std": [idx_std],
            "count": [T],
        }

    # Patch task_index stats — get new task index from the old one stored in min
    if "task_index" in s:
        old_ti = int(round(s["task_index"]["min"][0]))
        new_ti = float(task_remap.get(old_ti, old_ti))
        s["task_index"] = {
            "min": [new_ti],
            "max": [new_ti],
            "mean": [new_ti],
            "std": [0.0],
            "count": [T],
        }

    rec["stats"] = s
    return rec


def src_video_path(dataset_path: str, chunk_size: int, ep_idx: int, video_key: str) -> str:
    chunk_id = ep_idx // chunk_size
    return os.path.join(dataset_path, "videos", f"chunk-{chunk_id:03d}", video_key, f"episode_{ep_idx:06d}.mp4")


def dst_video_path(output_path: str, chunk_size: int, ep_idx: int, video_key: str) -> str:
    chunk_id = ep_idx // chunk_size
    return os.path.join(output_path, "videos", f"chunk-{chunk_id:03d}", video_key, f"episode_{ep_idx:06d}.mp4")


# ============================================================================
# Write meta files
# ============================================================================


def write_info_json(meta_dir: str, first_info: dict, total_episodes: int, total_frames: int,
                    total_tasks: int, chunk_size: int):
    fps = first_info["fps"]
    img_feat = first_info["features"]["image"]
    image_height = img_feat["shape"][0]
    image_width = img_feat["shape"][1]

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
            "state": {"dtype": "float32", "shape": [7], "names": ["ee_pose_and_gripper_width"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["delta_ee_pose_and_gripper_action"]},
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


# ============================================================================
# Main merge logic
# ============================================================================


def merge_datasets(dataset_paths: list, output_path: str, chunk_size: int = 1000):
    print("\n" + "=" * 80)
    print("LeRobot v2.1 Dataset Merge")
    print("=" * 80)
    for i, p in enumerate(dataset_paths):
        print(f"  Input [{i}]: {p}")
    print(f"  Output:    {output_path}")
    print(f"  Chunk size: {chunk_size}")
    print("=" * 80 + "\n")

    os.makedirs(output_path, exist_ok=True)
    out_meta_dir = os.path.join(output_path, "meta")
    os.makedirs(out_meta_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Pass 1: collect unique tasks, build per-dataset task_index remap
    # ------------------------------------------------------------------ #
    all_task_strings = []  # ordered unique task strings
    ds_metas = []          # [(info, tasks, episodes), ...]

    for ds_path in dataset_paths:
        info, tasks, episodes = read_meta(ds_path)
        ds_metas.append((info, tasks, episodes))
        for t in tasks:
            if t["task"] not in all_task_strings:
                all_task_strings.append(t["task"])

    # Write merged tasks.jsonl
    with open(os.path.join(out_meta_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for i, task in enumerate(all_task_strings):
            f.write(json.dumps({"task_index": i, "task": task}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # Pass 2: merge episodes
    # ------------------------------------------------------------------ #
    episodes_out_path = os.path.join(out_meta_dir, "episodes.jsonl")
    if os.path.exists(episodes_out_path):
        os.remove(episodes_out_path)

    stats_out_path = os.path.join(out_meta_dir, "episodes_stats.jsonl")
    if os.path.exists(stats_out_path):
        os.remove(stats_out_path)

    video_keys = ["observation.images.image", "observation.images.wrist_image"]
    first_info = ds_metas[0][0]

    global_ep_offset = 0   # running new episode index
    global_frame_offset = 0  # running global frame index
    total_episodes = 0
    total_frames = 0
    has_stats = True  # will be set False if any source lacks stats

    for ds_idx, (ds_path, (info, tasks, episodes)) in enumerate(zip(dataset_paths, ds_metas)):
        src_chunk_size = info["chunks_size"]

        # Build task_index remap: old task_index -> new task_index
        task_remap = {t["task_index"]: all_task_strings.index(t["task"]) for t in tasks}

        parquet_files = get_sorted_parquets(ds_path)
        ep_stats = read_episode_stats(ds_path)
        if not ep_stats:
            has_stats = False
            print(f"  [WARN] No episodes_stats.jsonl found in {ds_path}, stats will be skipped.")

        print(f"Dataset {ds_idx + 1}/{len(dataset_paths)}: {os.path.basename(ds_path)}")
        print(f"  {len(parquet_files)} episodes, episode offset = {global_ep_offset}, frame offset = {global_frame_offset}")

        for local_idx, parquet_path in enumerate(tqdm(parquet_files, desc=f"  Merging [{ds_idx}]")):
            orig_ep_idx = int(os.path.splitext(os.path.basename(parquet_path))[0].split("_")[-1])
            new_ep_idx = global_ep_offset + local_idx
            new_chunk_id = new_ep_idx // chunk_size

            # --- Read & patch parquet ---
            table = pq.read_table(parquet_path)
            T = len(table)

            new_episode_col = pa.array([new_ep_idx] * T, type=pa.int64())
            new_index_col = pa.array(range(global_frame_offset, global_frame_offset + T), type=pa.int64())
            old_task_indices = table.column("task_index").to_pylist()
            new_task_col = pa.array([task_remap[ti] for ti in old_task_indices], type=pa.int64())

            schema_meta = table.schema.metadata  # preserve HuggingFace metadata

            table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", new_episode_col)
            table = table.set_column(table.schema.get_field_index("index"),         "index",         new_index_col)
            table = table.set_column(table.schema.get_field_index("task_index"),    "task_index",    new_task_col)

            # Restore schema metadata (set_column may drop it)
            table = table.replace_schema_metadata(schema_meta)

            # --- Write parquet ---
            out_chunk_dir = os.path.join(output_path, "data", f"chunk-{new_chunk_id:03d}")
            os.makedirs(out_chunk_dir, exist_ok=True)
            pq.write_table(table, os.path.join(out_chunk_dir, f"episode_{new_ep_idx:06d}.parquet"))

            # --- Copy videos ---
            for vk in video_keys:
                src = src_video_path(ds_path, src_chunk_size, orig_ep_idx, vk)
                dst = dst_video_path(output_path, chunk_size, new_ep_idx, vk)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"\n  [WARN] Missing video: {src}")

            # --- Append episodes.jsonl ---
            ep_meta = episodes.get(orig_ep_idx, {})
            ep_tasks = ep_meta.get("tasks", [all_task_strings[0]])
            with open(episodes_out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"episode_index": new_ep_idx, "tasks": ep_tasks, "length": T},
                                   ensure_ascii=False) + "\n")

            # --- Append episodes_stats.jsonl ---
            if has_stats and orig_ep_idx in ep_stats:
                patched = patch_stats_record(
                    ep_stats[orig_ep_idx], new_ep_idx, global_frame_offset, T, task_remap
                )
                with open(stats_out_path, "a", encoding="utf-8") as sf:
                    sf.write(json.dumps(patched, ensure_ascii=False) + "\n")
            elif has_stats:
                print(f"\n  [WARN] Missing stats for episode {orig_ep_idx} in {ds_path}")

            global_frame_offset += T

        global_ep_offset += len(parquet_files)
        total_episodes += len(parquet_files)

    total_frames = global_frame_offset

    # ------------------------------------------------------------------ #
    # Write info.json
    # ------------------------------------------------------------------ #
    write_info_json(
        out_meta_dir,
        first_info=first_info,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=len(all_task_strings),
        chunk_size=chunk_size,
    )

    print(f"\n{'=' * 80}")
    print("Merge Complete!")
    print(f"{'=' * 80}")
    print(f"Output:         {output_path}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames:   {total_frames}")
    print(f"Total tasks:    {len(all_task_strings)}")
    print(f"{'=' * 80}\n")


# ============================================================================
# Entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Merge LeRobot v2.1 datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "/workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_cleaned_real",
            "/workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_cleaned_mimicgen",
        ],
        help="Paths to input datasets in order (e.g. real mimicgen1 mimicgen2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/BlockPAP-v1_Mix",
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Episodes per chunk (default: 1000)",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        if not os.path.isdir(ds):
            print(f"[ERROR] Dataset not found: {ds}")
            return

    merge_datasets(args.datasets, args.output, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
