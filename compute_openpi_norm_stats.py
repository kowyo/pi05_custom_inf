#!/usr/bin/env python3
"""Compute and save OpenPI normalization stats for RLinf dataconfigs.

This script generates `norm_stats.json` under:
  <model_path>/<repo_id>/norm_stats.json

Example:
  python scripts/compute_openpi_norm_stats.py \
      --model-path /workspace1/zhijun/hf_download/models/pi05_base \
      --config-name pi05_blockpap_mix
"""

from __future__ import annotations

import dataclasses
import argparse
import math
import os
from pathlib import Path

import numpy as np
import openpi.shared.normalize as normalize
import openpi.training.data_loader as data_loader

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute OpenPI norm stats (state/actions) for RLinf configs."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model directory used by training (also used as assets dir by RLinf).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_blockpap_mix",
        help="OpenPI dataconfig name from rlinf.models.embodiment.openpi.dataconfig.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for stats computation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for dataloader workers.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of batches to process (for quick checks).",
    )
    parser.add_argument(
        "--lerobot-home",
        type=str,
        default=None,
        help="Local LeRobot dataset root (e.g. /workspace1/zhijun/mg_dataset).",
    )
    parser.add_argument(
        "--hf-offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 to avoid any hub access.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Override dataset repo_id in dataconfig. Use local path for local datasets, "
            "e.g. /workspace1/zhijun/mg_dataset/BlockPAP-v1_Mix."
        ),
    )
    parser.add_argument(
        "--asset-id",
        type=str,
        default=None,
        help=(
            "Override asset_id used for norm-stats loading/saving. "
            "Defaults to basename(repo-id) when --repo-id is set."
        ),
    )
    parser.add_argument(
        "--fast-parquet-only",
        action="store_true",
        help=(
            "Fast path: read only state/actions from parquet and skip image/transforms. "
            "Recommended for local datasets like BlockPAP."
        ),
    )
    parser.add_argument(
        "--add-bias",
        action="store_true",
        default=False,
        help=(
            "Apply sim bias correction to sim data rows: "
            "state[2] += 0.10 m (z +10 cm), state[5] -= pi/4 rad (rz -45 deg). "
            "Requires --num-real to identify where sim data begins."
        ),
    )
    parser.add_argument(
        "--num-real",
        type=int,
        default=None,
        help=(
            "Number of rows that are real robot data (rows 0..num_real-1). "
            "Rows from num_real onward are treated as sim data for bias correction."
        ),
    )
    return parser.parse_args()


def _resolve_repo_root(repo_id: str) -> Path:
    repo_path = Path(repo_id)
    if repo_path.exists():
        return repo_path

    lerobot_home = os.environ.get("HF_LEROBOT_HOME", "")
    if lerobot_home:
        candidate = Path(lerobot_home) / repo_id
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve dataset root for repo_id={repo_id}. "
        "Set --repo-id to a local absolute path or set HF_LEROBOT_HOME correctly."
    )


def _apply_sim_bias(
    state_arr: np.ndarray, chunk_start: int, num_real: int | None
) -> np.ndarray:
    """Apply sim bias correction to rows that belong to sim data.

    For sim rows: state[2] += 0.10 (m), state[5] -= pi/4 (45 degrees in radians).
    Real rows are the first `num_real` rows; everything after is sim.
    """
    if num_real is None:
        return state_arr  # no-op if num_real not specified

    chunk_end = chunk_start + len(state_arr)
    if chunk_end <= num_real:
        return state_arr  # all real, no change

    state_arr = state_arr.copy()
    sim_offset = max(0, num_real - chunk_start)  # index within chunk where sim begins

    if state_arr.shape[1] > 2:
        state_arr[sim_offset:, 2] += 0.10  # z += 10 cm
    if state_arr.shape[1] > 5:
        state_arr[sim_offset:, 5] -= np.pi / 4  # rz -= 45 degrees (in radians)

    return state_arr


def _compute_from_parquet_only(
    repo_root: Path,
    action_dim: int,
    max_batches: int | None,
    batch_size: int,
    add_bias: bool = False,
    num_real: int | None = None,
) -> dict[str, normalize.RunningStats]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Fast parquet path requires pyarrow. Install pyarrow or run without --fast-parquet-only."
        ) from exc

    parquet_files = sorted((repo_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {repo_root / 'data'}")

    max_rows = None
    if max_batches is not None:
        max_rows = max(1, max_batches) * batch_size

    if add_bias and num_real is None:
        raise ValueError(
            "--add-bias requires --num-real to indicate where sim data starts."
        )

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    consumed_rows = 0
    for file_idx, parquet_file in enumerate(parquet_files, start=1):
        table = pq.read_table(parquet_file, columns=["state", "actions"])
        state_col = table.column("state").to_pylist()
        action_col = table.column("actions").to_pylist()

        state_arr = np.asarray(state_col, dtype=np.float32)
        action_arr = np.asarray(action_col, dtype=np.float32)

        if state_arr.ndim == 1:
            state_arr = state_arr[:, None]
        if action_arr.ndim == 1:
            action_arr = action_arr[:, None]

        # Match FrankaEEInputs behavior: pad trailing dims to model action_dim (default 32).
        if state_arr.shape[1] < action_dim:
            state_arr = np.pad(
                state_arr,
                ((0, 0), (0, action_dim - state_arr.shape[1])),
                mode="constant",
            )
        if action_arr.shape[1] < action_dim:
            action_arr = np.pad(
                action_arr,
                ((0, 0), (0, action_dim - action_arr.shape[1])),
                mode="constant",
            )

        if max_rows is not None and consumed_rows + len(state_arr) > max_rows:
            keep = max_rows - consumed_rows
            if keep <= 0:
                break
            state_arr = state_arr[:keep]
            action_arr = action_arr[:keep]

        if add_bias:
            state_arr = _apply_sim_bias(state_arr, consumed_rows, num_real)

        stats["state"].update(state_arr)
        stats["actions"].update(action_arr)
        consumed_rows += len(state_arr)

        if file_idx == 1 or file_idx % 50 == 0:
            print(
                f"Fast path processed files {file_idx}/{len(parquet_files)}, rows={consumed_rows}",
                flush=True,
            )

        if max_rows is not None and consumed_rows >= max_rows:
            break

    print(f"Fast path total rows used={consumed_rows}", flush=True)
    return stats


def main() -> None:
    args = parse_args()

    if args.lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    # Local v2.0 datasets often trigger timestamp sync warnings; training code also disables it.
    os.environ.setdefault("LEROBOT_DISABLE_TIMESTAMP_SYNC_CHECK", "1")

    print(f"HF_LEROBOT_HOME={os.environ.get('HF_LEROBOT_HOME', '')}")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')}")

    cfg = get_openpi_config(
        config_name=args.config_name,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )

    if args.repo_id is not None:
        data_cfg_factory = cfg.data
        if not dataclasses.is_dataclass(data_cfg_factory):
            raise TypeError("cfg.data is not a dataclass; cannot override repo_id")

        if not hasattr(data_cfg_factory, "assets") or not dataclasses.is_dataclass(
            data_cfg_factory.assets
        ):
            raise TypeError(
                "cfg.data.assets is not a dataclass; cannot override asset_id"
            )

        resolved_asset_id = args.asset_id or Path(args.repo_id).name
        overridden_assets = dataclasses.replace(
            data_cfg_factory.assets, asset_id=resolved_asset_id
        )
        overridden_data = dataclasses.replace(
            data_cfg_factory, repo_id=args.repo_id, assets=overridden_assets
        )
        cfg = dataclasses.replace(cfg, data=overridden_data)

    if args.num_workers is not None:
        cfg = dataclasses.replace(cfg, num_workers=args.num_workers)

    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    if args.fast_parquet_only:
        repo_root = _resolve_repo_root(data_cfg.repo_id)
        print(f"Using fast parquet path from: {repo_root}", flush=True)
        if args.add_bias:
            print(
                f"Sim bias enabled: num_real={args.num_real}, "
                "state[2]+=0.10m, state[5]-=pi/4 for sim rows",
                flush=True,
            )
        stats = _compute_from_parquet_only(
            repo_root=repo_root,
            action_dim=cfg.model.action_dim,
            max_batches=args.max_batches,
            batch_size=cfg.batch_size,
            add_bias=args.add_bias,
            num_real=args.num_real,
        )
    else:
        dataset = data_loader.create_torch_dataset(
            data_cfg, cfg.model.action_horizon, cfg.model
        )
        dataset = data_loader.TransformedDataset(
            dataset,
            [*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs],
        )

        total_batches = max(1, math.ceil(len(dataset) / cfg.batch_size))
        if args.max_batches is not None:
            total_batches = min(total_batches, max(1, args.max_batches))

        stats = {
            "state": normalize.RunningStats(),
            "actions": normalize.RunningStats(),
        }

        max_samples = min(len(dataset), total_batches * cfg.batch_size)
        print(
            f"Total samples={len(dataset)}, using max_samples={max_samples}, "
            f"batch_size={cfg.batch_size}, total_batches={total_batches}",
            flush=True,
        )

        for batch_idx, start_idx in enumerate(
            range(0, max_samples, cfg.batch_size), start=1
        ):
            end_idx = min(start_idx + cfg.batch_size, max_samples)
            batch_items = [dataset[i] for i in range(start_idx, end_idx)]

            state_batch = np.stack(
                [np.asarray(item["state"]) for item in batch_items], axis=0
            )
            action_batch = np.stack(
                [np.asarray(item["actions"]) for item in batch_items], axis=0
            )

            stats["state"].update(state_batch)
            stats["actions"].update(action_batch)
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"Processed {batch_idx}/{total_batches} batches...", flush=True)

    norm_stats = {k: v.get_statistics() for k, v in stats.items()}

    out_dir = Path(args.model_path) / data_cfg.asset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(out_dir, norm_stats)

    print(f"Saved norm stats to: {out_dir / 'norm_stats.json'}")
    print(f"repo_id: {data_cfg.repo_id}")
    print(f"asset_id: {data_cfg.asset_id}")


if __name__ == "__main__":
    main()
