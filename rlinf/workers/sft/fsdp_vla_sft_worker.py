# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class _DistributedWeightedSampler(torch.utils.data.Sampler):
    """Per-frame weighted sampler that splits samples evenly across DDP ranks.

    All ranks share the same random draw (seeded by ``seed + epoch``) but each
    rank takes every ``world_size``-th index starting at its own ``rank``.  This
    ensures:
      - no overlap between ranks within one epoch
      - the global draw respects the desired real/sim weight ratio
    """

    def __init__(
        self,
        weights: np.ndarray,
        num_samples_per_replica: int,
        rank: int,
        world_size: int,
        seed: int = 0,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples = num_samples_per_replica
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        total = self.world_size * self.num_samples
        indices = torch.multinomial(self.weights, total, replacement=True, generator=g)
        return iter(indices[self.rank :: self.world_size].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _SimYawBiasDataset:
    """在 transform_dataset 之前包装原始数据集。

    对 episode_index >= num_real_episodes 的帧（sim 数据），施加以下偏置：
      - state[2]（z）：+Z_BIAS，补偿 sim/real EE 高度约定差异（~10 cm）
      - state[5]（yaw）：+YAW_BIAS，补偿 sim/real EE 朝向约定差异（-45°）

    delta action 无需修正：连续帧的常数偏置在作差时互相抵消。
    """

    YAW_BIAS: float = -np.pi / 4  # -45°
    Z_BIAS:   float = 0.1          # +10 cm

    def __init__(self, dataset, num_real_episodes: int, log_every: int):
        self._dataset = dataset
        self._num_real_episodes = num_real_episodes
        self._log_every = log_every
        # 每个 worker 独立维护计数器（多 worker 时各自统计各自处理的样本）
        self._call_count = 0
        self._real_count = 0
        self._sim_count = 0
        self._last_sim_state = None  # 最近一条 sim state（bias 后）

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx):
        import logging

        sample = self._dataset[idx]
        episode_idx = sample.get("episode_index", None)
        if episode_idx is None:
            return sample

        # episode_index 可能是 tensor 或标量，统一转为 int
        if hasattr(episode_idx, "item"):
            episode_idx = episode_idx.item()

        is_sim = int(episode_idx) >= self._num_real_episodes
        if is_sim:
            sample = dict(sample)  # 浅拷贝，避免修改原始数据
            state = sample["state"]
            if hasattr(state, "clone"):          # torch.Tensor
                state = state.clone()
                state[2] = state[2] + self.Z_BIAS
                state[5] = state[5] + self.YAW_BIAS
            else:                                # numpy / list
                state = np.array(state, dtype=np.float32)
                state[2] += self.Z_BIAS
                state[5] += self.YAW_BIAS
            sample["state"] = state
            self._sim_count += 1
            self._last_sim_state = state
        else:
            self._real_count += 1

        self._call_count += 1
        if self._call_count % self._log_every == 0:
            # 仅 rank-0 打印，多 worker 时每个 worker 独立统计
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0

            if rank == 0:
                logging.info(
                    f"[SimBias] 最近 {self._log_every} 个样本: "
                    f"real={self._real_count}, sim={self._sim_count} "
                    f"(num_real_episodes 阈值={self._num_real_episodes})"
                )
                if self._last_sim_state is not None:
                    if hasattr(self._last_sim_state, "numpy"):
                        state_np = self._last_sim_state.detach().cpu().numpy()
                    else:
                        state_np = np.asarray(self._last_sim_state)
                    logging.info(
                        f"[SimBias] sim state 示例 (bias 后): "
                        f"[x={state_np[0]:.4f}, y={state_np[1]:.4f}, "
                        f"z={state_np[2]:.4f} (+{self.Z_BIAS*100:.0f}cm), "
                        f"roll={np.degrees(state_np[3]):.1f}°, pitch={np.degrees(state_np[4]):.1f}°, "
                        f"yaw={np.degrees(state_np[5]):.1f}° ({np.degrees(self.YAW_BIAS):+.0f}°)]"
                    )
            # 重置本轮计数
            self._real_count = 0
            self._sim_count = 0

        return sample


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            action_subsample_stride = getattr(
                self.cfg.actor.model.openpi, "action_subsample_stride", 1
            ) or 1
            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                action_subsample_stride=action_subsample_stride,
            )

            num_real_episodes = getattr(self.cfg.data, "num_real_episodes", None)
            alpha = getattr(self.cfg.data, "co_training_ratio", None)

            if num_real_episodes is not None and alpha is not None:
                data_loader = self._build_weighted_openpi_loader(
                    config, num_real_episodes, float(alpha),
                    action_subsample_stride=action_subsample_stride,
                )
            else:
                # For non-weighted path: temporarily inflate action_horizon so the dataset
                # loads action_horizon*stride frames; stride-aware action transform reduces
                # it back to the model action_horizon.
                if action_subsample_stride > 1:
                    import dataclasses
                    inflated_model = dataclasses.replace(
                        config.model,
                        action_horizon=config.model.action_horizon * action_subsample_stride,
                    )
                    config_for_load = dataclasses.replace(config, model=inflated_model)
                else:
                    config_for_load = config
                data_loader = openpi_data_loader.create_data_loader(
                    config_for_load, framework="pytorch", shuffle=True
                )
            return data_loader, data_loader.data_config()
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _build_weighted_openpi_loader(
        self, config, num_real_episodes: int, alpha: float, action_subsample_stride: int = 1
    ):
        """Build an OpenPI DataLoader with α-ratio weighted episode sampling.

        Frames from the first ``num_real_episodes`` episodes are treated as real
        (sampled with probability 1-α collectively); the remaining episodes are
        sim (sampled with probability α collectively).
        """
        from openpi.training.data_loader import (
            DataLoaderImpl,
            TorchDataLoader,
            create_torch_dataset,
            transform_dataset,
        )

        data_config = config.data.create(config.assets_dirs, config.model)
        # Load action_horizon * stride frames so that stride-aware action transforms
        # reduce back to the exact action_horizon the model expects.
        load_horizon = config.model.action_horizon * action_subsample_stride
        dataset = create_torch_dataset(data_config, load_horizon, config.model)
        # 在 RepackTransform 之前施加 sim yaw 偏置（此时样本仍有 state/episode_index 原始 key）
        # Controlled by cfg.data.apply_sim_bias (default True for backward compat).
        apply_sim_bias = getattr(self.cfg.data, "apply_sim_bias", True)
        if apply_sim_bias:
            dataset = _SimYawBiasDataset(dataset, num_real_episodes, log_every=config.batch_size)
        dataset = transform_dataset(dataset, data_config)

        # Unwrap transform chain to reach the underlying LeRobotDataset
        base = dataset
        while hasattr(base, "_dataset"):
            base = base._dataset
        episode_indices = np.array(base.hf_dataset["episode_index"])

        # Assign per-frame weights based on episode type
        n = len(episode_indices)
        real_mask = episode_indices < num_real_episodes
        sim_mask = ~real_mask
        n_real = int(real_mask.sum())
        n_sim = int(sim_mask.sum())

        weights = np.zeros(n, dtype=np.float64)
        if n_real > 0:
            weights[real_mask] = (1.0 - alpha) / n_real
        if n_sim > 0:
            weights[sim_mask] = alpha / n_sim

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = self._world_size
        local_batch_size = config.batch_size // world_size
        num_samples_per_replica = n // world_size

        import logging
        logging.info(
            f"[WeightedSampler] alpha={alpha}, real_episodes={num_real_episodes}, "
            f"real_frames={n_real}, sim_frames={n_sim}, "
            f"num_samples_per_replica={num_samples_per_replica}"
        )

        sampler = _DistributedWeightedSampler(
            weights=weights,
            num_samples_per_replica=num_samples_per_replica,
            rank=rank,
            world_size=world_size,
            seed=getattr(config, "seed", 0),
        )

        loader = TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=config.num_workers,
            seed=getattr(config, "seed", 0),
            framework="pytorch",
        )
        return DataLoaderImpl(data_config, loader)

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        observation, actions = next(self.data_iter)

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
            if x is not None
            else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        # train model return the loss
        return losses
