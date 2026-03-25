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

"""Pi0.5 evaluation on BlockPAP-v1 (external_cam, pd_ee_delta_pose control).

State / action convention (must match training in mg_generate_blockpap_data.py):
  state  (7,): [tcp_x, tcp_y, tcp_z, euler_x, euler_y, euler_z, gripper_total_width]
               - TCP pose in WORLD frame (= robot base frame since base is at origin)
               - Euler XYZ extrinsic (scipy convention)
               - gripper_total_width = qpos[7] + qpos[8]  in metres [0, 0.08]

  action (7,): [Δx, Δy, Δz, Δeuler_x, Δeuler_y, Δeuler_z, Δgripper_total_width]
               - delta EE pose in world frame
               - delta gripper total width in metres

ManiSkill control mode: pd_ee_delta_pose  (frame = root_translation:root_aligned_body_rotation)
  env action (7,): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, finger1_target]
               - Δpos in root (= world) frame,  Δrot as Euler-XYZ in root frame
               - finger1_target: absolute per-finger target in [0, 0.04] m (finger2 mimics)

Usage example:
    cd /workspace1/zhijun/RLinf
    python toolkits/eval_scripts_openpi/blockpap_eval.py \\
        --exp_name blockpap_pi05_eval \\
        --config_name pi05_blockpap_mix \\
        --pretrained_path logs/20260311-12:27:57/test_data_alpha_ratio/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt \\
        --num_episodes 20 --action_chunk 8 --num_steps 5 --num_save_videos 10
"""

import collections
import os
import pathlib
import sys

import imageio
import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Register BlockPAP-v1 env ────────────────────────────────────────────────
_REAL2SIM = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "real_franka"
    / "real2sim_env"
)
sys.path.insert(0, str(_REAL2SIM))
import pick_and_place  # noqa: F401 – registers BlockPAP-v1 (also provides TRAJ_ID global)

import gymnasium as gym
from scipy.spatial.transform import Rotation

from toolkits.eval_scripts_openpi import setup_logger
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT

# ── Constants ────────────────────────────────────────────────────────────────

CAM_NAME = "external_cam"
TASK_DESCRIPTION = "pick up the block and place it on the coaster"

# pd_ee_delta_pose arm action bounds (metres / rad per step)
POS_LIMIT = 0.1
ROT_LIMIT = 0.1


# ── Helpers ──────────────────────────────────────────────────────────────────


def _to_numpy(t):
    """Convert tensor (possibly batched) or array to 1-D numpy."""
    if hasattr(t, "cpu"):
        t = t.cpu()
    a = np.asarray(t).flatten()
    return a


def _load_blockpap_policy(args):
    """Load pi0.5 policy for BlockPAP evaluation.

    Handles two weight formats:
      1. safetensors directory (base model or HF download):
           args.pretrained_path  →  directory containing model.safetensors
           Norm stats loaded from  pretrained_path / <asset_id> / norm_stats.json
      2. RLinf SFT checkpoint (.pt file):
           args.pretrained_path  →  path to full_weights.pt  (OR its parent dir)
           args.norm_stats_path  →  directory with norm stats
                                    (default: hf_download/models/pi05_base)
    """
    import safetensors.torch
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import checkpoints as _checkpoints

    config = _CONFIGS_DICT[args.config_name]
    pt_path = pathlib.Path(args.pretrained_path)

    # ── Determine if this is a .pt or safetensors checkpoint ────────────────
    if pt_path.is_file() and pt_path.suffix == ".pt":
        # RLinf SFT checkpoint: load state dict directly
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:3]} …")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = args.norm_stats_path
    else:
        # safetensors checkpoint directory (base model or standard HF)
        weight_path = str(pt_path / "model.safetensors")
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        safetensors.torch.load_model(model, weight_path, strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = str(pt_path)

    # ── Load norm stats ──────────────────────────────────────────────────────
    data_config = config.data.create(config.assets_dirs, config.model)
    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is None; cannot locate norm stats.")

    norm_stats = _checkpoints.load_norm_stats(norm_stats_dir, asset_id)

    # ── Build policy with transforms ─────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = _policy.Policy(
        model,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": args.num_steps},
        metadata=config.policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )
    return policy


def get_ee_state(base_env):
    """Return (state_7d, gripper_total_width).

    state_7d = [tcp_x, tcp_y, tcp_z, euler_x, euler_y, euler_z, gripper_width]
    - TCP pose in world frame, Euler XYZ (extrinsic / scipy convention).
    - gripper_width = finger1 + finger2 joint positions (metres), total [0, 0.08] m.
    """
    tcp = base_env.agent.tcp
    p = _to_numpy(tcp.pose.p)[:3]
    q_wxyz = _to_numpy(tcp.pose.q)[:4]  # SAPIEN [w, x, y, z]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    euler = Rotation.from_quat(q_xyzw).as_euler("xyz").astype(np.float32)

    qpos = _to_numpy(base_env.agent.robot.get_qpos())
    gripper_width = float(qpos[7]) + float(qpos[8])  # total width [0, 0.08] m

    state = np.concatenate([p, euler, [gripper_width]], dtype=np.float32)
    return state, gripper_width


def get_image(obs):
    """Extract external_cam RGB image as uint8 HWC numpy array."""
    img = obs["sensor_data"][CAM_NAME]["rgb"]
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    else:
        img = np.asarray(img)
    if img.ndim == 4:  # [B, H, W, C] → [H, W, C]
        img = img[0]
    if img.dtype != np.uint8:
        img = (
            np.clip(img, 0, 255) if img.max() > 1.0 else (img * 255).clip(0, 255)
        ).astype(np.uint8)
    return img


def is_success(info):
    """Return True if any env reports success (handles tensor or bool)."""
    s = info.get("success", False)
    if isinstance(s, torch.Tensor):
        return bool(s.any().item())
    return bool(s)


# ── Main evaluation loop ─────────────────────────────────────────────────────


def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    np.random.seed(args.seed)

    # ── Override TRAJ_ID so env.reset() uses the requested trajectory ───────
    pick_and_place.TRAJ_ID = args.traj_id
    logger.info(
        f"TRAJ_ID set to '{args.traj_id}' "
        f"({'random initial states' if args.traj_id == 'random' else 'fixed initial state'})"
    )

    # ── Policy ──────────────────────────────────────────────────────────────
    logger.info("Loading pi0.5 policy …")
    policy = _load_blockpap_policy(args)
    logger.info("Policy loaded.")

    # ── Environment ─────────────────────────────────────────────────────────
    logger.info(
        f"Creating BlockPAP-v1  cam_t={args.cam_t!r}  control_mode=pd_ee_delta_pose"
    )
    env = gym.make(
        "BlockPAP-v1",
        obs_mode="rgb",
        render_mode="rgb_array",
        control_mode="pd_ee_delta_pose",
        cam_t=args.cam_t,
    )
    base_env = env.unwrapped
    logger.info(f"Action space: {env.action_space}")

    total_episodes = 0
    total_successes = 0

    for episode_idx in range(args.num_episodes):
        logger.info(f"\n─── Episode {episode_idx + 1}/{args.num_episodes} ───")

        policy.reset()
        obs, _ = env.reset()
        action_plan = collections.deque()

        # Collect frames for video
        frames = [get_image(obs)]

        # Initialise gripper state tracker
        _, gripper_width = get_ee_state(base_env)

        success = False
        for t in range(args.max_steps):
            img = get_image(obs)
            frames.append(img)

            state, gripper_width = get_ee_state(base_env)

            # Apply sim→real biases to match training convention.
            # state[2] is z (TCP height), state[5] is euler_z (yaw).
            # Rendering/logging uses the unmodified sim state.
            SIM_Z_BIAS = 0.1 if args.state_bias else 0.0  # +10 cm
            SIM_YAW_BIAS = -np.pi / 4 if args.state_bias else 0.0  # -45°
            state_for_model = state.copy()
            state_for_model[2] += SIM_Z_BIAS
            state_for_model[5] += SIM_YAW_BIAS

            # Re-plan when action queue is empty
            if not action_plan:
                observation = {
                    "observation/image": img,
                    "observation/state": state_for_model,
                    "prompt": TASK_DESCRIPTION,
                }
                action_chunk = policy.infer(observation)["actions"]
                # action_chunk: (horizon, 7) numpy
                assert action_chunk.ndim == 2 and action_chunk.shape[1] == 7, (
                    f"Unexpected action shape: {action_chunk.shape}"
                )
                action_plan.extend(action_chunk[: args.action_chunk])

            model_action = action_plan.popleft()  # (7,): [Δpos(3), Δeuler(3), Δgripper]

            # ── Debug: log action magnitudes every log_interval steps ───────
            if t % args.log_interval == 0:
                logger.info(
                    f"  [DEBUG t={t}] raw model action[:7]  = {np.round(model_action, 5).tolist()}"
                )
                logger.info(
                    f"  [DEBUG t={t}] pos_delta_norm (m)    = {np.linalg.norm(model_action[:3]):.5f}"
                )
                logger.info(
                    f"  [DEBUG t={t}] rot_delta_norm (rad)  = {np.linalg.norm(model_action[3:6]):.5f}"
                )
                logger.info(
                    f"  [DEBUG t={t}] gripper_abs (0=close) = {model_action[6]:.5f}"
                )
                logger.info(
                    f"  [DEBUG t={t}] EE state (sim, raw) "
                    f"z={state[2]:.4f}  yaw={np.degrees(state[5]):.1f}°  "
                    f"full={np.round(state, 4).tolist()}"
                )
                logger.info(
                    f"  [DEBUG t={t}] EE state (to model, biased) "
                    f"z={state_for_model[2]:.4f}(+{SIM_Z_BIAS * 100:.0f}cm)  "
                    f"yaw={np.degrees(state_for_model[5]):.1f}°({np.degrees(SIM_YAW_BIAS):+.0f}°)  "
                    f"full={np.round(state_for_model, 4).tolist()}"
                )

            # ── Normalize to ManiSkill pd_ee_delta_pose action space ─────────
            # ManiSkill uses clip_and_scale_action: input ∈ [-1,1] → output in [low, high]
            #   position:  output = 0.1 * input  (pos_lower=-0.1, pos_upper=0.1)
            #   rotation:  output = input * rot_lower = input * (-0.1)
            #              so input = actual_rad / rot_lower = actual_rad / (-0.1)
            #              this preserves sign: +0.01 rad → input -0.1 → output +0.01 rad ✓
            arm_norm = model_action[:6].astype(np.float32)
            arm_norm[:3] = arm_norm[:3] / POS_LIMIT  # meters → [-1, 1]
            arm_norm[3:6] = arm_norm[3:6] / (
                -POS_LIMIT
            )  # radians → [-1, 1] (rot_lower=-0.1)

            # ── Gripper: absolute [0,1] → normalized per-finger target ──────
            # Training: action[6] = absolute gripper total width in [0,1]
            #           (0 = fully closed, 1 = fully open = 0.08 m total width)
            # Env:  PDJointPosMimicController with lower=-0.01, upper=0.04 per finger
            #       uses clip_and_scale_action → input must be in [-1, 1]
            GRIPPER_LOWER = -0.01
            GRIPPER_UPPER = 0.04
            target_total = (
                float(np.clip(model_action[6], 0.0, 1.0)) * 0.08
            )  # → [0, 0.08] m
            finger_target = target_total / 2.0  # per finger [0, 0.04]
            gripper_norm = np.float32(
                2.0 * (finger_target - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER)
                - 1.0
            )  # → [-1, 1]

            env_action = np.append(arm_norm, gripper_norm)  # (7,)

            obs, _reward, terminated, truncated, info = env.step(env_action)

            if is_success(info):
                success = True
                logger.info(f"  SUCCESS at step {t + 1}")
                # Capture one last frame after success
                frames.append(get_image(obs))
                break

            if bool(terminated) or bool(truncated):
                break

        total_episodes += 1
        total_successes += int(success)

        logger.info(f"  Result : {'SUCCESS' if success else 'FAIL'}")
        logger.info(
            f"  Running: {total_successes}/{total_episodes} "
            f"= {total_successes / total_episodes:.1%}"
        )

        # ── Save video ───────────────────────────────────────────────────────
        if total_episodes <= args.num_save_videos:
            suffix = "success" if success else "failure"
            out_path = (
                pathlib.Path(args.log_dir)
                / args.exp_name
                / f"blockpap_ep{episode_idx:03d}_{suffix}.mp4"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            subsampled = frames[:: args.video_temp_subsample]
            imageio.mimwrite(
                str(out_path), subsampled, fps=max(1, 20 // args.video_temp_subsample)
            )
            logger.info(f"  Video  : {out_path}  ({len(subsampled)} frames)")

    env.close()

    sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    logger.info("\n=============== FINAL RESULTS ===============")
    logger.info(f"Success rate : {total_successes}/{total_episodes} = {sr:.2%}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate pi0.5 on BlockPAP-v1 using external_cam and pd_ee_delta_pose control."
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory for log files and videos"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="blockpap_pi05_eval",
        help="Experiment name (used for log/video filenames)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_blockpap_mix",
        help="OpenPI data config name (must match training config)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the fine-tuned weights. "
        "Either a directory with model.safetensors "
        "OR a full_weights.pt file from RLinf SFT training. "
        "Example (RLinf): "
        "logs/20260311-12:27:57/test_data_alpha_ratio/checkpoints/"
        "global_step_4000/actor/model_state_dict/full_weights.pt",
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default="../hf_download/models/pi05_base",
        help="Directory that contains the norm stats subdirectory "
        "(<asset_id>/norm_stats.json). Defaults to the pi05 base "
        "model download, where BlockPAP-v1_Mix/norm_stats.json lives.",
    )
    parser.add_argument(
        "--cam_t",
        type=str,
        default="og",
        choices=["og", "0302", "0303"],
        help="Camera translation preset (must match training data)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="Total number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3000,
        help="Max env steps per episode (BlockPAP-v1 has 600 step limit)",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=8,
        help="Action chunk size: replan every N steps",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Flow-matching denoising steps for pi0.5",
    )
    parser.add_argument(
        "--num_save_videos",
        type=int,
        default=10,
        help="Save videos for the first N episodes",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=1,
        help="Save every Nth frame to reduce video file size",
    )
    parser.add_argument(
        "--traj_id",
        type=str,
        default="random",
        choices=["random", "0", "15", "25", "40", "45"],
        help="Trajectory ID for env.reset(). 'random' randomizes block/coaster "
        "positions each episode; a number fixes the initial configuration.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Print debug info every N timesteps (default: 50)",
    )
    parser.add_argument(
        "--state_bias",
        type=lambda x: x.lower() != "false",
        default=True,
        metavar="true|false",
        help="Enable sim→real state biases (+10 cm Z, -45° yaw). "
        "Set to false when the model was trained without these offsets. (default: true)",
    )
    args = parser.parse_args()
    main(args)
