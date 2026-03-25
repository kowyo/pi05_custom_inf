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

"""Pi0.5 real-robot evaluation on a Franka arm (BlockPAP task, PyTorch inference).

State / action convention (must match training in mg_generate_blockpap_data.py):
  state  (7,): [tcp_x, tcp_y, tcp_z, euler_x, euler_y, euler_z, gripper_total_width]
               - TCP pose in robot BASE (world) frame
               - Euler XYZ extrinsic (scipy convention)
               - gripper_total_width in metres [0, 0.08]

  action (7,): [Δx, Δy, Δz, Δeuler_x, Δeuler_y, Δeuler_z, gripper_norm]
               - delta EE pose in world frame
               - gripper_norm in [0, 1]: 0 = fully closed, 1 = fully open

Usage example:
    cd /workspace1/zhijun/RLinf
    python toolkits/eval_scripts_openpi/blockpap_real_eval.py \\
        --config_name pi05_blockpap_mix \\
        --pretrained_path logs/20260311-12:27:57/test_data_alpha_ratio/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt \\
        --nuc_ip 192.168.1.143 \\
        --external_camera_serial 123456789 \\
        --num_episodes 5 --action_chunk 8 --num_steps 5
"""

import collections
import datetime
import os
import pathlib
import queue
import sys
import threading
import time

import cv2
import imageio
import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Path setup ────────────────────────────────────────────────────────────────
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "openpi"
# franka_interface.py lives here (standalone, not a package)
sys.path.insert(0, str(_OPENPI_ROOT / "examples" / "lab"))
# openpi.lab_utils lives under src/
sys.path.insert(0, str(_OPENPI_ROOT / "src"))

from scipy.spatial.transform import Rotation

from franka_interface import FrankaInterface, MockRobot
import openpi.lab_utils.camera_utils as camera_utils

from toolkits.eval_scripts_openpi import setup_logger
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT

# ── Constants ─────────────────────────────────────────────────────────────────

# TASK_DESCRIPTION = "pick up the block and place it on the coaster"
TASK_DESCRIPTION = "stack the gray cube on the white cube"

# Safety limits for delta actions
MAX_POSITION_DELTA = 0.05  # 5 cm per step
MAX_ROTATION_DELTA = 0.2  # ~11 degrees per step

# Gripper geometry
GRIPPER_MAX_WIDTH = 0.08  # total width (both fingers), metres
GRIPPER_CLOSE_THRESHOLD = 0.04  # total width below this → treat as closed


# ── Policy loading ────────────────────────────────────────────────────────────


def _load_hf_norm_stats(hf_stats_path: str) -> dict:
    """Convert a HuggingFace LeRobot stats.json to OpenPI norm_stats format.

    LeRobot key names (observation.state / action) are remapped to the OpenPI
    convention (state / actions) expected by transforms.Normalize/Unnormalize.
    """
    import json
    from openpi.shared.normalize import NormStats  # type: ignore[import]

    with open(hf_stats_path) as f:
        hf = json.load(f)

    def _arr(key, field):
        return np.array(hf[key][field], dtype=np.float32)

    return {
        "state": NormStats(
            mean=_arr("observation.state", "mean"),
            std=_arr("observation.state", "std"),
            q01=_arr("observation.state", "q01"),
            q99=_arr("observation.state", "q99"),
        ),
        "actions": NormStats(
            mean=_arr("action", "mean"),
            std=_arr("action", "std"),
            q01=_arr("action", "q01"),
            q99=_arr("action", "q99"),
        ),
    }


def _load_blockpap_policy(args):
    """Load pi0.5 PyTorch policy from a .pt or safetensors checkpoint."""
    import safetensors.torch
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import checkpoints as _checkpoints

    config = _CONFIGS_DICT[args.config_name]
    pt_path = pathlib.Path(args.pretrained_path)

    if pt_path.is_file() and pt_path.suffix == ".pt":
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:3]} …")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = args.norm_stats_path
    elif pt_path.is_file() and pt_path.suffix == ".safetensors":
        # Direct path to a .safetensors file
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        safetensors.torch.load_model(model, str(pt_path), strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = args.norm_stats_path
    else:
        # Directory containing model.safetensors
        weight_path = str(pt_path / "model.safetensors")
        model = pi0_pytorch.PI0Pytorch(config=config.model)
        safetensors.torch.load_model(model, weight_path, strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm_stats_dir = str(pt_path)

    data_config = config.data.create(config.assets_dirs, config.model)
    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is None; cannot locate norm stats.")

    if getattr(args, "hf_stats_path", None):
        norm_stats = _load_hf_norm_stats(args.hf_stats_path)
    else:
        norm_stats = _checkpoints.load_norm_stats(norm_stats_dir, asset_id)

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


# ── Robot state helpers ───────────────────────────────────────────────────────


def get_robot_state(robot, target_ee_pose=None):
    """Read 7-D state from the real robot.

    Args:
        target_ee_pose: If provided (7D: [x,y,z,qx,qy,qz,qw]), use this as the
            EE pose instead of calling get_ee_pose(). Using the last commanded
            target pose rather than the measured pose reduces accumulation of
            sensor noise and controller tracking error.
            On the first step pass None to get the real measured pose.

    Returns:
        state (7,): [tcp_x, tcp_y, tcp_z, euler_x, euler_y, euler_z, gripper_total_width]
    """
    if target_ee_pose is not None:
        ee_pose = target_ee_pose.astype(np.float32)
    else:
        ee_pose = robot.get_ee_pose().astype(np.float32)  # [x, y, z, qx, qy, qz, qw]

    position = ee_pose[:3]
    quat_xyzw = ee_pose[3:]  # scipy expects [qx, qy, qz, qw]
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz").astype(np.float32)
    gripper_width = float(robot.get_gripper_position()[0])  # always read from hardware

    return np.concatenate([position, euler, [gripper_width]], dtype=np.float32)


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Euler XYZ (scipy extrinsic) → quaternion [qx, qy, qz, qw]."""
    return Rotation.from_euler("xyz", euler).as_quat().astype(np.float32)


# ── Camera display ────────────────────────────────────────────────────────────


class CameraDisplay:
    """File-based camera display compatible with openpi camera_viewer.py.

    Docker has no display server, so we write frames to a shared directory and
    let the user run camera_viewer.py on the HOST machine to view them:

        # On the HOST (outside Docker):
        python /home/showlab/Users/zhijun/openpi/examples/lab/camera_viewer.py
    """

    FRAME_DIR = "/home/showlab/Users/zhijun/eval/real_eval/tmp"
    FRAME_FILE = os.path.join(FRAME_DIR, "combined_frame.npy")
    STEP_FILE = os.path.join(FRAME_DIR, "step.txt")
    INFO_FILE = os.path.join(FRAME_DIR, "info.txt")

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.running = False
        self.quit_requested = False

    def start(self):
        os.makedirs(self.FRAME_DIR, exist_ok=True)
        self.running = True
        self._thread.start()
        print(f"  [Display] Writing frames to: {self.FRAME_DIR}")
        print("  [Display] On the HOST run:")
        print(
            f"    FRAME_DIR={self.FRAME_DIR} python /home/showlab/Users/zhijun/openpi/examples/lab/camera_viewer.py"
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        t: int,
        gripper_state: str,
        action: np.ndarray,
        wrist_frame_bgr: np.ndarray | None = None,
        side_frame_bgr: np.ndarray | None = None,
    ):
        """Push frame(s) (BGR) to writer thread. Drops if busy."""
        if not self.running:
            return
        wrist_copy = wrist_frame_bgr.copy() if wrist_frame_bgr is not None else None
        side_copy = side_frame_bgr.copy() if side_frame_bgr is not None else None
        try:
            self._q.put_nowait(
                (
                    frame_bgr.copy(),
                    t,
                    gripper_state,
                    action.copy(),
                    wrist_copy,
                    side_copy,
                )
            )
        except queue.Full:
            pass

    def _loop(self):
        while self.running:
            try:
                frame_bgr, t, gripper_state, action, wrist_bgr, side_bgr = self._q.get(
                    timeout=0.3
                )
            except queue.Empty:
                continue

            H, W = frame_bgr.shape[:2]
            pos_mm = float(np.linalg.norm(action[:3]) * 1000)
            rot_deg = float(np.degrees(np.linalg.norm(action[3:6])))

            def _overlay(img, label):
                out = img.copy()
                cv2.putText(
                    out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2
                )
                return out

            ext_disp = _overlay(
                frame_bgr, f"External  step={t}  gripper={gripper_state}"
            )
            cv2.putText(
                ext_disp,
                f"|Dpos|={pos_mm:.2f}mm  |Drot|={rot_deg:.2f}deg",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 230, 0),
                1,
            )

            panels = [ext_disp]
            if wrist_bgr is not None:
                wrist_disp = _overlay(cv2.resize(wrist_bgr, (W, H)), "Wrist")
                panels.append(wrist_disp)
            if side_bgr is not None:
                side_disp = _overlay(cv2.resize(side_bgr, (W, H)), "Side")
                panels.append(side_disp)
            disp = np.hstack(panels)

            # Atomic write (temp → rename)
            tmp = self.FRAME_FILE + ".tmp.npy"
            np.save(tmp, disp)
            os.replace(tmp, self.FRAME_FILE)
            with open(self.STEP_FILE, "w") as f:
                f.write(str(t))
            with open(self.INFO_FILE, "w") as f:
                f.write(
                    f"gripper={gripper_state}  |Dpos|={pos_mm:.2f}mm  |Drot|={rot_deg:.2f}deg"
                )

    def stop(self):
        self.running = False
        self._thread.join(timeout=2)
        # clean up shared files
        for p in (self.FRAME_FILE, self.STEP_FILE, self.INFO_FILE):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ── Gripper controller ────────────────────────────────────────────────────────


class GripperController:
    """Binary gripper controller with hysteresis and cooldown."""

    def __init__(self, cooldown_steps: int = 10):
        self.cooldown_steps = cooldown_steps
        self.last_state: bool | None = None  # True = closed
        self.steps_since_cmd = 0

    def reset(self):
        self.last_state = None
        self.steps_since_cmd = 0

    def step(self, robot, gripper_norm: float):
        """Send gripper command when state changes and cooldown allows.

        Args:
            gripper_norm: [0, 1] where 0 = fully closed, 1 = fully open.
        """
        target_width = float(np.clip(gripper_norm, 0.0, 1.0)) * GRIPPER_MAX_WIDTH
        should_close = target_width < GRIPPER_CLOSE_THRESHOLD

        self.steps_since_cmd += 1

        # Initialise state from hardware on first call
        if self.last_state is None:
            hw_width = float(robot.get_gripper_position()[0])
            self.last_state = hw_width < GRIPPER_CLOSE_THRESHOLD
            print(
                f"  [GRIPPER] Initialised: "
                f"{'CLOSED' if self.last_state else 'OPEN'} "
                f"(hw_width={hw_width * 1000:.1f} mm)"
            )

        prev_ok = getattr(robot, "get_gripper_prev_cmd_success", lambda: True)()
        if (
            should_close != self.last_state
            and self.steps_since_cmd >= self.cooldown_steps
            and prev_ok
        ):
            print(
                f"  [GRIPPER] {'OPEN' if self.last_state else 'CLOSED'} → "
                f"{'CLOSED' if should_close else 'OPEN'}"
            )
            robot.control_gripper(should_close)
            self.last_state = should_close
            self.steps_since_cmd = 0
        elif should_close != self.last_state:
            print(
                f"  [GRIPPER] cooldown "
                f"({self.steps_since_cmd}/{self.cooldown_steps} steps)"
            )


# ── Action execution ──────────────────────────────────────────────────────────


def execute_delta_action(robot, model_action: np.ndarray, current_state: np.ndarray):
    """Apply a delta EEF action to the real robot.

    Args:
        model_action (7,): [Δx, Δy, Δz, Δeuler_x, Δeuler_y, Δeuler_z, gripper_norm]
        current_state (7,): current robot state from get_robot_state()

    Returns:
        target_ee_pose (7,): [x, y, z, qx, qy, qz, qw] sent to robot.
    """
    delta_pos = np.clip(model_action[:3], -MAX_POSITION_DELTA, MAX_POSITION_DELTA)
    delta_rot = np.clip(model_action[3:6], -MAX_ROTATION_DELTA, MAX_ROTATION_DELTA)

    if not (
        np.allclose(model_action[:3], delta_pos)
        and np.allclose(model_action[3:6], delta_rot)
    ):
        print(
            f"  [WARN] Action clipped: "
            f"Δpos {np.round(model_action[:3], 4)} → {np.round(delta_pos, 4)}  "
            f"Δrot {np.round(model_action[3:6], 4)} → {np.round(delta_rot, 4)}"
        )

    target_pos = current_state[:3] + delta_pos
    target_euler = current_state[3:6] + delta_rot
    target_quat = _euler_to_quat(target_euler)

    target_ee_pose = np.concatenate([target_pos, target_quat]).astype(np.float32)
    robot.update_desired_ee_pose(target_ee_pose)
    return target_ee_pose


# ── Camera helper ─────────────────────────────────────────────────────────────


def setup_camera(args):
    """Create and return (external_cam, wrist_cam, side_cam).

    wrist_cam and side_cam are None if the corresponding serial is not provided.
    """
    if args.use_mock_camera:
        print("  Using MOCK camera (no hardware)")
        ext_cam = camera_utils.MockCamera(width=640, height=480)
        wrist_cam = (
            camera_utils.MockCamera(width=640, height=480)
            if args.wrist_camera_serial
            else None
        )
        side_cam = (
            camera_utils.MockCamera(width=640, height=480)
            if args.side_camera_serial
            else None
        )
        return ext_cam, wrist_cam, side_cam

    devices = camera_utils.list_realsense_devices()
    print(f"  Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        print(f"    [{i}] {dev['name']}  serial={dev['serial_number']}")

    if not devices:
        raise RuntimeError(
            "No RealSense cameras found. "
            "Use --use_mock_camera for testing without hardware."
        )

    ext_cam = camera_utils.RealSenseCamera(
        serial_number=args.external_camera_serial,
        width=640,
        height=480,
        fps=30,
    )
    if args.camera_exposure is not None:
        _set_camera_exposure(ext_cam, args.camera_exposure, args.camera_gain)

    wrist_cam = None
    if args.wrist_camera_serial:
        print(f"  Initialising wrist camera (serial: {args.wrist_camera_serial}) …")
        wrist_cam = camera_utils.RealSenseCamera(
            serial_number=args.wrist_camera_serial,
            width=640,
            height=480,
            fps=30,
        )
        if args.wrist_camera_exposure is not None:
            _set_camera_exposure(
                wrist_cam, args.wrist_camera_exposure, args.wrist_camera_gain
            )
        print("  Wrist camera ready.")

    side_cam = None
    if args.side_camera_serial:
        print(f"  Initialising side camera (serial: {args.side_camera_serial}) …")
        side_cam = camera_utils.RealSenseCamera(
            serial_number=args.side_camera_serial,
            width=640,
            height=480,
            fps=30,
        )
        if args.side_camera_exposure is not None:
            _set_camera_exposure(
                side_cam, args.side_camera_exposure, args.side_camera_gain
            )
        print("  Side camera ready.")

    return ext_cam, wrist_cam, side_cam


def _set_camera_exposure(cam, exposure_us: int, gain: int):
    """Disable auto-exposure and set manual exposure/gain on a RealSenseCamera.

    Args:
        exposure_us: Exposure time in microseconds (e.g. 6000 for D435).
                     D435 color sensor range: ~1 – 165000 μs.
        gain:        Analog gain (e.g. 64).
    """
    try:
        import pyrealsense2 as rs

        profile = cam.pipeline.get_active_profile()
        color_sensor = profile.get_device().query_sensors()[
            1
        ]  # index 1 = RGB sensor on D435
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, float(exposure_us))
        color_sensor.set_option(rs.option.gain, float(gain))
        print(f"  [Camera] Manual exposure set: exposure={exposure_us} μs  gain={gain}")
    except Exception as e:
        print(f"  [Camera][WARN] Failed to set exposure: {e}")


# ── Main evaluation loop ─────────────────────────────────────────────────────


def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    np.random.seed(args.seed)

    # ── Policy ──────────────────────────────────────────────────────────────
    logger.info("Loading pi0.5 policy …")
    policy = _load_blockpap_policy(args)
    logger.info("Policy loaded.")

    # ── Camera ──────────────────────────────────────────────────────────────
    logger.info("Initialising camera(s) …")
    cam, wrist_cam, side_cam = setup_camera(args)
    logger.info(
        f"Camera(s) ready. wrist={'yes' if wrist_cam else 'no'}  side={'yes' if side_cam else 'no'}"
    )

    # ── Robot ────────────────────────────────────────────────────────────────
    if args.use_mock_robot:
        logger.info("Using MOCK robot (no hardware)")
        robot = MockRobot(ip=args.nuc_ip, port=args.nuc_port)
    else:
        logger.info(f"Connecting to robot at {args.nuc_ip}:{args.nuc_port} …")
        robot = FrankaInterface(ip=args.nuc_ip, port=args.nuc_port)

    Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
    Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])

    gripper_ctrl = GripperController(cooldown_steps=args.gripper_cooldown_steps)
    dt = 1.0 / args.control_frequency

    # ── Camera display ───────────────────────────────────────────────────────
    display = None
    if args.show_camera:
        display = CameraDisplay()
        display.start()

    # ── Episode loop ─────────────────────────────────────────────────────────
    session_video_dir: pathlib.Path | None = None  # set on first episode

    try:
        for episode_idx in range(args.num_episodes):
            logger.info(f"\n─── Episode {episode_idx + 1}/{args.num_episodes} ───")
            logger.info("Move robot to start pose, then press Enter to begin …")
            try:
                input()
            except EOFError:
                pass

            # Start controller here so it doesn't time out while waiting for Enter
            logger.info("Starting Cartesian impedance controller …")
            robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
            logger.info("Controller started.")

            # Initialise session video directory on the first episode
            if session_video_dir is None and args.save_video:
                session_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                session_video_dir = (
                    pathlib.Path(args.log_dir) / args.exp_name / session_ts
                )
                session_video_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Video directory: {session_video_dir}")

            policy.reset()
            gripper_ctrl.reset()
            action_plan = collections.deque()
            frames = []
            target_pose = None  # None → use real get_ee_pose() on first step

            for t in range(args.max_steps):
                step_start = time.time()

                # ── 1. Robot state ───────────────────────────────────────────
                # After the first step use the last commanded target pose instead
                # of the measured pose to reduce sensor noise / tracking error.
                state = get_robot_state(robot, target_pose)

                # ── 2. Camera image(s) ───────────────────────────────────────
                ret, frame, _ = cam.read()
                if not ret:
                    logger.warning(
                        f"  [t={t}] External camera read failed, skipping step"
                    )
                    continue
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # uint8 HWC RGB

                wrist_img = None
                ret_w, wrist_frame = False, None
                if wrist_cam is not None:
                    ret_w, wrist_frame, _ = wrist_cam.read()
                    if ret_w:
                        wrist_img = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
                    else:
                        logger.warning(
                            f"  [t={t}] Wrist camera read failed, skipping wrist input"
                        )

                side_img = None
                ret_s, side_frame = False, None
                if side_cam is not None:
                    ret_s, side_frame, _ = side_cam.read()
                    if ret_s:
                        side_img = cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB)
                    else:
                        logger.warning(
                            f"  [t={t}] Side camera read failed, skipping side input"
                        )

                if args.save_video:
                    frames.append(img.copy())

                # ── 3. Policy inference (re-plan when queue is empty) ────────
                if not action_plan:
                    observation = {
                        "observation/image": img,
                        "observation/state": state,
                        "prompt": args.task_description,
                    }
                    if wrist_img is not None:
                        observation["observation/wrist_image"] = wrist_img
                    if side_img is not None:
                        observation["observation/side_image"] = side_img
                    inference_start = time.time()
                    action_chunk = policy.infer(observation)["actions"]
                    inference_ms = (time.time() - inference_start) * 1000

                    assert action_chunk.ndim == 2 and action_chunk.shape[1] == 7, (
                        f"Unexpected action shape: {action_chunk.shape}"
                    )
                    action_plan.extend(action_chunk[: args.action_chunk])
                    logger.info(
                        f"  [t={t}] NEW CHUNK  size={len(action_plan)}  "
                        f"inference={inference_ms:.1f} ms"
                    )

                model_action = action_plan.popleft()  # (7,)

                # ── 4. Debug logging ─────────────────────────────────────────
                if t % args.log_interval == 0:
                    logger.info(
                        f"  [t={t}] action={np.round(model_action, 5).tolist()}"
                    )
                    logger.info(f"  [t={t}] state ={np.round(state, 4).tolist()}")

                # ── 5. Execute action ────────────────────────────────────────
                target_pose = execute_delta_action(robot, model_action, state)
                gripper_ctrl.step(robot, model_action[6])

                # ── 6. Camera display ────────────────────────────────────────
                if display is not None:
                    gripper_str = "CLOSED" if gripper_ctrl.last_state else "OPEN"
                    display.update(
                        frame,
                        t,
                        gripper_str,
                        model_action,
                        wrist_frame_bgr=wrist_frame
                        if wrist_cam is not None and ret_w
                        else None,
                        side_frame_bgr=side_frame
                        if side_cam is not None and ret_s
                        else None,
                    )
                    if display.quit_requested:
                        logger.info("  User pressed q, stopping episode.")
                        break

                # ── 7. Timing ────────────────────────────────────────────────
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                elif t % args.log_interval == 0:
                    logger.warning(
                        f"  [t={t}] step overran: {elapsed * 1000:.1f} ms "
                        f"(target {dt * 1000:.1f} ms)"
                    )

            logger.info(f"  Episode {episode_idx + 1} done ({t + 1} steps).")

            # ── Save video ───────────────────────────────────────────────────
            if args.save_video and frames and session_video_dir is not None:
                out_path = session_video_dir / f"ep{episode_idx + 1:03d}.mp4"
                subsampled = frames[:: args.video_temp_subsample]
                imageio.mimwrite(
                    str(out_path),
                    subsampled,
                    fps=max(1, 30 // args.video_temp_subsample),
                )
                logger.info(f"  Video saved: {out_path}  ({len(subsampled)} frames)")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")

    finally:
        if display is not None:
            display.stop()
        logger.info("Cleaning up …")
        try:
            robot.terminate_current_policy()
        except Exception:
            pass  # "no controller running" is expected if episode ended naturally
        try:
            robot.close()
        except Exception as e:
            logger.warning(f"  Robot close error: {e}")
        try:
            cam.release()
        except Exception as e:
            logger.warning(f"  Camera cleanup error: {e}")
        if wrist_cam is not None:
            try:
                wrist_cam.release()
            except Exception as e:
                logger.warning(f"  Wrist camera cleanup error: {e}")
        if side_cam is not None:
            try:
                side_cam.release()
            except Exception as e:
                logger.warning(f"  Side camera cleanup error: {e}")
        logger.info("Done.")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate pi0.5 (PyTorch .pt) on real Franka for BlockPAP task."
    )

    # ── Logging ──
    parser.add_argument(
        "--log_dir", type=str, default="/home/showlab/Users/zhijun/eval/real_eval"
    )
    parser.add_argument("--exp_name", type=str, default="blockpap_real_eval")

    # ── Policy ──
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_blockpap_mix",
        help="OpenPI data config name (must match training)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to full_weights.pt or safetensors directory",
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default="../my_ckpt/models/pi05_base",
        help="Directory containing <asset_id>/norm_stats.json",
    )
    parser.add_argument(
        "--hf_stats_path",
        type=str,
        default=None,
        help="Path to a HuggingFace LeRobot Dataset v3 stats.json",
    )
    parser.add_argument(
        "--num_steps", type=int, default=5, help="Flow-matching denoising steps"
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=8,
        help="Execute this many actions before re-planning",
    )
    parser.add_argument("--task_description", type=str, default=TASK_DESCRIPTION)

    # ── Episode ──
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=250)

    # ── Robot ──
    parser.add_argument("--nuc_ip", type=str, default="192.168.1.143")
    parser.add_argument("--nuc_port", type=int, default=4242)
    parser.add_argument(
        "--control_frequency",
        type=float,
        default=10.0,
        help="Hz – target control loop rate",
    )
    parser.add_argument("--gripper_cooldown_steps", type=int, default=10)
    parser.add_argument(
        "--use_mock_robot",
        action="store_true",
        help="Use MockRobot (no hardware) for testing",
    )

    # ── Camera ──
    parser.add_argument(
        "--external_camera_serial",
        type=str,
        default=None,
        help="RealSense serial number for external camera",
    )
    parser.add_argument(
        "--wrist_camera_serial",
        type=str,
        default=None,
        help="RealSense serial number for wrist camera. "
        "If set, wrist image is fed to the model's left_wrist_0_rgb slot.",
    )
    parser.add_argument(
        "--use_mock_camera",
        action="store_true",
        help="Use MockCamera (no hardware) for testing",
    )
    parser.add_argument(
        "--camera_exposure",
        type=int,
        default=None,
        help="Manual exposure in microseconds for external D435 (e.g. 6000).",
    )
    parser.add_argument(
        "--camera_gain",
        type=int,
        default=64,
        help="Analog gain for external camera (default: 64)",
    )
    parser.add_argument(
        "--wrist_camera_exposure",
        type=int,
        default=None,
        help="Manual exposure in microseconds for wrist camera.",
    )
    parser.add_argument(
        "--wrist_camera_gain",
        type=int,
        default=64,
        help="Analog gain for wrist camera (default: 64)",
    )
    parser.add_argument(
        "--side_camera_serial",
        type=str,
        default=None,
        help="RealSense serial number for side camera. "
        "If set, side image is fed to the model's observation/side_image slot.",
    )
    parser.add_argument(
        "--side_camera_exposure",
        type=int,
        default=None,
        help="Manual exposure in microseconds for side camera.",
    )
    parser.add_argument(
        "--side_camera_gain",
        type=int,
        default=64,
        help="Analog gain for side camera (default: 64)",
    )
    parser.add_argument(
        "--show_camera",
        action="store_true",
        help="Show live camera preview window (requires display)",
    )

    # ── Video ──
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument(
        "--video_temp_subsample", type=int, default=1, help="Save every Nth frame"
    )

    # ── Misc ──
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
