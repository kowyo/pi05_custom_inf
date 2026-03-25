"""Quick RealSense camera check.

Lists all connected devices and continuously reads frames from the chosen
camera, printing FPS and showing a live preview window (press q to quit).

Usage:
    # List devices only
    python check_realsense.py --list

    # Read from first available camera
    python check_realsense.py

    # Read from a specific serial
    python check_realsense.py --serial 123456789

    # No display (headless server)
    python check_realsense.py --no_display --num_frames 30
"""

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

# ── path setup (same as blockpap_real_eval.py) ────────────────────────────────
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "openpi"
sys.path.insert(0, str(_OPENPI_ROOT / "src"))

import openpi.lab_utils.camera_utils as camera_utils


def list_devices():
    devices = camera_utils.list_realsense_devices()
    if not devices:
        print("[WARN] No RealSense devices found.")
        print("  - Check USB connection")
        print("  - Run: rs-enumerate-devices")
        return []
    print(f"Found {len(devices)} RealSense device(s):")
    for i, d in enumerate(devices):
        print(f"  [{i}] name={d['name']}  serial={d['serial_number']}")
    return devices


def check_camera(serial: str | None, num_frames: int, show_display: bool):
    print(f"\nInitialising camera (serial={serial or 'any'}) ...")
    cam = camera_utils.RealSenseCamera(
        serial_number=serial,
        width=640,
        height=480,
        fps=30,
    )
    print("Camera initialised. Reading frames ...\n")

    frame_times = []
    prev_t = time.time()

    for i in range(num_frames if num_frames > 0 else 10**9):
        ret, frame, _ = cam.read()
        now = time.time()

        if not ret or frame is None:
            print(f"[Frame {i:4d}] READ FAILED")
            continue

        dt = now - prev_t
        prev_t = now
        frame_times.append(dt)

        fps_inst = 1.0 / dt if dt > 0 else 0.0
        h, w, c = frame.shape
        mean_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).mean(axis=(0, 1))

        if i % 10 == 0:
            print(
                f"[Frame {i:4d}]  {w}x{h}x{c}  "
                f"fps={fps_inst:5.1f}  "
                f"mean_RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f})"
            )

        if show_display:
            # Overlay info
            info = f"Frame {i}  {fps_inst:.1f} fps  serial={serial or 'any'}"
            cv2.putText(frame, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("RealSense check (press q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nUser pressed q, stopping.")
                break

    cam.release()
    if show_display:
        cv2.destroyAllWindows()

    if frame_times:
        fps_mean = 1.0 / np.mean(frame_times)
        fps_std  = np.std([1.0 / t for t in frame_times if t > 0])
        print(f"\n── Summary ──────────────────────────────")
        print(f"  Frames read : {len(frame_times)}")
        print(f"  FPS         : {fps_mean:.1f} ± {fps_std:.1f}")
        print(f"  OK          : {'YES' if fps_mean > 5 else 'LOW – check connection'}")


def main():
    parser = argparse.ArgumentParser(description="RealSense camera check")
    parser.add_argument("--list", action="store_true",
                        help="List connected devices and exit")
    parser.add_argument("--serial", type=str, default=None,
                        help="Camera serial number (omit for first available)")
    parser.add_argument("--num_frames", type=int, default=60,
                        help="Number of frames to read (0 = infinite until q)")
    parser.add_argument("--no_display", action="store_true",
                        help="Disable OpenCV window (for headless servers)")
    args = parser.parse_args()

    devices = list_devices()

    if args.list or not devices:
        return

    check_camera(
        serial=args.serial,
        num_frames=args.num_frames,
        show_display=not args.no_display,
    )


if __name__ == "__main__":
    main()
