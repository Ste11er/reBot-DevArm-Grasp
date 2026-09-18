"""Shared camera/config helpers for scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    from ..drivers.camera import CameraDriver, make_camera
except ImportError:
    from drivers.camera import CameraDriver, make_camera


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hand_eye(project_root: str | Path, cam_type: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    hand_eye_path = Path(project_root) / "config" / "calibration" / str(cam_type).lower() / "hand_eye.npz"
    if not hand_eye_path.exists():
        return None, None

    data = np.load(str(hand_eye_path), allow_pickle=False)
    T = data["T_result"].astype(np.float64)
    mode = str(data["mode"][0])

    # 防退化检查：T 为单位阵说明标定失败/退化（曾输出过这种结果），
    # 直接拒绝使用，避免后续抓取位姿全部错误。
    if np.allclose(T, np.eye(4), atol=1e-6):
        print(f"[WARN] {hand_eye_path} 的 T_result 是单位阵 — 标定结果无效，请重新标定")
        return None, mode

    return T, mode


def hand_eye_compensation_matrix(cfg: dict[str, Any]) -> np.ndarray:
    calibration = cfg.get("calibration") or {}
    compensation = calibration.get("hand_eye_compensation_m") or {}
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [
        float(compensation.get("x", 0.0)),
        float(compensation.get("y", 0.0)),
        float(compensation.get("z", 0.0)),
    ]
    return T


def hand_eye_mode(cfg: dict[str, Any]) -> str:
    """Resolve the hand-eye mode from config: 'eye_in_hand' or 'eye_to_hand'."""
    mode = str((cfg.get("calibration") or {}).get("hand_eye_mode", "eye_in_hand")).lower()
    if mode not in ("eye_in_hand", "eye_to_hand"):
        raise ValueError(f"calibration.hand_eye_mode 无效: {mode!r} (可选 eye_in_hand / eye_to_hand)")
    return mode


def compose_cam_to_base_transform(
    T_tcp2base: np.ndarray,
    T_hand_eye: np.ndarray,
    cfg: dict[str, Any],
    mode: Optional[str] = None,
) -> np.ndarray:
    """Return T_cam2base for the configured hand-eye mode.

    eye_in_hand: T_cam2base = compensation @ T_tcp2base @ T_cam2gripper
        （T_hand_eye 是标定得到的 T_cam2gripper，随 TCP 位姿变化）
    eye_to_hand: T_cam2base = inv(T_hand_eye) @ compensation
        （T_hand_eye 是标定得到的 T_base2cam，为常量；与 TCP 无关）
    """
    T_compensation = hand_eye_compensation_matrix(cfg)
    T_hand_eye = np.asarray(T_hand_eye, dtype=np.float64)
    if mode is None:
        mode = hand_eye_mode(cfg)
    if mode == "eye_to_hand":
        return np.linalg.inv(T_hand_eye) @ T_compensation
    return T_compensation @ np.asarray(T_tcp2base, dtype=np.float64) @ T_hand_eye


def configure_camera(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> dict[str, Any]:
    cam_cfg = cfg.setdefault("camera", {})
    camera_type = getattr(args, "camera_type", None)
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    fps = getattr(args, "fps", None)

    if camera_type is not None:
        cam_cfg["type"] = camera_type
    cam_type = str(cam_cfg.get("type", "")).lower()
    if not cam_type:
        raise ValueError("camera.type is missing in config; pass --camera-type or set it in YAML")

    if width is not None:
        cam_cfg["color_width"] = int(width)
        cam_cfg["depth_width"] = int(width)
    if height is not None:
        cam_cfg["color_height"] = int(height)
        cam_cfg["depth_height"] = int(height)
    if fps is not None:
        cam_cfg["fps"] = int(fps)
    elif camera_type is not None and realsense_default_fps is not None and "realsense" in cam_type:
        cam_cfg["fps"] = int(realsense_default_fps)
    return cfg


def create_camera_from_args(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> CameraDriver:
    return make_camera(configure_camera(cfg, args, realsense_default_fps=realsense_default_fps))
