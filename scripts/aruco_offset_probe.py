"""ArUco 板点位偏移量测量工具。

用途：测量 calibration.hand_eye_compensation_m。
把 ArUco 板放在桌面某个位置，脚本实时显示检测到的标记：
  - SPACE: 记录"ArUco 板算出的点"（标记中心，经手眼变换到基座坐标系）
  - T:     记录"真实点"（手动把机械臂尖端对准标记中心后按 T，取当前 TCP 位置）
  - Q/ESC: 退出，输出每对点的偏差、平均值和建议的补偿值

原理（eye_in_hand，补偿在基座坐标系下直接加到计算结果上）：
  p_computed = p_raw + t_cfg
  建议新补偿  t_new = t_cfg + mean(p_true - p_computed)

测量前建议先把 hand_eye_compensation_m 清零再标定；脚本按当前配置计算，
并自动把当前补偿值折算进建议值。换多个位置（近/远/左/右）各测一组，
若各点偏差差异大（>5mm）说明有旋转/深度尺度误差，应重新做手眼标定。

Usage:
    python scripts/aruco_offset_probe.py
    python scripts/aruco_offset_probe.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.camera import make_camera
from utils.camera_utils import (
    compose_cam_to_base_transform,
    hand_eye_compensation_matrix,
    load_config,
    load_hand_eye,
)

# 复用手眼标定采集脚本的手动模式控制器（重力补偿，可拖着臂走）
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from collect_handeye_eih import GravityCompController  # noqa: E402

from drivers.robot.grasp_driver import GraspDriver  # noqa: E402
from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.controllers import RebotArmEndPose  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco 板点位偏移量测量")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--log",
        default=None,
        help="采样记录输出路径（JSON Lines）；默认 Log/aruco_offset_<ts>.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    cam_cfg = cfg.get("camera", {})
    cam_type = str(cam_cfg.get("type", "")).lower()
    aruco_cfg = (cfg.get("calibration") or {}).get("aruco") or {}

    T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
    if T_hand_eye is None:
        print("[ERROR] 手眼标定不可用（config/calibration 下无有效 hand_eye.npz），先完成手眼标定")
        return 1
    if hand_eye_mode != "eye_in_hand":
        print(f"[WARN] 当前模式 {hand_eye_mode}：补偿在相机坐标系下生效，本脚本的偏差仍按基座坐标给出")

    T_comp = hand_eye_compensation_matrix(cfg)
    print(f"当前补偿 (m): x={T_comp[0,3]:+.4f} y={T_comp[1,3]:+.4f} z={T_comp[2,3]:+.4f}")

    print(f"=== Camera: {cam_cfg.get('type')} ===")
    cam = make_camera(cfg)

    robot_cfg = cfg.get("robot", {})
    rebotarm = RebotArm()
    # 与 collect_handeye_eih.py --manual 相同的结构：
    # GraspDriver 只用它的 FK（get_tcp_pose），controller 不 start（避免位置锁定）；
    # GravityCompController 负责连接、使能并进入重力补偿，臂可以用手拖动。
    controller = RebotArmEndPose(rebotarm, arm_control_mode="mit", use_gravity_ff=True)
    grasp_driver = GraspDriver(
        rebotarm,
        controller,
        gripper_config=robot_cfg.get("gripper"),
        repo_root=robot_cfg.get("repo_root"),
    )
    gc_ctrl = GravityCompController(rebotarm)

    log_path = Path(args.log) if args.log else PROJECT_ROOT / "Log" / f"aruco_offset_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 采样记录: computed = ArUco 算出的基座坐标点; measured = TCP 对准后的位置
    samples: list[dict[str, Any]] = []
    pending_computed: Optional[dict[str, Any]] = None  # 最近一次记录、还未配对真实点的计算点

    def record(entry: dict[str, Any]) -> None:
        samples.append(entry)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    window = "ArUco Offset Probe"
    print("\n[流程] 放好 ArUco 板 -> SPACE 记录计算点 -> 手动把机械臂尖端对准标记中心 -> T 记录真实点")
    print("[Keys] SPACE=记录ArUco点  T=记录TCP真实点  Q/ESC=退出并汇总\n")

    try:
        cam.open()
        cam.warm_up(15)
        cam.setup_aruco(
            marker_length_m=float(aruco_cfg.get("marker_length_m", 0.05)),
            dict_id=int(aruco_cfg.get("dict_id", 0)),
            target_marker_id=aruco_cfg.get("target_marker_id"),
        )

        print("=== Robot: 重力补偿启动（可以用手拖动机械臂）===")
        gc_ctrl.start()

        while True:
            color_bgr, _ = cam.get_frame()
            if color_bgr is None:
                continue

            pose = cam.detect_aruco(color_bgr)
            display = cam.draw_aruco(color_bgr)

            if pose is not None:
                T_cam2base = compose_cam_to_base_transform(
                    grasp_driver.get_tcp_pose(), T_hand_eye, cfg, mode=hand_eye_mode,
                )
                p_cam = pose.T_marker2cam[:3, 3]
                p_base = (T_cam2base @ np.append(p_cam, 1.0))[:3]

                # 标记中心投影回图像像素，作为对准参照（十字准星）。
                # p_cam 已是相机系坐标，投影时 rvec/tvec 取零（仅用内参+畸变）。
                proj, _ = cv2.projectPoints(
                    p_cam.reshape(1, 1, 3).astype(np.float64),
                    np.zeros(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64),
                    np.asarray(cam.K, dtype=np.float64),
                    np.asarray(cam.D, dtype=np.float64).reshape(-1, 1),
                )
                px, py = float(proj[0, 0, 0]), float(proj[0, 0, 1])
                cv2.drawMarker(display, (int(px), int(py)), (0, 0, 255),
                               cv2.MARKER_CROSS, 28, 2)
                cv2.putText(display, f"aim ({int(px)},{int(py)})", (int(px) + 18, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                lines = [
                    f"marker id={pose.id}",
                    f"cam  xyz=({p_cam[0]:+.4f},{p_cam[1]:+.4f},{p_cam[2]:+.4f})",
                    f"base xyz=({p_base[0]:+.4f},{p_base[1]:+.4f},{p_base[2]:+.4f})",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(display, text, (10, 30 + 26 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 255, 140), 2)
            else:
                cv2.putText(display, "no marker", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2)

            if pending_computed is not None:
                cv2.putText(display, "[已记录计算点，请对准后按 T]", (10, display.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 215, 255), 2)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord(" "), 13):  # SPACE / Enter
                if pose is None:
                    print("[SPACE] 未检测到标记，忽略")
                    continue
                T_tcp = grasp_driver.get_tcp_pose()
                pending_computed = {
                    "type": "computed",
                    "t": time.time(),
                    "p_cam": p_cam.tolist(),
                    "p_base": p_base.tolist(),
                    "pixel_xy": [round(px, 1), round(py, 1)],
                    "tcp_xyz": T_tcp[:3, 3].tolist(),
                    "marker_id": pose.id,
                }
                record(pending_computed)
                print(f"[SPACE] 计算点(基座) xyz=({p_base[0]:+.4f},{p_base[1]:+.4f},{p_base[2]:+.4f})"
                      f"  图像位置=({int(px)},{int(py)})  <- 把尖端对准十字处标记中心后按 T")

            elif key in (ord("t"), ord("T")):
                T_tcp = grasp_driver.get_tcp_pose()
                measured = {
                    "type": "measured",
                    "t": time.time(),
                    "tcp_xyz": T_tcp[:3, 3].tolist(),
                }
                record(measured)
                if pending_computed is None:
                    print("[T] 没有未配对的计算点（应先按 SPACE），仅记录了 TCP 位置")
                    continue
                p_true = T_tcp[:3, 3]
                p_comp = np.asarray(pending_computed["p_base"])
                delta = p_true - p_comp
                pending_computed = None
                print(f"[T] 真实点  xyz=({p_true[0]:+.4f},{p_true[1]:+.4f},{p_true[2]:+.4f})")
                print(f"    偏差(真实-计算) dx={delta[0]:+.4f} dy={delta[1]:+.4f} dz={delta[2]:+.4f}  |d|={np.linalg.norm(delta)*1000:.1f}mm")

    finally:
        try:
            gc_ctrl.safe_home()  # 停止重力补偿、归位、断开连接
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # ---- 汇总：把每对 计算点/真实点 配对，输出平均偏差和建议补偿 ----
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pending: Optional[np.ndarray] = None
    for s in samples:
        if s["type"] == "computed":
            pending = np.asarray(s["p_base"])
        elif s["type"] == "measured" and pending is not None:
            pairs.append((pending, np.asarray(s["tcp_xyz"])))
            pending = None

    print(f"\n=== 汇总: {len(pairs)} 对测量点，记录已保存到 {log_path} ===")
    if not pairs:
        print("没有完整配对的样本（每对需要 SPACE + T 各一次）")
        return 0

    deltas = []
    for i, (p_comp, p_true) in enumerate(pairs, 1):
        d = p_true - p_comp
        deltas.append(d)
        print(f"  #{i} 计算=({p_comp[0]:+.4f},{p_comp[1]:+.4f},{p_comp[2]:+.4f})"
              f"  真实=({p_true[0]:+.4f},{p_true[1]:+.4f},{p_true[2]:+.4f})"
              f"  偏差=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})  |d|={np.linalg.norm(d)*1000:.1f}mm")

    mean_d = np.mean(deltas, axis=0)
    spread = np.max(np.abs(np.asarray(deltas) - mean_d), axis=0)
    print(f"\n平均偏差 (真实-计算, m): x={mean_d[0]:+.4f} y={mean_d[1]:+.4f} z={mean_d[2]:+.4f}")
    print(f"各点偏差波动 (max|d-mean|, m): x={spread[0]:.4f} y={spread[1]:.4f} z={spread[2]:.4f}")

    suggested = T_comp[:3, 3] + mean_d
    print(f"\n建议补偿 hand_eye_compensation_m (当前补偿 + 平均偏差):")
    print(f"  x: {suggested[0]:+.4f}")
    print(f"  y: {suggested[1]:+.4f}")
    print(f"  z: {suggested[2]:+.4f}")
    if np.any(spread > 0.005):
        print("\n[WARN] 各点偏差波动超过 5mm —— 偏差不是常量平移，"
              "常量补偿无法修正；请检查 marker_length_m 是否与实物一致并重新做手眼标定。")

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
