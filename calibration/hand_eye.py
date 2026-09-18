"""手眼标定 — 基于 OpenCV calibrateHandEye。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np


class CalibMode(Enum):
    EYE_IN_HAND = "eye_in_hand"   # 相机在末端，随末端运动
    EYE_TO_HAND = "eye_to_hand"   # 相机固定在基座/桌面，观察末端运动


_METHOD_MAP = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class CalibResult:
    # (4, 4) 标定结果；语义随 mode：
    #   eye_in_hand → T_cam2gripper（相机相对法兰）
    #   eye_to_hand → T_base2cam（基座系中相机的固定位姿）
    T_result: np.ndarray
    mode: str               # CalibMode.value
    n_samples: int
    method: str


@dataclass
class _Sample:
    T_gripper2base: np.ndarray   # (4, 4)
    T_marker2cam:   np.ndarray   # (4, 4)


class HandEyeCalibrator:
    """
    手眼标定器。

    Eye-in-Hand 模式（相机随末端运动，标定板固定）：
        求解 T_cam2gripper，使得
            T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        在所有姿态下恒成立。

    Eye-to-Hand 模式（相机固定在桌面/基座，标定板装在末端随臂运动）：
        求解 T_base2cam，使得
            T_marker2cam = inv(T_base2cam) @ T_gripper2base @ T_marker2gripper
        在所有姿态下恒成立。
        实现：将 T_gripper2base 取逆（即 T_base2gripper）作为 OpenCV 的
        gripper2base 输入，此时 OpenCV 解出的 X 恰为 T_base2cam（已用合成
        数据数值验证），无需再取逆。

    使用方法：
        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND)   # 或 EYE_TO_HAND
        calib.add_sample(T_gripper2base, T_marker2cam)
        ...
        result = calib.calibrate()
        HandEyeCalibrator.save(result, "hand_eye.npz")
    """

    def __init__(
        self,
        mode: CalibMode = CalibMode.EYE_IN_HAND,
        method: str = "TSAI",
    ) -> None:
        if not isinstance(mode, CalibMode):
            raise ValueError(f"mode must be a CalibMode, got {mode!r}")
        self._mode = mode
        self._method = method.upper()
        self._samples: List[_Sample] = []

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def add_sample(
        self,
        T_gripper2base: np.ndarray,
        T_marker2cam: np.ndarray,
    ) -> None:
        """
        添加一个标定样本。

        Args:
            T_gripper2base: (4,4) 末端到基座的变换（正运动学 FK 输出）
            T_marker2cam:   (4,4) 标记到相机的变换（ArUco 检测输出）

        两种模式采样方式：
            eye_in_hand: 标记板固定，机械臂带着相机运动
            eye_to_hand: 标记板装在末端随臂运动，相机固定
        """
        self._samples.append(_Sample(
            T_gripper2base=np.asarray(T_gripper2base, dtype=np.float64),
            T_marker2cam=np.asarray(T_marker2cam, dtype=np.float64),
        ))

    def calibrate(self, min_samples: int = 5) -> CalibResult:
        """
        计算手眼变换。

        Args:
            min_samples: 最少样本数（< 此值会抛出异常）

        Returns:
            CalibResult：
                eye_in_hand → T_result = T_cam2gripper
                eye_to_hand → T_result = T_base2cam
        """
        if self.n_samples < min_samples:
            raise ValueError(
                f"样本不足：{self.n_samples} < {min_samples}，请继续采集"
            )

        cv_method = _METHOD_MAP.get(self._method, cv2.CALIB_HAND_EYE_TSAI)

        R_g2b = [s.T_gripper2base[:3, :3] for s in self._samples]
        t_g2b = [s.T_gripper2base[:3,  3].reshape(3, 1) for s in self._samples]
        R_t2c = [s.T_marker2cam[:3, :3] for s in self._samples]
        t_t2c = [s.T_marker2cam[:3,  3].reshape(3, 1) for s in self._samples]

        if self._mode is CalibMode.EYE_TO_HAND:
            # Eye-to-hand: 把 base2gripper 喂给 gripper2base 参数位，
            # OpenCV 解出的 X 即 T_base2cam（相机在基座系中的固定位姿）。
            R_in = [np.linalg.inv(R) for R in R_g2b]
            t_in = [-np.linalg.inv(R) @ t for R, t in zip(R_g2b, t_g2b)]
        else:
            R_in = R_g2b
            t_in = t_g2b

        R_X, t_X = cv2.calibrateHandEye(
            R_in, t_in, R_t2c, t_t2c, method=cv_method
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_X
        T[:3,  3] = t_X.flatten()

        return CalibResult(
            T_result=T,
            mode=self._mode.value,
            n_samples=self.n_samples,
            method=self._method,
        )

    @staticmethod
    def save(result: CalibResult, path: Union[str, Path]) -> None:
        """保存标定结果为 .npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            T_result=result.T_result,
            mode=np.array([result.mode]),
            n_samples=np.array([result.n_samples]),
            method=np.array([result.method]),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> CalibResult:
        """从 .npz 文件加载标定结果。"""
        data = np.load(str(path), allow_pickle=False)
        return CalibResult(
            T_result=data["T_result"],
            mode=str(data["mode"][0]),
            n_samples=int(data["n_samples"][0]),
            method=str(data["method"][0]) if "method" in data else "TSAI",
        )
