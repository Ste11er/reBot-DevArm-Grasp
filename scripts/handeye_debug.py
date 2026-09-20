"""Debug instrumentation and safety guards for the hand-eye calibration run.

Background (2026-09-19 incident): during an eye-in-hand auto traversal, right
after the pose-1 -> pose-2 transition, motor 1 spun clockwise into its hard
stop and bounced back. Offline replay of the preset poses through the SDK
planner (IK + geodesic + CLIK) produces a smooth, in-limit trajectory, so the
trigger lives in the runtime state: the measured joint feedback used as the
IK seed, the planned trajectory actually dispatched, or the raw MIT command
stream. This module records all three and adds guards that abort or kill a
move before it can command a violent joint jump.

Usage (auto mode only):
    python scripts/collect_handeye_eih.py --debug

Output (Log/handeye_debug_<ts>/):
  events.log   - every move dispatch/abort/kill, IK result, motor snapshots,
                 heartbeat summaries, final motor fault reports
  cmd.npz      - per control tick (~500 Hz): timestamp, commanded q_target,
                 cached measured q, moving flag. Flushed every 10 s so a hard
                 crash does not lose the buffer.
  move_XX.npz  - every planned joint trajectory (including aborted ones)

Guards:
  1. move_to_traj replica: validates the measured q and the planned
     trajectory (measured-q sanity, first-point continuity, per-step jump,
     total travel, joint limits) BEFORE the send thread starts.
  2. per control tick: clamps q_target inside URDF limits minus a margin and
     kills the running trajectory if the commanded target jumps away from the
     measured position.

The guards intentionally keep the arm holding its current position instead of
executing a suspect move, so an incident becomes a safe stop plus a log entry.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reBotArm_control_py.kinematics import compute_fk, pos_rot_to_se3, pad_q_for_model
from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik
from reBotArm_control_py.trajectory import (
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
)
from reBotArm_control_py.controllers import RebotArmEndPose


# Guard thresholds, in radians. Normal moves replayed offline stay far below
# these (max per-step delta ~0.008 rad, tracking error < 0.05 rad).
FIRST_POINT_MAX_DELTA_RAD = 0.20   # traj[0] vs measured q
STEP_MAX_DELTA_RAD = 0.05          # max step between trajectory points
JOINT_TRAVEL_MAX_RAD = 2.5         # whole-move travel of a single joint
LIVE_JUMP_RAD = 0.35               # |q_target - q_meas| trip level
LIMIT_MARGIN_RAD = 0.03            # clamp this far inside URDF limits
Q_OUT_OF_LIMIT_TOL_RAD = 0.15      # measured q beyond limits => invalid
FEEDBACK_FROZEN_WARN_S = 1.0       # q_meas bit-stable while moving => warn

HEARTBEAT_S = 2.0
CMD_FLUSH_S = 10.0
RECORD_SECONDS = 900

_ACTIVE_LOGGER: Optional["HandEyeDebugLogger"] = None
_ORIG_MOVE_TO_TRAJ = RebotArmEndPose.move_to_traj


class HandEyeDebugLogger:
    def __init__(self, controller: Any, rebotarm: Any, root: Path) -> None:
        self.ctrl = controller
        self.arm = rebotarm
        self.n = int(controller._n)
        model = controller._model
        self.lo = np.array(model.lowerPositionLimit[: self.n], dtype=float)
        self.hi = np.array(model.upperPositionLimit[: self.n], dtype=float)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(root) / "Log" / f"handeye_debug_{ts}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._ev = open(self.log_dir / "events.log", "a", buffering=1)

        self.counters = {
            "mit_err": 0, "fb_err": 0, "kills": 0, "aborts": 0,
            "no_fb_ticks": 0, "frozen_warns": 0, "tick_err": 0,
        }

        rate = float(rebotarm.rate)
        rows = int(rate * RECORD_SECONDS) + 1000
        self._t = np.zeros(rows)
        self._qc = np.zeros((rows, self.n))
        self._qm = np.full((rows, self.n), np.nan)
        self._mv = np.zeros(rows, dtype=np.uint8)
        self._i = 0
        self._recording = False
        self._finalized = False
        self._prev_qm: Optional[np.ndarray] = None
        self._last_change = time.monotonic()
        self._last_frozen_warn = 0.0
        self._last_hb = 0.0
        self._last_flush = 0.0
        self._move_k = 0
        # Captured here (and again in install()) so _tick works even if
        # install() is skipped in tests.
        self._orig_loop = controller._loop_cb

    # ── logging helpers ────────────────────────────────────────────────

    def event(self, msg: str) -> None:
        self._ev.write(f"[{time.strftime('%H:%M:%S')}.{int(time.time()*1000)%1000:03d}] {msg}\n")

    def _fmt(self, v: np.ndarray) -> str:
        return np.round(np.asarray(v, dtype=float), 4).tolist()

    def log_motor_snapshot(self, tag: str) -> None:
        parts = []
        for name, m in self.arm._motor_map.items():
            try:
                st = m.get_state()
            except Exception as e:  # pragma: no cover - defensive
                parts.append(f"{name}:ERR({e})")
                continue
            if st is None:
                parts.append(f"{name}:NO_STATE")
            else:
                parts.append(
                    f"{name}:pos={st.pos:+.4f} vel={st.vel:+.3f} tau={st.torq:+.3f} "
                    f"status={st.status_code} t_mos={st.t_mos:.0f} t_rot={st.t_rotor:.0f}"
                )
        self.event(f"motors[{tag}] " + " | ".join(parts))

    def dump_traj(self, pts: Optional[np.ndarray], q_meas: np.ndarray,
                  q_end: Optional[np.ndarray], note: str = "") -> None:
        self._move_k += 1
        path = self.log_dir / f"move_{self._move_k:02d}.npz"
        data: dict[str, Any] = {"q_meas": q_meas, "note": note}
        if pts is not None:
            data["pts"] = pts
        if q_end is not None:
            data["q_end"] = q_end
        np.savez(path, **data)

    # ── installation ───────────────────────────────────────────────────

    def install(self) -> None:
        # Wrap the controller tick. GraspDriver._loop_cb looks up
        # controller._loop_cb at call time, so an instance attribute wins.
        self._orig_loop = self.ctrl._loop_cb
        logger = self

        def _wrapped_loop(r, dt):
            try:
                logger._tick(r, dt)
            except Exception:
                logger.counters["tick_err"] += 1

        self.ctrl._loop_cb = _wrapped_loop

        # Count per-motor bus errors (normally swallowed by `except CallError: pass`).
        for name, m in list(self.arm._motor_map.items()):
            self._wrap_motor(name, m)

        grp = self.arm.groups.get("arm")
        self.event("attach: guards active")
        self.event(
            f"config: mode={self.ctrl._arm_control_mode} gravity_ff={self.ctrl._use_gravity_ff} "
            f"rate={self.arm.rate} joints={grp.joint_names if grp else '?'}"
        )
        if grp is not None:
            self.event(
                f"config: mit_kp={self._fmt(grp._mit_kp)} mit_kd={self._fmt(grp._mit_kd)} "
                f"limits_lo={self._fmt(self.lo)} limits_hi={self._fmt(self.hi)}"
            )
        self.event(
            f"thresholds: first_point={FIRST_POINT_MAX_DELTA_RAD} step={STEP_MAX_DELTA_RAD} "
            f"travel={JOINT_TRAVEL_MAX_RAD} live_jump={LIVE_JUMP_RAD} "
            f"limit_margin={LIMIT_MARGIN_RAD}"
        )
        self.log_motor_snapshot("attach")
        self._recording = True
        self._last_hb = time.monotonic()
        self._last_flush = time.monotonic()

    def _wrap_motor(self, name: str, m: Any) -> None:
        orig_send = m.send_mit
        orig_req = m.request_feedback
        cnt = self.counters

        def send(*a, **k):
            try:
                return orig_send(*a, **k)
            except Exception:
                cnt["mit_err"] += 1
                return None

        def req(*a, **k):
            try:
                return orig_req(*a, **k)
            except Exception:
                cnt["fb_err"] += 1
                return None

        m.send_mit = send
        m.request_feedback = req

    # ── per-tick guard + recorder ──────────────────────────────────────

    def _cached_q_meas(self, r: Any) -> np.ndarray:
        qm = np.full(self.n, np.nan)
        grp = r.groups.get("arm") if hasattr(r, "groups") else None
        if grp is None:
            return qm
        for k, jc in enumerate(grp._jcfgs[: self.n]):
            st = grp._mm[jc.name].get_state()
            if st is not None:
                qm[k] = st.pos
        return qm

    def _tick(self, r, dt: float) -> None:
        now = time.monotonic()
        qm = self._cached_q_meas(r)
        ctrl = self.ctrl

        if np.all(np.isfinite(qm)):
            # Final clamp: never command outside URDF limits (minus margin).
            np.clip(ctrl._q_target, self.lo + LIMIT_MARGIN_RAD,
                    self.hi - LIMIT_MARGIN_RAD, out=ctrl._q_target)

            fresh = (self._prev_qm is None
                     or np.max(np.abs(qm - self._prev_qm)) < 0.5)
            in_range = (np.all(qm > self.lo - Q_OUT_OF_LIMIT_TOL_RAD)
                        and np.all(qm < self.hi + Q_OUT_OF_LIMIT_TOL_RAD))
            if fresh and in_range:
                err = float(np.max(np.abs(ctrl._q_target - qm)))
                if err > LIVE_JUMP_RAD:
                    # Kill the trajectory sender and hold where we are.
                    ctrl._stop_send.set()
                    ctrl._q_target[:] = qm
                    ctrl._qd_target[:] = 0.0
                    ctrl._moving = False
                    self.counters["kills"] += 1
                    self.event(
                        f"LIVE KILL err={err:.3f} q_cmd={self._fmt(ctrl._q_target)} "
                        f"q_meas={self._fmt(qm)}"
                    )
                    print(f"[dbg] LIVE KILL: target jumped {err:.2f} rad from measured; "
                          f"holding position (see {self.log_dir})")

            # Feedback-frozen monitor: encoder bits must change while moving.
            if self._prev_qm is not None and np.any(qm != self._prev_qm):
                self._last_change = now
            if ctrl._moving and (now - self._last_change) > FEEDBACK_FROZEN_WARN_S:
                if now - self._last_frozen_warn > 5.0:
                    self._last_frozen_warn = now
                    self.counters["frozen_warns"] += 1
                    self.event(f"FEEDBACK FROZEN while moving: q_meas={self._fmt(qm)}")

            self._prev_qm = qm.copy()
        else:
            self.counters["no_fb_ticks"] += 1

        # Real controller tick: sends MIT with the (guarded) q_target.
        self._orig_loop(r, dt)

        if self._recording and self._i < self._t.shape[0]:
            self._t[self._i] = now
            self._qc[self._i] = ctrl._q_target
            self._qm[self._i] = qm
            self._mv[self._i] = 1 if ctrl._moving else 0
            self._i += 1

        if now - self._last_hb > HEARTBEAT_S:
            self._last_hb = now
            self._heartbeat()

        if now - self._last_flush > CMD_FLUSH_S:
            self._last_flush = now
            self._flush_cmd()

    def _heartbeat(self) -> None:
        qm = self._qm[self._i - 1] if self._i > 0 else np.full(self.n, np.nan)
        qc = self._qc[self._i - 1] if self._i > 0 else np.full(self.n, np.nan)
        err = (float(np.nanmax(np.abs(qc - qm)))
               if np.all(np.isfinite(qm)) and np.all(np.isfinite(qc)) else float("nan"))
        c = self.counters
        line = (f"[dbg] cmd_err={err:.3f} rad | mit_err={c['mit_err']} fb_err={c['fb_err']} "
                f"kills={c['kills']} aborts={c['aborts']} nofb={c['no_fb_ticks']} "
                f"frozen={c['frozen_warns']} tick_err={c['tick_err']} | "
                f"ticks={self._i} moving={int(self.ctrl._moving)}")
        print(line)
        self.event("hb " + line)

    def _flush_cmd(self, final: bool = False) -> None:
        if self._i == 0:
            return
        np.savez(
            self.log_dir / "cmd.npz",
            t=self._t[: self._i], q_cmd=self._qc[: self._i],
            q_meas=self._qm[: self._i], moving=self._mv[: self._i],
            final=final,
        )

    # ── teardown ───────────────────────────────────────────────────────

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._recording = False
        try:
            time.sleep(0.05)
        except Exception:
            pass

        self.log_motor_snapshot("finalize")
        for name, m in self.arm._motor_map.items():
            try:
                fault, warn = m.robstride_get_fault_report()
                self.event(f"fault[{name}] fault=0x{fault:08X} warning=0x{warn:08X}")
            except Exception as e:
                self.event(f"fault[{name}] read failed: {e}")

        if self._i >= 2:
            period_ms = float(np.median(np.diff(self._t[: self._i])) * 1000.0)
            self.event(f"summary: ticks={self._i} median_period={period_ms:.3f} ms")
        self.event(f"summary: counters={self.counters}")
        self._flush_cmd(final=True)
        self._ev.close()
        print(f"[dbg] Debug data saved to: {self.log_dir}")


# ── guarded move_to_traj (class-level patch) ────────────────────────────

def _debug_move_to_traj(
    self,
    x: float, y: float, z: float,
    roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
    duration: float = 2.0,
) -> bool:
    dbg = _ACTIVE_LOGGER
    if dbg is None:
        return _ORIG_MOVE_TO_TRAJ(
            self, x, y, z, roll=roll, pitch=pitch, yaw=yaw, duration=duration,
        )
    if not self._running:
        return False

    q_raw, qd_raw, tau_raw = self.rebotarm.get_state()
    q_meas = np.asarray(q_raw[: self._n], dtype=float).copy()
    q_start = pad_q_for_model(self._model, q_raw, self._n)
    dbg.log_motor_snapshot("pre_move")

    lo = np.array(self._model.lowerPositionLimit[: self._n], dtype=float)
    hi = np.array(self._model.upperPositionLimit[: self._n], dtype=float)

    def _hold_and_abort(reason: str) -> bool:
        self._stop_send.set()
        safe = np.where(np.isfinite(q_meas), q_meas, 0.0)
        self._q_target[:] = np.clip(safe, lo + LIMIT_MARGIN_RAD, hi - LIMIT_MARGIN_RAD)
        self._qd_target[:] = 0.0
        self._moving = False
        dbg.counters["aborts"] += 1
        dbg.event(f"MOVE ABORTED: {reason} | q_meas={dbg._fmt(q_meas)}")
        print(f"[dbg] MOVE ABORTED: {reason}; see {dbg.log_dir}")
        return False

    if (not np.all(np.isfinite(q_meas))
            or np.any(q_meas < lo - Q_OUT_OF_LIMIT_TOL_RAD)
            or np.any(q_meas > hi + Q_OUT_OF_LIMIT_TOL_RAD)):
        return _hold_and_abort("measured q invalid (feedback lost/corrupt?)")

    T_target = pos_rot_to_se3(
        np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw,
    )

    ik_result = solve_ik(
        self._model, self._data, self._end_frame_id,
        T_target, q_start, self._ik_solver_params,
        controlled_joints=self._n,
    )
    if not ik_result.success:
        dbg.event(f"IK FAIL err={ik_result.error:.4f} q_meas={dbg._fmt(q_meas)}")
        print(f"[RebotArmEndPose/Traj] IK 失败  err={ik_result.error:.4f}")
        return False

    q_end = ik_result.q
    q_end_padded = pad_q_for_model(self._model, q_end, self._n)

    T_start = compute_fk(self._model, q_start)[2]
    T_end = compute_fk(self._model, q_end_padded)[2]

    if duration <= 0:
        dist = float(np.linalg.norm(T_target.translation() - T_start.translation()))
        duration = max(1.0, dist / 0.1)

    cart_traj = plan_cartesian_geodesic_trajectory(
        T_start, T_end, duration, self._traj_params,
    )

    joint_traj = track_trajectory(
        self._model, self._end_frame_id,
        cart_traj.trajectory, q_start, self._clik_params,
        null_gain=0.1,
    )
    if not joint_traj:
        dbg.event("empty trajectory")
        print("[RebotArmEndPose/Traj] 轨迹为空")
        return False

    pts = [pt.q[: self._n].copy() for pt in joint_traj]
    P = np.array(pts)

    # ── trajectory validation before anything is sent ──
    reasons = []
    if not np.all(np.isfinite(P)):
        reasons.append("non-finite values in planned trajectory")
    if np.any(P < lo - 1e-9) or np.any(P > hi + 1e-9):
        reasons.append("planned trajectory exceeds joint limits")
    first_delta = float(np.max(np.abs(P[0] - q_meas)))
    if first_delta > FIRST_POINT_MAX_DELTA_RAD:
        reasons.append(f"first point {first_delta:.3f} rad away from measured q")
    step_max = float(np.max(np.abs(np.diff(P, axis=0)))) if len(P) > 1 else 0.0
    if step_max > STEP_MAX_DELTA_RAD:
        reasons.append(f"per-step jump {step_max:.3f} rad")
    travel = float(np.max(np.abs(P[-1] - P[0])))
    if travel > JOINT_TRAVEL_MAX_RAD:
        reasons.append(f"single-joint travel {travel:.3f} rad")
    n_unconverged = sum(1 for pt in joint_traj if not pt.ik_success)

    if reasons:
        dbg.dump_traj(P, q_meas, np.asarray(q_end), note="aborted")
        return _hold_and_abort("; ".join(reasons))

    # ── identical to the original dispatch tail ──
    self._stop_send.set()
    if self._send_thread is not None:
        self._send_thread.join(timeout=5.0)

    self._traj = pts
    self._moving = True
    self._stop_send.clear()
    self._send_thread = threading.Thread(
        target=self._send_loop, args=(duration,), daemon=True,
    )
    self._send_thread.start()

    dbg.dump_traj(P, q_meas, np.asarray(q_end), note="ok")
    dbg.event(
        f"move ok: n={len(P)} dur={duration:.2f}s ik_err={ik_result.error:.2e} "
        f"first_delta={first_delta:.3f} step_max={step_max:.3f} travel={travel:.3f} "
        f"unconverged={n_unconverged}/{len(P)} q_end={dbg._fmt(q_end)}"
    )
    return True


def attach_debug(controller: Any, rebotarm: Any, root: Path) -> HandEyeDebugLogger:
    """Install guards + logging on an auto-mode controller. Call after
    grasp_driver.start()."""
    global _ACTIVE_LOGGER
    dbg = HandEyeDebugLogger(controller, rebotarm, root)
    _ACTIVE_LOGGER = dbg
    RebotArmEndPose.move_to_traj = _debug_move_to_traj
    dbg.install()
    return dbg
