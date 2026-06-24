"""Run one CS2810 Phase 2 match with Viser visualization and JSON results."""

import json
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

# Use EGL for headless offscreen rendering (must be set before any MuJoCo import).
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mediapy as media
import requests
import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers.video_recorder import VideoRecorder
from mjlab.viewer import ViserPlayViewer, ViewerConfig

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.assets.robots.unitree_g1.g1_constants import FULL_COLLISION, HOME_KEYFRAME
from src.tasks.soccer import mdp
from src.tasks.soccer.ball import get_ball_cfg
from src.tasks.soccer.goal import get_goal_cfg
from src.tasks.soccer.ground import get_ground_cfg
from src.tasks.soccer.mdp.goalkeeper_obs import _GK_DEFAULT_JOINT_POS, get_gk_robot_cfg
from src.tasks.soccer.soccer_env_cfg import _add_soccer_scene_postproc


PHASE2_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PHASE2_DIR / "phase2_config.yaml"
DEFAULT_RESULTS_DIR = PHASE2_DIR / "results"

_SHOOTER_CFG = SceneEntityCfg("shooter")
_GK_CFG = SceneEntityCfg("goalkeeper")
_BALL_CFG = SceneEntityCfg("ball")


def _load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "team"


def _utc_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _vec3(values: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))



def _copy_tensor_1d_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


class _TimedVideoRecorder(VideoRecorder):
    def __init__(
        self,
        *args: Any,
        step_dt: float,
        capture_fps: float = 24.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._step_dt = step_dt
        self._capture_interval_s = 1.0 / capture_fps if capture_fps > 0 else 0.0
        self._capture_fps = capture_fps
        self._sim_time_s = 0.0
        self._next_capture_s = 0.0
        self._write_threads: list[threading.Thread] = []
        self._write_errors: list[BaseException] = []
        self._write_lock = threading.Lock()

    def _start_recording(self) -> None:
        super()._start_recording()
        self._sim_time_s = 0.0
        self._next_capture_s = 0.0

    def step(self, action: torch.Tensor) -> Any:
        if self.is_recording:
            self._sim_time_s += self._step_dt
        return super().step(action)

    def _record_frame(self) -> None:
        if self._wrapped_env.render_mode == "rgb_array":
            if self._sim_time_s < self._next_capture_s:
                return
            while self._next_capture_s <= self._sim_time_s:
                self._next_capture_s += self._capture_interval_s
            frame = self._wrapped_env.render()
            if frame is not None:
                rgb_frame = (
                    frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
                )
                self.current_video_frames.append(rgb_frame)

    def _finish_recording(self) -> None:
        if self.current_video_frames:
            video_frames = []
            for frame in self.current_video_frames:
                frame = np.asarray(frame) if not isinstance(frame, np.ndarray) else frame
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                video_frames.append(frame)

            fps = int(round(self._capture_fps)) if self._capture_fps > 0 else self._wrapped_env.metadata.get("render_fps", 30)
            video_path = str(self.current_video_path)
            disable_logger = self.disable_logger
            def _write_video() -> None:
                try:
                    media.write_video(video_path, video_frames, fps=fps)
                    if not disable_logger:
                        print(f"[INFO] Saved video to {video_path}")
                except BaseException as exc:
                    with self._write_lock:
                        self._write_errors.append(exc)

            thread = threading.Thread(
                target=_write_video,
                name=f"video-write-{self.video_count}",
                daemon=True,
            )
            thread.start()
            self._write_threads.append(thread)

        self.is_recording = False
        self.current_video_frames = []
        self.current_video_path = None
        self.video_count += 1
        self.trigger_type = None

    def finish_trial(self) -> None:
        if self.is_recording:
            self._finish_recording()

    def wait_for_writes(self) -> None:
        for thread in self._write_threads:
            thread.join()
        if self._write_errors:
            raise self._write_errors[0]


def _build_raw_state(
    env_base: ManagerBasedRlEnv,
    prev_action_shooter: torch.Tensor,
    prev_action_gk: torch.Tensor,
    raw_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scene = env_base.scene
    shooter = scene["shooter"]
    goalkeeper = scene["goalkeeper"]
    ball = scene["ball"]

    if raw_state is None:
        raw_state = {
            "shooter": {},
            "goalkeeper": {},
            "ball": {},
        }

    shooter_state = raw_state["shooter"]
    goalkeeper_state = raw_state["goalkeeper"]
    ball_state = raw_state["ball"]

    shooter_state["root_pos"] = _copy_tensor_1d_to_list(shooter.data.root_link_pos_w[0])
    shooter_state["root_quat"] = _copy_tensor_1d_to_list(shooter.data.root_link_quat_w[0])
    shooter_state["root_lin_vel"] = _copy_tensor_1d_to_list(shooter.data.root_link_lin_vel_w[0])
    shooter_state["root_ang_vel"] = _copy_tensor_1d_to_list(shooter.data.root_link_ang_vel_w[0])
    shooter_state["joint_pos"] = _copy_tensor_1d_to_list(shooter.data.joint_pos[0])
    shooter_state["joint_vel"] = _copy_tensor_1d_to_list(shooter.data.joint_vel[0])
    shooter_state["last_action"] = _copy_tensor_1d_to_list(prev_action_shooter[0])

    goalkeeper_state["root_pos"] = _copy_tensor_1d_to_list(goalkeeper.data.root_link_pos_w[0])
    goalkeeper_state["root_quat"] = _copy_tensor_1d_to_list(goalkeeper.data.root_link_quat_w[0])
    goalkeeper_state["root_lin_vel"] = _copy_tensor_1d_to_list(goalkeeper.data.root_link_lin_vel_w[0])
    goalkeeper_state["root_ang_vel"] = _copy_tensor_1d_to_list(goalkeeper.data.root_link_ang_vel_w[0])
    goalkeeper_state["joint_pos"] = _copy_tensor_1d_to_list(goalkeeper.data.joint_pos[0])
    goalkeeper_state["joint_vel"] = _copy_tensor_1d_to_list(goalkeeper.data.joint_vel[0])
    goalkeeper_state["last_action"] = _copy_tensor_1d_to_list(prev_action_gk[0])

    ball_state["pos"] = _copy_tensor_1d_to_list(ball.data.root_link_pos_w[0])
    ball_state["vel"] = _copy_tensor_1d_to_list(ball.data.root_link_vel_w[0, :3])

    return raw_state


def _make_shooter_robot(config: dict[str, Any]) -> Any:
    cfg = get_g1_robot_cfg()
    scene = config["scene"]
    cfg.init_state = replace(
        HOME_KEYFRAME,
        pos=_vec3(scene["shooter_pos"]),
        rot=_yaw_to_quat(float(scene["shooter_yaw"])),
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


def _make_goalkeeper_robot(config: dict[str, Any]) -> Any:
    cfg = get_gk_robot_cfg()
    scene = config["scene"]
    cfg.init_state = replace(
        cfg.init_state,
        pos=_vec3(scene["goalkeeper_pos"]),
        rot=_yaw_to_quat(float(scene["goalkeeper_yaw"])),
        joint_pos=_GK_DEFAULT_JOINT_POS,
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


def make_compete_env_cfg(config: dict[str, Any]) -> ManagerBasedRlEnvCfg:
    scene_cfg = config["scene"]
    sim_cfg = config["sim"]
    entities: dict[str, Any] = {
        "ground": get_ground_cfg(),
        "ball": get_ball_cfg(pos=_vec3(scene_cfg["ball_pos"])),
        "goal": get_goal_cfg(pos=_vec3(config["goal"]["pos"])),
        "shooter": _make_shooter_robot(config),
        "goalkeeper": _make_goalkeeper_robot(config),
    }

    actions: dict[str, ActionTermCfg] = {
        "shooter_joint_pos": JointPositionActionCfg(
            entity_name="shooter",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        ),
        "goalkeeper_joint_pos": JointPositionActionCfg(
            entity_name="goalkeeper",
            actuator_names=(".*",),
            scale=float(config["actions"]["goalkeeper_scale"]),
            use_default_offset=True,
        ),
    }

    events: dict[str, EventTermCfg] = {
        "reset_shooter_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": _SHOOTER_CFG},
        ),
        "reset_shooter_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("shooter", joint_names=(".*",)),
            },
        ),
        "reset_goalkeeper_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": _GK_CFG},
        ),
        "reset_goalkeeper_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("goalkeeper", joint_names=(".*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": _BALL_CFG},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=entities,
            num_envs=1,
            spec_fn=_add_soccer_scene_postproc,
        ),
        observations={
            "shooter_actor": ObservationGroupCfg(
                terms={
                    "dummy": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "shooter/imu_ang_vel"},
                    ),
                },
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
            "goalkeeper_actor": ObservationGroupCfg(
                terms={
                    "dummy": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "goalkeeper/imu_ang_vel"},
                    ),
                },
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
        },
        actions=actions,
        commands={},
        events=events,
        rewards={
            "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="shooter",
            body_name="torso_link",
            lookat=(0.0, 0.0, 1.0),
            distance=4.3,
            elevation=-8.0,
            azimuth=195.0,
            fovy=50.0,
            height=480,
            width=640,
        ),
        sim=SimulationCfg(
            nconmax=int(sim_cfg["nconmax"]),
            njmax=int(sim_cfg["njmax"]),
            contact_sensor_maxmatch=int(sim_cfg["contact_sensor_maxmatch"]),
            mujoco=MujocoCfg(
                timestep=float(sim_cfg["timestep"]),
                iterations=int(sim_cfg["iterations"]),
                ls_iterations=int(sim_cfg["ls_iterations"]),
                ccd_iterations=int(sim_cfg["ccd_iterations"]),
            ),
        ),
        decimation=int(sim_cfg["decimation"]),
        episode_length_s=float(config["episode_length_s"]),
    )


class ZeroPolicy:
    def __init__(self, action_dim: int, device: str):
        self._zero = torch.zeros(1, action_dim, device=device)

    def __call__(self, _input: Any) -> torch.Tensor:
        return self._zero

    def reset(self) -> None:
        pass


class ApiPolicy:
    def __init__(self, url: str, action_dim: int, device: str, timeout: float = 2.0):
        self._url = url.rstrip("/")
        self._action_dim = action_dim
        self._device = device
        self._timeout = timeout
        resp = requests.post(f"{self._url}/reset", json={}, timeout=self._timeout)
        resp.raise_for_status()
        print(f"[INFO] API connected: {self._url} (act_dim={action_dim})", flush=True)

    def __call__(self, raw_state: dict[str, Any]) -> torch.Tensor:
        resp = requests.post(f"{self._url}/act", json=raw_state, timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()
        action = torch.tensor(payload["action"], device=self._device, dtype=torch.float32)
        if action.shape != (1, self._action_dim):
            raise RuntimeError(
                f"{self._url}/act returned shape {tuple(action.shape)}, "
                f"expected (1, {self._action_dim})"
            )
        return action

    def reset(self) -> None:
        try:
            requests.post(f"{self._url}/reset", json={}, timeout=self._timeout)
        except requests.RequestException:
            pass


def _call_policies_parallel(
    shooter_policy: Any,
    goalkeeper_policy: Any,
    raw_state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    results: dict[str, torch.Tensor] = {}
    errors: list[BaseException] = []

    def _run(name: str, policy: Any) -> None:
        try:
            results[name] = policy(raw_state)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=_run, args=("shooter", shooter_policy), daemon=True),
        threading.Thread(target=_run, args=("goalkeeper", goalkeeper_policy), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results["shooter"], results["goalkeeper"]


class CombinedPolicy:
    def __init__(
        self,
        shooter_policy: Any,
        goalkeeper_policy: Any,
        env_base: ManagerBasedRlEnv,
        device: str,
    ):
        self._shooter = shooter_policy
        self._goalkeeper = goalkeeper_policy
        self._env_base = env_base
        self._prev_action_s = torch.zeros(1, 29, device=device)
        self._prev_action_g = torch.zeros(1, 29, device=device)
        self._raw_state_cache: dict[str, Any] = {"shooter": {}, "goalkeeper": {}, "ball": {}}

    def __call__(self, _obs: dict[str, Any]) -> torch.Tensor:
        raw = _build_raw_state(
            self._env_base,
            self._prev_action_s,
            self._prev_action_g,
            self._raw_state_cache,
        )
        s_act = self._shooter(raw)
        g_act = self._goalkeeper(raw)
        self._prev_action_s = s_act.detach().clone()
        self._prev_action_g = g_act.detach().clone()
        return torch.cat([s_act, g_act], dim=-1)

    def reset(self) -> None:
        self._shooter.reset()
        self._goalkeeper.reset()
        self._prev_action_s.zero_()
        self._prev_action_g.zero_()


class PassiveViserViewer(ViserPlayViewer):
    """Viser viewer that renders the environment without stepping physics."""

    def __init__(
        self,
        *args: Any,
        scoreboard: "ScoreboardState | None" = None,
        start_event: threading.Event | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._scoreboard = scoreboard
        self._start_event = start_event
        self._scoreboard_html = None
        self._start_button = None
        self._render_interval_s = 1.0 / 24.0
        self._next_render_at = 0.0

    def setup(self) -> None:
        super().setup()
        # Put the viewer behind the shooter so the ball travels toward the goal
        # in the direction the camera is facing. The fixed tournament scene uses
        # shooter at +x, goalkeeper at the origin, and goal at -x.
        _cam_pos = np.array([8.8, -0.8, 2.2])
        _look_at = np.array([1.8, 0.0, 0.9])

        # Disable the built-in camera tracking so it doesn't reset our
        # position whenever a client connects or the checkbox is toggled.
        self._scene.camera_tracking_enabled = False

        # Override camera for already-connected clients.
        for client in self._server.get_clients().values():
            client.camera.position = _cam_pos
            client.camera.look_at = _look_at

        # Override camera for any client that connects later.
        @self._server.on_client_connect
        def _(_client):
            _client.camera.position = _cam_pos
            _client.camera.look_at = _look_at

        if self._scoreboard is not None:
            import viser

            with self._server.gui.add_folder("Match Scoreboard"):
                if self._start_event is not None:
                    self._start_button = self._server.gui.add_button(
                        "Start Trials",
                        icon=viser.Icon.PLAYER_PLAY,
                        color="green",
                    )

                    @self._start_button.on_click
                    def _(_) -> None:
                        if self._start_event is None or self._start_event.is_set():
                            return
                        self._start_event.set()
                        self._scoreboard.set_phase("starting")
                        self._start_button.label = "Trials Started"
                        self._start_button.disabled = True

                self._scoreboard_html = self._server.gui.add_html("")
            self._update_scoreboard_display()

    def _step_physics(self, dt: float) -> None:
        del dt
        return

    def reset_environment(self) -> None:
        return

    def _update_scoreboard_display(self) -> None:
        if self._scoreboard_html is None or self._scoreboard is None:
            return
        self._scoreboard_html.content = self._scoreboard.to_html()

    def tick(self) -> bool:
        now = time.perf_counter()
        if self._next_render_at and now < self._next_render_at:
            time.sleep(min(self._next_render_at - now, self._render_interval_s))
            return False
        rendered = super().tick()
        self._next_render_at = time.perf_counter() + self._render_interval_s
        self._update_scoreboard_display()
        return rendered


@dataclass
class ScoreboardState:
    shooter_team: str
    goalkeeper_team: str
    total_trials: int
    current_trial: int = 0
    phase: str = "initializing"
    results: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_phase(self, phase: str, current_trial: int | None = None) -> None:
        with self.lock:
            self.phase = phase
            if current_trial is not None:
                self.current_trial = current_trial

    def record(self, winner: str) -> None:
        with self.lock:
            self.results.append(winner)

    def snapshot(self) -> tuple[int, str, list[str]]:
        with self.lock:
            return self.current_trial, self.phase, list(self.results)

    def to_html(self) -> str:
        current_trial, phase, results = self.snapshot()
        shooter_wins = sum(1 for r in results if r == "shooter")
        gk_wins = sum(1 for r in results if r == "goalkeeper")
        symbols = []
        for result in results:
            if result == "shooter":
                color, label = "#16a34a", "S"
            elif result == "goalkeeper":
                color, label = "#2563eb", "G"
            else:
                color, label = "#dc2626", "!"
            symbols.append(
                "<span style='display:inline-flex;align-items:center;justify-content:center;"
                "width:22px;height:22px;border-radius:50%;margin:2px;color:white;"
                f"font-weight:700;background:{color};'>{label}</span>"
            )
        for _ in range(max(0, self.total_trials - len(results))):
            symbols.append(
                "<span style='display:inline-flex;align-items:center;justify-content:center;"
                "width:22px;height:22px;border-radius:50%;margin:2px;color:#6b7280;"
                "font-weight:700;background:#e5e7eb;'>-</span>"
            )
        current = current_trial if current_trial > 0 else "-"
        return f"""
        <div style="font-size:0.9em;line-height:1.35;padding:0 1em 0.6em 1em;">
          <strong>{self.shooter_team}</strong> vs <strong>{self.goalkeeper_team}</strong><br/>
          <strong>Current trial:</strong> {current}/{self.total_trials}<br/>
          <strong>Phase:</strong> {phase}<br/>
          <strong>Score:</strong> Shooter {shooter_wins} - {gk_wins} Goalkeeper<br/>
          <div style="margin-top:6px;">{''.join(symbols)}</div>
          <div style="color:#6b7280;margin-top:4px;">S = shooter goal, G = goalkeeper save</div>
        </div>
        """


def _ball_entered_goal(ball_pos: torch.Tensor, config: dict[str, Any]) -> bool:
    goal = config["goal"]
    x, y, z = ball_pos[0].item(), ball_pos[1].item(), ball_pos[2].item()
    return (
        x <= float(goal["plane_x"])
        and abs(y) <= float(goal["half_width"])
        and z <= float(goal["height"])
    )


def _minimal_config_audit(config: dict[str, Any], max_steps: int) -> dict[str, Any]:
    return {
        "episode_length_s": float(config["episode_length_s"]),
        "max_steps": max_steps,
        "robot": {
            "joint_order": config["robot"]["joint_order"],
            "shooter_initial_joints": config["robot"]["shooter_initial_joints"],
            "goalkeeper_initial_joints": config["robot"]["goalkeeper_initial_joints"],
            "shooter_yaw": float(config["scene"]["shooter_yaw"]),
            "goalkeeper_yaw": float(config["scene"]["goalkeeper_yaw"]),
        },
        "ball": {
            "initial_pos": config["scene"]["ball_pos"],
        },
        "sim": {
            "timestep": float(config["sim"]["timestep"]),
            "decimation": int(config["sim"]["decimation"]),
            "step_dt": float(config["sim"]["timestep"]) * int(config["sim"]["decimation"]),
        },
        "ground_contact": config["ground_contact"],
    }


def run_trial(
    trial_index: int,
    env: RslRlVecEnvWrapper,
    env_base: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
    config: dict[str, Any],
    max_steps: int,
    step_dt: float,
    realtime: bool,
    scoreboard: ScoreboardState | None = None,
    video_recorder: _TimedVideoRecorder | None = None,
) -> dict[str, Any]:
    if scoreboard is not None:
        scoreboard.set_phase("running", trial_index)
    env.reset()
    shooter_policy.reset()
    goalkeeper_policy.reset()

    device = env.unwrapped.device
    prev_action_s = torch.zeros(1, 29, device=device)
    prev_action_g = torch.zeros(1, 29, device=device)
    ball = env.unwrapped.scene["ball"]
    raw_state_cache: dict[str, Any] = {"shooter": {}, "goalkeeper": {}, "ball": {}}
    goal_scored = False
    error: str | None = None
    steps = 0
    start_time = time.perf_counter()
    trial_start = start_time
    for _ in range(max_steps):
        try:
            with torch.inference_mode():
                raw = _build_raw_state(env_base, prev_action_s, prev_action_g, raw_state_cache)
                s_act, g_act = _call_policies_parallel(shooter_policy, goalkeeper_policy, raw)
        except Exception as exc:
            error = str(exc)
            break

        result = env.step(torch.cat([s_act, g_act], dim=-1))
        steps += 1
        prev_action_s = s_act.detach().clone()
        prev_action_g = g_act.detach().clone()

        ball_pos = ball.data.root_link_pos_w[0].cpu()
        if _ball_entered_goal(ball_pos, config):
            goal_scored = True
            break

        terminated = result[2]
        terminated = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        if terminated:
            break

        if realtime:
            target_time = start_time + steps * step_dt
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    elapsed_total = time.perf_counter() - trial_start
    winner = "shooter" if goal_scored else "goalkeeper"
    if error is not None:
        winner = "error"
    if scoreboard is not None:
        scoreboard.record(winner)
    ball_pos = ball.data.root_link_pos_w[0].cpu()
    return {
        "trial": trial_index,
        "winner": winner,
        "goal_scored": goal_scored,
        "steps": steps,
        "elapsed_s": elapsed_total,
        "ball_final_pos": ball_pos.tolist(),
        "error": error,
    }


def run_match(
    cfg: "CompeteConfig",
    config: dict[str, Any],
    env: RslRlVecEnvWrapper,
    env_base: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
    max_steps: int,
    step_dt: float,
    scoreboard: ScoreboardState | None = None,
    video_recorder: _TimedVideoRecorder | None = None,
) -> dict[str, Any]:
    print(f"[INFO] Running {cfg.num_trials} trials, max_steps={max_steps}", flush=True)
    trials: list[dict[str, Any]] = []
    goals = 0
    errors = 0
    for index in range(1, cfg.num_trials + 1):
        if hasattr(env_base, "episode_count"):
            try:
                env_base.episode_count = index - 1
            except Exception:
                pass
        stats = run_trial(
            index,
            env,
            env_base,
            shooter_policy,
            goalkeeper_policy,
            config,
            max_steps,
            step_dt,
            cfg.realtime,
            scoreboard,
        )
        trials.append(stats)
        if stats["goal_scored"]:
            goals += 1
        if stats["error"]:
            errors += 1
        print(
            f"[TRIAL {index}/{cfg.num_trials}] winner={stats['winner']} "
            f"goal={stats['goal_scored']} steps={stats['steps']} "
            f"elapsed={stats.get('elapsed_s', 0.0):.2f}s",
            flush=True,
        )
        if stats["error"]:
            print(f"[ERROR] {stats['error']}", flush=True)
        if video_recorder is not None:
            video_recorder.finish_trial()

    goalkeeper_wins = cfg.num_trials - goals - errors
    summary = {
        "num_trials": cfg.num_trials,
        "goals": goals,
        "goalkeeper_wins": goalkeeper_wins,
        "errors": errors,
        "winner_decision": "shooter" if goals > cfg.num_trials / 2 else "goalkeeper",
    }
    print(f"[SUMMARY] {summary}", flush=True)
    if scoreboard is not None:
        scoreboard.set_phase("finished")
    return {"summary": summary, "trials": trials}


def wait_for_start(
    start_event: threading.Event,
    scoreboard: ScoreboardState | None = None,
    auto_start: bool = False,
) -> None:
    if auto_start:
        start_event.set()
    if scoreboard is not None:
        scoreboard.set_phase("waiting for Start Trials", 0)
    print("[INFO] Ready. Waiting for Start Trials.", flush=True)
    start_event.wait()
    if scoreboard is not None:
        scoreboard.set_phase("starting", 0)
    print("[INFO] Start Trials pressed; beginning official trials.", flush=True)


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] Result JSON: {path}", flush=True)


@dataclass
class CompeteConfig:
    shooter_api: str | None = None
    goalkeeper_api: str | None = None
    shooter_team: str = "ShooterTeam"
    goalkeeper_team: str = "GoalkeeperTeam"
    match_id: str | None = None
    num_trials: int = 10
    config_path: str = str(DEFAULT_CONFIG_PATH)
    results_json: str | None = None
    viser_host: str = "0.0.0.0"
    viser_port: int = 7000
    no_viewer: bool = False
    save_video: bool = True
    request_timeout: float = 2.0
    realtime: bool = True
    device: str | None = None
    seed: int = 2810


def run_compete(cfg: CompeteConfig) -> dict[str, Any]:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    configure_torch_backends()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    config = _load_config(cfg.config_path)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    step_dt = float(config["sim"]["timestep"]) * int(config["sim"]["decimation"])
    max_steps = int(round(float(config["episode_length_s"]) / step_dt))
    timestamp = _utc_timestamp()

    # Determine match identity and output paths.
    # When launched by the tournament server cfg.match_id is already a unique
    # timestamped name — use it as-is.  For ad‑hoc runs generate one.
    if cfg.match_id:
        match_id = cfg.match_id
    else:
        match_id = (
            f"{timestamp}_"
            f"{_sanitize(cfg.shooter_team)}_shooter_vs_"
            f"{_sanitize(cfg.goalkeeper_team)}_goalkeeper"
        )
    match_dir = DEFAULT_RESULTS_DIR / match_id
    match_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(cfg.results_json) if cfg.results_json else (match_dir / "result.json")

    print(f"[INFO] Match id: {match_id}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Viser: http://{cfg.viser_host}:{cfg.viser_port}", flush=True)

    env_cfg = make_compete_env_cfg(config)
    # Only enable offscreen rendering when we actually save video. This keeps
    # the no-video path aligned with the old fast implementation.
    render_mode = "rgb_array" if cfg.save_video else None
    env_base_raw = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if cfg.save_video:
        env_base_raw._offline_renderer.close()
        env_base_raw._offline_renderer = None
    act_dim_shooter = env_base_raw.action_manager.get_term("shooter_joint_pos").action_dim
    act_dim_goalkeeper = env_base_raw.action_manager.get_term("goalkeeper_joint_pos").action_dim

    # Set up video recording only when requested.
    video_folder = match_dir / "videos"
    env_base = env_base_raw
    if cfg.save_video:
        video_folder.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Video recording: {video_folder}", flush=True)
    else:
        print("[INFO] Video recording disabled.", flush=True)

    # Disable real-time pacing in no-viewer mode for max throughput
    if cfg.no_viewer:
        cfg.realtime = False

    shooter_policy: Any
    goalkeeper_policy: Any
    try:
        shooter_policy = (
            ApiPolicy(cfg.shooter_api, act_dim_shooter, device, cfg.request_timeout)
            if cfg.shooter_api
            else ZeroPolicy(act_dim_shooter, device)
        )
        goalkeeper_policy = (
            ApiPolicy(cfg.goalkeeper_api, act_dim_goalkeeper, device, cfg.request_timeout)
            if cfg.goalkeeper_api
            else ZeroPolicy(act_dim_goalkeeper, device)
        )
        result_holder: dict[str, Any] = {}
        video_recorder: _TimedVideoRecorder | None = None
        if cfg.save_video:
            video_recorder = _TimedVideoRecorder(
                env_base_raw,
                video_folder=video_folder,
                episode_trigger=lambda ep: True,
                video_length=None,
                disable_logger=True,
                step_dt=step_dt,
                capture_fps=24.0,
            )
            env_base = video_recorder
        env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)
        done_event = threading.Event()
        start_event = threading.Event()
        scoreboard = ScoreboardState(
            shooter_team=cfg.shooter_team,
            goalkeeper_team=cfg.goalkeeper_team,
            total_trials=cfg.num_trials,
        )

        def _worker() -> None:
            # Re-initialise offscreen renderer in *this* thread so the EGL
            # context belongs to the worker thread where VideoRecorder calls
            # render() during env.step().
            from mjlab.viewer import OffscreenRenderer

            if cfg.save_video:
                env_base_raw._offline_renderer = OffscreenRenderer(
                    model=env_base_raw.sim.mj_model,
                    cfg=env_base_raw.cfg.viewer,
                    scene=env_base_raw.scene,
                )
                env_base_raw._offline_renderer.initialize()
            try:
                wait_for_start(
                    start_event,
                    scoreboard,
                    auto_start=cfg.no_viewer,
                )
                result_holder.update(
                    run_match(
                        cfg,
                        config,
                        env,
                        env_base,
                        shooter_policy,
                        goalkeeper_policy,
                        max_steps,
                        step_dt,
                        scoreboard,
                        video_recorder,
                    )
                )
            except Exception as exc:
                result_holder["summary"] = {
                    "num_trials": cfg.num_trials,
                    "goals": 0,
                    "goalkeeper_wins": 0,
                    "errors": cfg.num_trials,
                    "winner_decision": "error",
                }
                result_holder["trials"] = []
                result_holder["fatal_error"] = str(exc)
                print(f"[FATAL] {exc}", flush=True)
            finally:
                # Clean up the worker-thread EGL renderer.
                try:
                    if cfg.save_video and env_base_raw._offline_renderer is not None:
                        env_base_raw._offline_renderer.close()
                        env_base_raw._offline_renderer = None
                except Exception:
                    pass
                if "summary" not in result_holder:
                    scoreboard.set_phase("failed")
                done_event.set()

        worker = threading.Thread(target=_worker, name="phase2-match", daemon=True)
        worker.start()

        if not cfg.no_viewer:
            try:
                import viser

                server = viser.ViserServer(host=cfg.viser_host, port=cfg.viser_port, label="phase2")
                combined = CombinedPolicy(shooter_policy, goalkeeper_policy, env_base, device)
                viewer = PassiveViserViewer(
                    env,
                    combined,
                    viser_server=server,
                    scoreboard=scoreboard,
                    start_event=start_event,
                )
                viewer.setup()
                try:
                    while viewer.is_running() and not done_event.is_set():
                        if not viewer.tick():
                            continue
                        viewer._update_stats()
                finally:
                    viewer.close()
            except TypeError:
                print("[WARN] ViserServer host/port signature mismatch; running without viewer.", flush=True)
                start_event.set()
                done_event.wait()
        else:
            done_event.wait()

        worker.join(timeout=5.0)
        if cfg.save_video and video_recorder is not None:
            video_recorder.wait_for_writes()
        # Collect recorded video paths only when video saving is enabled.
        video_paths = []
        if cfg.save_video:
            video_paths = sorted(
                f"{match_id}/videos/{p.name}"
                for p in video_folder.glob("*.mp4")
            )
        payload = {
            "timestamp": timestamp,
            "match_id": match_id,
            "teams": {
                "shooter": cfg.shooter_team,
                "goalkeeper": cfg.goalkeeper_team,
            },
            "apis": {
                "shooter": cfg.shooter_api,
                "goalkeeper": cfg.goalkeeper_api,
            },
            "minimal_config_audit": _minimal_config_audit(config, max_steps),
            "videos": video_paths,
            **result_holder,
        }
        _write_result(result_path, payload)
        return payload
    except Exception as exc:
        # Try to collect any videos that were recorded before the crash
        try:
            video_paths_err = []
            if cfg.save_video:
                video_paths_err = sorted(
                    f"{match_id}/videos/{p.name}"
                    for p in video_folder.glob("*.mp4")
                )
        except Exception:
            video_paths_err = []
        payload = {
            "timestamp": timestamp,
            "match_id": match_id,
            "teams": {"shooter": cfg.shooter_team, "goalkeeper": cfg.goalkeeper_team},
            "apis": {"shooter": cfg.shooter_api, "goalkeeper": cfg.goalkeeper_api},
            "minimal_config_audit": _minimal_config_audit(config, max_steps),
            "videos": video_paths_err,
            "summary": {
                "num_trials": cfg.num_trials,
                "goals": 0,
                "goalkeeper_wins": 0,
                "errors": cfg.num_trials,
                "winner_decision": "error",
            },
            "trials": [],
            "fatal_error": str(exc),
        }
        _write_result(result_path, payload)
        raise
    finally:
        try:
            env_base.close()
        except Exception:
            pass


def main() -> None:
    args = tyro.cli(CompeteConfig, prog="phase2-compete")
    run_compete(args)


if __name__ == "__main__":
    main()
