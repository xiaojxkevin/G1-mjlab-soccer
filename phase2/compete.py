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

# Team → (R, G, B) in 0-1 (viser) and hex (web). 14 evenly-spaced hues.
_TEAM_COLOR_MAP: dict[str, tuple[float, float, float, str]] = {
    "我爱吃海参": (0.851, 0.212, 0.212, "#D93636"),       # H=0    red
    "TeamWHY": (0.851, 0.494, 0.212, "#D97E36"),         # H=26   orange
    "Team3": (0.749, 0.663, 0.180, "#BFA92E"),           # H=51   gold
    "T4": (0.502, 0.749, 0.180, "#80BF2E"),              # H=77   yellow-green
    "TEAM5": (0.212, 0.749, 0.290, "#36BF4A"),           # H=103  green
    "Team6": (0.180, 0.749, 0.451, "#2EBF73"),           # H=129  teal-green
    "Team7": (0.180, 0.749, 0.620, "#2EBF9E"),           # H=154  teal
    "Kedox": (0.180, 0.620, 0.749, "#2E9EBF"),           # H=180  cyan
    "Team9": (0.212, 0.451, 0.851, "#3673D9"),           # H=206  blue
    "Team10": (0.294, 0.212, 0.851, "#4B36D9"),          # H=231  indigo
    "G你太美": (0.502, 0.212, 0.851, "#8036D9"),         # H=257  purple
    "Team12": (0.749, 0.212, 0.749, "#BF36BF"),          # H=283  magenta
    "守不住的队发大财": (0.851, 0.212, 0.682, "#D936AE"), # H=309  pink
    "Team14": (0.851, 0.212, 0.400, "#D93666"),          # H=334  rose
}
_DEFAULT_TEAM_COLOR: tuple[float, float, float, str] = (0.40, 0.40, 0.45, "#666672")

# Ordered team list for ID assignment (indices 1-14).
_TEAM_ID_LIST: list[str] = [
    "我爱吃海参",   # 1
    "TeamWHY",       # 2
    "Team3",         # 3
    "T4",            # 4
    "TEAM5",         # 5
    "Team6",         # 6
    "Team7",         # 7
    "Kedox",         # 8
    "Team9",         # 9
    "Team10",        # 10
    "G你太美",       # 11
    "Team12",        # 12
    "守不住的队发大财", # 13
    "Team14",        # 14
]
_TEAM_NAME_TO_ID: dict[str, int] = {name: i + 1 for i, name in enumerate(_TEAM_ID_LIST)}


def _team_display_name(name: str) -> str:
    """Return a display-safe name (ID for Chinese, original for English)."""
    tid = _TEAM_NAME_TO_ID.get(name)
    if tid is not None:
        return f"Team{tid}"
    return name


def _team_color_rgb(name: str) -> tuple[float, float, float]:
    """Return (R, G, B) in 0-1 range for a team name, or default gray."""
    entry = _TEAM_COLOR_MAP.get(name, _DEFAULT_TEAM_COLOR)
    return (entry[0], entry[1], entry[2])


def _team_color_hex(name: str) -> str:
    """Return hex color string for a team name, or default gray."""
    entry = _TEAM_COLOR_MAP.get(name, _DEFAULT_TEAM_COLOR)
    return entry[3]


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


class _PenaltyCombinedPolicy:
    """Combined policy for the penalty shootout Viser viewer.

    On each call returns the concatenated actions of the currently active
    shooter + goalkeeper pair.  Which pair is active is rotated by the
    regulation schedule and then alternates for sudden death.
    """

    def __init__(
        self,
        regulation_schedule: list[tuple[int, str, str, Any, Any]],
        policy_a_shooter: Any,
        policy_a_goalkeeper: Any,
        policy_b_shooter: Any,
        policy_b_goalkeeper: Any,
        env_base: ManagerBasedRlEnv,
        device: str,
    ):
        self._regulation_schedule = regulation_schedule
        self._policy_a_shooter = policy_a_shooter
        self._policy_a_goalkeeper = policy_a_goalkeeper
        self._policy_b_shooter = policy_b_shooter
        self._policy_b_goalkeeper = policy_b_goalkeeper
        self._env_base = env_base
        self._device = device
        self._prev_action_s = torch.zeros(1, 29, device=device)
        self._prev_action_g = torch.zeros(1, 29, device=device)
        self._raw_state_cache: dict[str, Any] = {"shooter": {}, "goalkeeper": {}, "ball": {}}
        # Pick the first pair from regulation schedule for the viewer
        self._active_shooter = regulation_schedule[0][3]
        self._active_goalkeeper = regulation_schedule[0][4]
        self._kick_counter = 0

    def __call__(self, _obs: dict[str, Any]) -> torch.Tensor:
        raw = _build_raw_state(
            self._env_base,
            self._prev_action_s,
            self._prev_action_g,
            self._raw_state_cache,
        )
        s_act = self._active_shooter(raw)
        g_act = self._active_goalkeeper(raw)
        self._prev_action_s = s_act.detach().clone()
        self._prev_action_g = g_act.detach().clone()
        return torch.cat([s_act, g_act], dim=-1)

    def reset(self) -> None:
        self._policy_a_shooter.reset()
        self._policy_a_goalkeeper.reset()
        self._policy_b_shooter.reset()
        self._policy_b_goalkeeper.reset()
        self._prev_action_s.zero_()
        self._prev_action_g.zero_()


class PassiveViserViewer(ViserPlayViewer):
    """Viser viewer that renders the environment without stepping physics."""

    def __init__(
        self,
        *args: Any,
        scoreboard: "ScoreboardState | None" = None,
        start_event: threading.Event | None = None,
        shooter_team_name: str = "",
        goalkeeper_team_name: str = "",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._scoreboard = scoreboard
        self._start_event = start_event
        self._scoreboard_html = None
        self._start_button = None
        self._render_interval_s = 1.0 / 24.0
        self._next_render_at = 0.0
        self._shooter_team_name = shooter_team_name
        self._goalkeeper_team_name = goalkeeper_team_name
        self._disc_shooter: Any = None
        self._disc_goalkeeper: Any = None
        self._last_shooter_color: tuple[float, float, float] | None = None
        self._last_gk_color: tuple[float, float, float] | None = None

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

        # --- Add coloured foot discs under each robot ---
        # Use z=0.025 so the disc sits clearly above the ground plane and
        # catches enough light to not render black.
        default_rgb = _team_color_rgb("")
        self._disc_shooter = self._server.scene.add_cylinder(
            "shooter_foot_disc",
            radius=0.35,
            height=0.025,
            color=default_rgb,
            position=(0, 0, 0.025),
            visible=True,
        )
        self._disc_goalkeeper = self._server.scene.add_cylinder(
            "goalkeeper_foot_disc",
            radius=0.35,
            height=0.025,
            color=default_rgb,
            position=(0, 0, 0.025),
            visible=True,
        )

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

    def _update_foot_discs(self) -> None:
        """Update foot disc positions and colours from current robot state.

        Viser handles don't support changing material colour after creation,
        so we remove + re-add the cylinder when the colour changes.
        """
        # Access the root MuJoCo scene through the env chain.
        # self.env is RslRlVecEnvWrapper → .unwrapped chains through
        # _TimedVideoRecorder (if present) → ManagerBasedRlEnv.
        env_unwrapped = self.env.unwrapped
        scene = env_unwrapped.scene
        s_pos = scene["shooter"].data.root_link_pos_w[0].cpu().numpy()
        g_pos = scene["goalkeeper"].data.root_link_pos_w[0].cpu().numpy()

        # Determine which team controls which entity.
        shooter_color_team = self._shooter_team_name
        gk_color_team = self._goalkeeper_team_name
        if isinstance(self._scoreboard, PenaltyScoreboardState):
            # Penalty mode: colours change per kick.
            sd = self._scoreboard
            shooting = sd.shooting_team
            if shooting == sd.team_a_name:
                shooter_color_team = sd.team_a_name
                gk_color_team = sd.team_b_name
            elif shooting == sd.team_b_name:
                shooter_color_team = sd.team_b_name
                gk_color_team = sd.team_a_name

        new_shooter_rgb = _team_color_rgb(shooter_color_team)
        new_gk_rgb = _team_color_rgb(gk_color_team)

        # --- Shooter disc: recreate if colour changed ---
        if new_shooter_rgb != self._last_shooter_color:
            if self._disc_shooter is not None:
                self._disc_shooter.remove()
            self._disc_shooter = self._server.scene.add_cylinder(
                "shooter_foot_disc",
                radius=0.35,
                height=0.025,
                color=new_shooter_rgb,
                position=(float(s_pos[0]), float(s_pos[1]), 0.025),
                visible=True,
            )
            self._last_shooter_color = new_shooter_rgb
        else:
            self._disc_shooter.position = (float(s_pos[0]), float(s_pos[1]), 0.025)

        # --- Goalkeeper disc: recreate if colour changed ---
        if new_gk_rgb != self._last_gk_color:
            if self._disc_goalkeeper is not None:
                self._disc_goalkeeper.remove()
            self._disc_goalkeeper = self._server.scene.add_cylinder(
                "goalkeeper_foot_disc",
                radius=0.35,
                height=0.025,
                color=new_gk_rgb,
                position=(float(g_pos[0]), float(g_pos[1]), 0.025),
                visible=True,
            )
            self._last_gk_color = new_gk_rgb
        else:
            self._disc_goalkeeper.position = (float(g_pos[0]), float(g_pos[1]), 0.025)

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
        self._update_foot_discs()
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
        s_name = _team_display_name(self.shooter_team)
        g_name = _team_display_name(self.goalkeeper_team)
        return f"""
        <div style="font-size:0.9em;line-height:1.35;padding:0 1em 0.6em 1em;">
          <strong>{s_name}</strong> vs <strong>{g_name}</strong><br/>
          <strong>Current trial:</strong> {current}/{self.total_trials}<br/>
          <strong>Phase:</strong> {phase}<br/>
          <strong>Score:</strong> Shooter {shooter_wins} - {gk_wins} Goalkeeper<br/>
          <div style="margin-top:6px;">{''.join(symbols)}</div>
          <div style="color:#6b7280;margin-top:4px;">S = shooter goal, G = goalkeeper save</div>
        </div>
        """


@dataclass
class PenaltyScoreboardState:
    """Scoreboard for penalty shootout matches displayed in the Viser viewer."""

    team_a_name: str
    team_b_name: str
    score_a: int = 0
    score_b: int = 0
    current_kick: int = 0
    total_kicks: int = 10
    shooting_team: str = ""
    phase: str = "initializing"
    kick_results: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_phase(self, phase: str, current_kick: int | None = None) -> None:
        with self.lock:
            self.phase = phase
            if current_kick is not None:
                self.current_kick = current_kick

    def record_kick(self, result: str) -> None:
        with self.lock:
            self.kick_results.append(result)

    def snapshot(self) -> tuple[int, str, list[str]]:
        with self.lock:
            return self.current_kick, self.phase, list(self.kick_results)

    def to_html(self) -> str:
        current_kick, phase, results = self.snapshot()
        symbols = []
        for result in results:
            if result == "goal":
                color, label = "#16a34a", "G"
            elif result == "save":
                color, label = "#2563eb", "S"
            else:
                color, label = "#dc2626", "!"
            symbols.append(
                "<span style='display:inline-flex;align-items:center;justify-content:center;"
                "width:22px;height:22px;border-radius:50%;margin:2px;color:white;"
                f"font-weight:700;background:{color};'>{label}</span>"
            )
        for _ in range(max(0, self.total_kicks - len(results))):
            symbols.append(
                "<span style='display:inline-flex;align-items:center;justify-content:center;"
                "width:22px;height:22px;border-radius:50%;margin:2px;color:#6b7280;"
                "font-weight:700;background:#e5e7eb;'>-</span>"
            )
        # Show additional sudden-death circles when phase is sudden_death
        if phase == "sudden_death":
            for _ in range(4):
                symbols.append(
                    "<span style='display:inline-flex;align-items:center;justify-content:center;"
                    "width:22px;height:22px;border-radius:50%;margin:2px;color:#6b7280;"
                    "font-weight:700;background:#e5e7eb;'>?</span>"
                )
        current = current_kick if current_kick > 0 else "-"
        shooting = self.shooting_team or "-"
        shooting_display = _team_display_name(shooting)
        total_display = f"{self.total_kicks}+SD" if phase == "sudden_death" else str(self.total_kicks)
        name_a = _team_display_name(self.team_a_name)
        name_b = _team_display_name(self.team_b_name)
        return f"""
        <div style="font-size:0.9em;line-height:1.35;padding:0 1em 0.6em 1em;">
          <strong>{name_a}</strong> vs <strong>{name_b}</strong><br/>
          <strong>Score:</strong> {name_a} {self.score_a} - {self.score_b} {name_b}<br/>
          <strong>Current kick:</strong> {current}/{total_display} &nbsp; <strong>Shooting:</strong> {shooting_display}<br/>
          <strong>Phase:</strong> {phase}<br/>
          <div style="margin-top:6px;">{''.join(symbols)}</div>
          <div style="color:#6b7280;margin-top:4px;">G = goal scored, S = goalkeeper save</div>
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


def _rename_last_video(
    video_folder: Path,
    video_recorder: _TimedVideoRecorder,
    kick_idx: int,
    shooting_team: str,
    keeping_team: str,
    is_sudden_death: bool = False,
) -> Path | None:
    """Wait for the last video write to finish, then rename it with kick metadata."""
    video_recorder.wait_for_writes()
    existing = sorted(video_folder.glob("*.mp4"))
    if not existing:
        return None
    latest = existing[-1]
    phase = "sd" if is_sudden_death else "reg"
    # Use team display names (Team1..Team14) for filename safety.
    s_disp = _team_display_name(shooting_team).replace(" ", "")
    k_disp = _team_display_name(keeping_team).replace(" ", "")
    new_name = (
        f"kick_{kick_idx:02d}_{phase}_"
        f"{s_disp}_shoots_"
        f"{k_disp}_keeps.mp4"
    )
    new_path = latest.with_name(new_name)
    latest.rename(new_path)
    return new_path


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
    goal_scored_at_step: int | None = None
    error: str | None = None
    steps = 0
    start_time = time.perf_counter()
    trial_start = start_time
    post_goal_steps = max(1, int(round(1.0 / step_dt)))
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
            if goal_scored_at_step is None:
                goal_scored_at_step = steps
        if goal_scored_at_step is not None and steps - goal_scored_at_step >= post_goal_steps:
            break

        terminated = result[2]
        terminated = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        if terminated and not goal_scored:
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
        "goal_scored_at_step": goal_scored_at_step,
        "steps": steps,
        "elapsed_s": elapsed_total,
        "ball_final_pos": ball_pos.tolist(),
        "error": error,
    }


def run_penalty_kick(
    kick_index: int,
    shooting_team: str,
    keeping_team: str,
    env: RslRlVecEnvWrapper,
    env_base: ManagerBasedRlEnv,
    active_shooter_policy: Any,
    active_goalkeeper_policy: Any,
    config: dict[str, Any],
    max_steps: int,
    step_dt: float,
    realtime: bool,
    scoreboard: PenaltyScoreboardState | None = None,
) -> dict[str, Any]:
    """Run a single penalty kick: one shooter vs one goalkeeper."""
    if scoreboard is not None:
        scoreboard.set_phase("running", kick_index)
        scoreboard.shooting_team = shooting_team
    env.reset()
    active_shooter_policy.reset()
    active_goalkeeper_policy.reset()

    device = env.unwrapped.device
    prev_action_s = torch.zeros(1, 29, device=device)
    prev_action_g = torch.zeros(1, 29, device=device)
    ball = env.unwrapped.scene["ball"]
    raw_state_cache: dict[str, Any] = {"shooter": {}, "goalkeeper": {}, "ball": {}}
    goal_scored = False
    goal_scored_at_step: int | None = None
    error: str | None = None
    steps = 0
    start_time = time.perf_counter()
    kick_start = start_time
    post_goal_steps = max(1, int(round(1.0 / step_dt)))
    for _ in range(max_steps):
        try:
            with torch.inference_mode():
                raw = _build_raw_state(env_base, prev_action_s, prev_action_g, raw_state_cache)
                s_act, g_act = _call_policies_parallel(
                    active_shooter_policy, active_goalkeeper_policy, raw
                )
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
            if goal_scored_at_step is None:
                goal_scored_at_step = steps
        if goal_scored_at_step is not None and steps - goal_scored_at_step >= post_goal_steps:
            break

        terminated = result[2]
        terminated = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        if terminated and not goal_scored:
            break

        if realtime:
            target_time = start_time + steps * step_dt
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    elapsed_total = time.perf_counter() - kick_start
    kick_result = "goal" if goal_scored else "save"
    if error is not None:
        kick_result = "error"
    if scoreboard is not None:
        if goal_scored:
            scoreboard.record_kick("goal")
        elif error:
            scoreboard.record_kick("error")
        else:
            scoreboard.record_kick("save")
    ball_pos = ball.data.root_link_pos_w[0].cpu()
    return {
        "kick": kick_index,
        "shooting_team": shooting_team,
        "keeping_team": keeping_team,
        "goal_scored": goal_scored,
        "goal_scored_at_step": goal_scored_at_step,
        "steps": steps,
        "elapsed_s": elapsed_total,
        "ball_final_pos": ball_pos.tolist(),
        "error": error,
    }


def run_penalty_shootout(
    cfg: "CompeteConfig",
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a full penalty shootout between two teams."""
    timestamp = _utc_timestamp()

    # --- Validate config ---
    missing = []
    for field_name in [
        "team_a_name", "team_a_shooter_api", "team_a_goalkeeper_api",
        "team_b_name", "team_b_shooter_api", "team_b_goalkeeper_api",
    ]:
        if getattr(cfg, field_name) is None:
            missing.append(field_name)
    if missing:
        raise ValueError(f"Penalty shootout requires: {', '.join(missing)}")

    # --- Match identity ---
    if cfg.match_id:
        match_id = cfg.match_id
    else:
        match_id = (
            f"{timestamp}_"
            f"{_sanitize(cfg.team_a_name)}_vs_"
            f"{_sanitize(cfg.team_b_name)}_penalty"
        )
    match_dir = DEFAULT_RESULTS_DIR / match_id
    match_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(cfg.results_json) if cfg.results_json else (match_dir / "result.json")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    step_dt = float(config["sim"]["timestep"]) * int(config["sim"]["decimation"])
    max_steps = int(round(float(config["episode_length_s"]) / step_dt))

    print(f"[INFO] Match id: {match_id}", flush=True)
    print(f"[INFO] Mode: penalty_shootout", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Viser: http://{cfg.viser_host}:{cfg.viser_port}", flush=True)

    # --- Build env ---
    env_cfg = make_compete_env_cfg(config)
    render_mode = "rgb_array" if cfg.save_video else None
    env_base_raw = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if cfg.save_video:
        env_base_raw._offline_renderer.close()
        env_base_raw._offline_renderer = None
    act_dim_shooter = env_base_raw.action_manager.get_term("shooter_joint_pos").action_dim
    act_dim_goalkeeper = env_base_raw.action_manager.get_term("goalkeeper_joint_pos").action_dim

    # --- Video recording ---
    video_folder = match_dir / "videos"
    env_base = env_base_raw
    if cfg.save_video:
        video_folder.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Video recording: {video_folder}", flush=True)
    else:
        print("[INFO] Video recording disabled.", flush=True)

    if cfg.no_viewer:
        cfg.realtime = False

    try:
        # --- Create 4 ApiPolicies ---
        policy_a_shooter: Any = (
            ApiPolicy(cfg.team_a_shooter_api, act_dim_shooter, device, cfg.request_timeout)
            if cfg.team_a_shooter_api
            else ZeroPolicy(act_dim_shooter, device)
        )
        policy_a_goalkeeper: Any = (
            ApiPolicy(cfg.team_a_goalkeeper_api, act_dim_goalkeeper, device, cfg.request_timeout)
            if cfg.team_a_goalkeeper_api
            else ZeroPolicy(act_dim_goalkeeper, device)
        )
        policy_b_shooter: Any = (
            ApiPolicy(cfg.team_b_shooter_api, act_dim_shooter, device, cfg.request_timeout)
            if cfg.team_b_shooter_api
            else ZeroPolicy(act_dim_shooter, device)
        )
        policy_b_goalkeeper: Any = (
            ApiPolicy(cfg.team_b_goalkeeper_api, act_dim_goalkeeper, device, cfg.request_timeout)
            if cfg.team_b_goalkeeper_api
            else ZeroPolicy(act_dim_goalkeeper, device)
        )

        # --- Wrap env ---
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
        scoreboard = PenaltyScoreboardState(
            team_a_name=cfg.team_a_name,
            team_b_name=cfg.team_b_name,
            total_kicks=10,
        )

        # --- Pre-define kick schedule for regulation ---
        # Kick 1,3,5,7,9: team_a shoots, team_b keeps
        # Kick 2,4,6,8,10: team_b shoots, team_a keeps
        regulation_schedule: list[tuple[int, str, str, Any, Any]] = []
        for i in range(5):
            ki = i * 2 + 1
            regulation_schedule.append(
                (ki, cfg.team_a_name, cfg.team_b_name, policy_a_shooter, policy_b_goalkeeper)
            )
            ki = i * 2 + 2
            regulation_schedule.append(
                (ki, cfg.team_b_name, cfg.team_a_name, policy_b_shooter, policy_a_goalkeeper)
            )

        def _worker() -> None:
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
                    None,
                    auto_start=cfg.no_viewer,
                )
                if scoreboard is not None:
                    scoreboard.set_phase("regulation", 1)

                # --- Regulation kicks ---
                score_a = 0
                score_b = 0
                regulation_results: list[dict[str, Any]] = []
                sudden_death_results: list[dict[str, Any]] = []
                total_errors = 0
                kick_video_map: dict[int, str] = {}

                for kick_idx, shooting, keeping, s_pol, g_pol in regulation_schedule:
                    if scoreboard is not None:
                        scoreboard.score_a = score_a
                        scoreboard.score_b = score_b
                    stats = run_penalty_kick(
                        kick_idx, shooting, keeping,
                        env, env_base, s_pol, g_pol,
                        config, max_steps, step_dt, cfg.realtime,
                        scoreboard,
                    )
                    regulation_results.append(stats)
                    if stats["goal_scored"]:
                        if shooting == cfg.team_a_name:
                            score_a += 1
                        else:
                            score_b += 1
                    if stats["error"]:
                        total_errors += 1
                    result_str = "GOAL" if stats["goal_scored"] else "SAVE"
                    if stats["error"]:
                        result_str = "ERROR"
                    print(
                        f"[KICK {kick_idx}/10] [{result_str}] {shooting} (shooter) vs {keeping} (keeper) "
                        f"goal={stats['goal_scored']} steps={stats['steps']} "
                        f"score={cfg.team_a_name} {score_a}-{score_b} {cfg.team_b_name}",
                        flush=True,
                    )
                    if stats["error"]:
                        print(f"[ERROR] Kick {kick_idx}: {stats['error']}", flush=True)
                    if video_recorder is not None:
                        video_recorder.finish_trial()
                        renamed = _rename_last_video(
                            video_folder, video_recorder,
                            kick_idx, shooting, keeping,
                        )
                        if renamed is not None:
                            kick_video_map[kick_idx] = f"{match_id}/videos/{renamed.name}"

                # --- Sudden death ---
                sd_kick_idx = 11
                while (
                    score_a == score_b
                    and sd_kick_idx - 11 < cfg.max_sudden_death_rounds * 2
                ):
                    if scoreboard is not None:
                        scoreboard.set_phase("sudden_death", sd_kick_idx)
                        scoreboard.total_kicks = 10 + (sd_kick_idx - 11) + 1
                        scoreboard.score_a = score_a
                        scoreboard.score_b = score_b

                    # Team A shoots first in sudden death
                    stats_a = run_penalty_kick(
                        sd_kick_idx,
                        cfg.team_a_name, cfg.team_b_name,
                        env, env_base, policy_a_shooter, policy_b_goalkeeper,
                        config, max_steps, step_dt, cfg.realtime,
                        scoreboard,
                    )
                    sudden_death_results.append(stats_a)
                    if stats_a["goal_scored"]:
                        score_a += 1
                    if stats_a["error"]:
                        total_errors += 1
                    result_str_a = "GOAL" if stats_a["goal_scored"] else "SAVE"
                    print(
                        f"[SD KICK {sd_kick_idx}] [{result_str_a}] {cfg.team_a_name} (shooter) vs {cfg.team_b_name} (keeper) "
                        f"goal={stats_a['goal_scored']} "
                        f"score={cfg.team_a_name} {score_a}-{score_b} {cfg.team_b_name}",
                        flush=True,
                    )
                    if video_recorder is not None:
                        video_recorder.finish_trial()
                        renamed = _rename_last_video(
                            video_folder, video_recorder,
                            sd_kick_idx, cfg.team_a_name, cfg.team_b_name,
                            is_sudden_death=True,
                        )
                        if renamed is not None:
                            kick_video_map[sd_kick_idx] = f"{match_id}/videos/{renamed.name}"
                    sd_kick_idx += 1

                    # Team B shoots
                    stats_b = run_penalty_kick(
                        sd_kick_idx,
                        cfg.team_b_name, cfg.team_a_name,
                        env, env_base, policy_b_shooter, policy_a_goalkeeper,
                        config, max_steps, step_dt, cfg.realtime,
                        scoreboard,
                    )
                    sudden_death_results.append(stats_b)
                    if stats_b["goal_scored"]:
                        score_b += 1
                    if stats_b["error"]:
                        total_errors += 1
                    result_str_b = "GOAL" if stats_b["goal_scored"] else "SAVE"
                    print(
                        f"[SD KICK {sd_kick_idx}] [{result_str_b}] {cfg.team_b_name} (shooter) vs {cfg.team_a_name} (keeper) "
                        f"goal={stats_b['goal_scored']} "
                        f"score={cfg.team_a_name} {score_a}-{score_b} {cfg.team_b_name}",
                        flush=True,
                    )
                    if video_recorder is not None:
                        video_recorder.finish_trial()
                        renamed = _rename_last_video(
                            video_folder, video_recorder,
                            sd_kick_idx, cfg.team_b_name, cfg.team_a_name,
                            is_sudden_death=True,
                        )
                        if renamed is not None:
                            kick_video_map[sd_kick_idx] = f"{match_id}/videos/{renamed.name}"
                    sd_kick_idx += 1

                    # Check for winner after each pair
                    if score_a != score_b:
                        break

                # --- Determine winner ---
                if score_a > score_b:
                    winner = cfg.team_a_name
                elif score_b > score_a:
                    winner = cfg.team_b_name
                else:
                    winner = "draw"

                summary = {
                    "winner": winner,
                    "score_a": score_a,
                    "score_b": score_b,
                    "regulation_kicks": len(regulation_results),
                    "sudden_death_kicks": len(sudden_death_results),
                    "errors": total_errors,
                }
                result_holder["summary"] = summary
                result_holder["trials"] = regulation_results
                result_holder["sudden_death_trials"] = sudden_death_results
                result_holder["score"] = {
                    cfg.team_a_name: score_a,
                    cfg.team_b_name: score_b,
                }
                result_holder["kick_video_map"] = kick_video_map
                print(f"[SUMMARY] {summary}", flush=True)
                if scoreboard is not None:
                    scoreboard.set_phase("finished")
                    scoreboard.score_a = score_a
                    scoreboard.score_b = score_b
            except Exception as exc:
                result_holder["summary"] = {
                    "winner": "error",
                    "score_a": 0,
                    "score_b": 0,
                    "regulation_kicks": 0,
                    "sudden_death_kicks": 0,
                    "errors": 10,
                }
                result_holder["trials"] = []
                result_holder["sudden_death_trials"] = []
                result_holder["score"] = {}
                result_holder["fatal_error"] = str(exc)
                print(f"[FATAL] {exc}", flush=True)
            finally:
                try:
                    if cfg.save_video and env_base_raw._offline_renderer is not None:
                        env_base_raw._offline_renderer.close()
                        env_base_raw._offline_renderer = None
                except Exception:
                    pass
                if "summary" not in result_holder:
                    if scoreboard is not None:
                        scoreboard.set_phase("failed")
                done_event.set()

        worker = threading.Thread(target=_worker, name="phase2-penalty", daemon=True)
        worker.start()

        # --- Viewer ---
        if not cfg.no_viewer:
            try:
                import viser

                server = viser.ViserServer(host=cfg.viser_host, port=cfg.viser_port, label="phase2")
                # Use a CombinedPolicy-like wrapper that dynamically picks the
                # correct pair of policies based on the current kick.
                combined = _PenaltyCombinedPolicy(
                    regulation_schedule,
                    policy_a_shooter, policy_a_goalkeeper,
                    policy_b_shooter, policy_b_goalkeeper,
                    env_base, device,
                )
                viewer = PassiveViserViewer(
                    env,
                    combined,
                    viser_server=server,
                    scoreboard=scoreboard,
                    start_event=start_event,
                    shooter_team_name=cfg.team_a_name or "",
                    goalkeeper_team_name=cfg.team_b_name or "",
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
                print(
                    "[WARN] ViserServer host/port signature mismatch; "
                    "running without viewer.",
                    flush=True,
                )
                start_event.set()
                done_event.wait()
        else:
            done_event.wait()

        worker.join(timeout=5.0)
        if cfg.save_video and video_recorder is not None:
            video_recorder.wait_for_writes()
        video_paths: list[str] = []
        if cfg.save_video:
            video_paths = sorted(
                f"{match_id}/videos/{p.name}"
                for p in video_folder.glob("*.mp4")
            )
        payload = {
            "timestamp": timestamp,
            "match_id": match_id,
            "mode": "penalty_shootout",
            "teams": {
                "team_a": {
                    "name": cfg.team_a_name,
                    "shooter_api": cfg.team_a_shooter_api,
                    "goalkeeper_api": cfg.team_a_goalkeeper_api,
                },
                "team_b": {
                    "name": cfg.team_b_name,
                    "shooter_api": cfg.team_b_shooter_api,
                    "goalkeeper_api": cfg.team_b_goalkeeper_api,
                },
            },
            "minimal_config_audit": _minimal_config_audit(config, max_steps),
            "videos": video_paths,
            **result_holder,
        }
        _write_result(result_path, payload)
        return payload
    except Exception as exc:
        try:
            video_paths_err: list[str] = []
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
            "mode": "penalty_shootout",
            "teams": {
                "team_a": {
                    "name": cfg.team_a_name,
                    "shooter_api": cfg.team_a_shooter_api,
                    "goalkeeper_api": cfg.team_a_goalkeeper_api,
                },
                "team_b": {
                    "name": cfg.team_b_name,
                    "shooter_api": cfg.team_b_shooter_api,
                    "goalkeeper_api": cfg.team_b_goalkeeper_api,
                },
            },
            "minimal_config_audit": _minimal_config_audit(config, max_steps),
            "videos": video_paths_err,
            "summary": {
                "winner": "error",
                "score_a": 0,
                "score_b": 0,
                "regulation_kicks": 0,
                "sudden_death_kicks": 0,
                "errors": 10,
            },
            "trials": [],
            "sudden_death_trials": [],
            "score": {},
            "fatal_error": str(exc),
        }
        _write_result(result_path, payload)
        raise
    finally:
        try:
            env_base.close()
        except Exception:
            pass


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
    # --- Shared fields ---
    mode: str = "compete"               # "compete" | "penalty_shootout"
    match_id: str | None = None
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

    # --- Compete-mode fields ---
    shooter_api: str | None = None
    goalkeeper_api: str | None = None
    shooter_team: str = "ShooterTeam"
    goalkeeper_team: str = "GoalkeeperTeam"
    num_trials: int = 5

    # --- Penalty-shootout fields ---
    team_a_name: str | None = None
    team_a_shooter_api: str | None = None
    team_a_goalkeeper_api: str | None = None
    team_b_name: str | None = None
    team_b_shooter_api: str | None = None
    team_b_goalkeeper_api: str | None = None
    max_sudden_death_rounds: int = 5


def run_compete(cfg: CompeteConfig) -> dict[str, Any]:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    configure_torch_backends()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # --- Penalty shootout mode: delegate to dedicated orchestrator ---
    if cfg.mode == "penalty_shootout":
        config = _load_config(cfg.config_path)
        return run_penalty_shootout(cfg, config)

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
                    shooter_team_name=cfg.shooter_team or "",
                    goalkeeper_team_name=cfg.goalkeeper_team or "",
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
