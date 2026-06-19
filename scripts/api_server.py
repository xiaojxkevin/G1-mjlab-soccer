"""Reference policy server for Phase 2 tournament.

Implements the standard REST API that ``compete.py`` calls during cross-evaluation.

Receives raw MuJoCo state (both robots + ball) and computes its own observation
tensor.  Teams should customize ``compute_obs()`` to match their training setup.

  POST /act    - receive raw state, return action
  POST /reset  - reset policy hidden state and history buffer

Usage:
  # Shooter server
  python scripts/api_server.py --checkpoint shooter.pt --port 8000 --task shooter

  # Goalkeeper server
  python scripts/api_server.py --checkpoint goalkeeper.pt --port 8001 --task goalkeeper

Test with curl:
  curl -X POST http://localhost:8000/reset
  curl -X POST http://localhost:8000/act \\
       -H "Content-Type: application/json" \\
       -d '{"shooter":{"root_pos":[4,0,0.8],...},"goalkeeper":{...},"ball":{...}}'

-------------------------------------------------------------------------------
CUSTOMIZATION GUIDE
-------------------------------------------------------------------------------

Teams MUST customize ``compute_obs()`` to match their policy's observation
space.  The default implementation computes a standard proprioception + ball
observation.  If your policy uses different terms, scaling factors, reference
frames, or history length, update the function accordingly.
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import os
from typing import Any

import torch
import tyro
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from src.tasks.soccer.mdp.goalkeeper_obs import _GK_JOINT_NAMES, _REF_DEFAULT_DOF_POS

# ---------------------------------------------------------------------------
# Default joint positions  (match training configs)
# ---------------------------------------------------------------------------

_SHOOTER_DEFAULT_JOINT_POS = torch.zeros(29, dtype=torch.float32)

_GK_DEFAULT_JOINT_POS = torch.tensor(_REF_DEFAULT_DOF_POS, dtype=torch.float32)

_GK_STATIC_GUARD_ENABLED = os.environ.get("GK_STATIC_GUARD", "0") == "1"
_GK_STATIC_GUARD_SPEED = float(os.environ.get("GK_STATIC_GUARD_SPEED", "0.5"))
_GK_STATIC_GUARD_HISTORY = int(os.environ.get("GK_STATIC_GUARD_HISTORY", "5"))
_GK_STATIC_GUARD_MODE = os.environ.get("GK_STATIC_GUARD_MODE", "masked_policy")
_GK_STATIC_GUARD_ACTION = os.environ.get("GK_STATIC_GUARD_ACTION", "stand")
_GK_STATIC_GUARD_ACTION_SCALE = float(os.environ.get("GK_STATIC_GUARD_ACTION_SCALE", "0.35"))
_GK_STATIC_GUARD_BALL_LOCAL = torch.tensor(
    [float(v) for v in os.environ.get("GK_STATIC_GUARD_BALL_LOCAL", "5.0,0.0,0.5").split(",")],
    dtype=torch.float32,
)


def _home_joint_value(joint_name: str) -> float:
    for pattern, value in HOME_KEYFRAME.joint_pos.items():
        if pattern == joint_name:
            return float(value)
        if pattern.startswith(".*") and joint_name.endswith(pattern[2:]):
            return float(value)
    return 0.0


_GK_STAND_GUARD_ACTION = (
    torch.tensor(
        [_home_joint_value(name) for name in _GK_JOINT_NAMES],
        dtype=torch.float32,
    )
    - _GK_DEFAULT_JOINT_POS
) / 0.25

# ---------------------------------------------------------------------------
# Observation computation  (CUSTOMIZE: match your training observation space)
# ---------------------------------------------------------------------------

def _tensor(values: Any, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


class ShooterObsComputer:
    """Reconstruct the 160-D Stage-II shooter actor observation from raw state."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        device: str,
        shooter_motion_index: int | None = None,
        aim_mode: str = "center",
    ):
        self.env = env
        self.device = torch.device(device)
        self.command = env.command_manager.get_term("motion")
        self.default_joint_pos = (
            env.scene["robot"].data.default_joint_pos[0].detach().clone().to(self.device)
        )
        action_term = env.action_manager.get_term("joint_pos")
        self.action_scale = action_term._scale[0].detach().clone().to(self.device)
        self.motion_count = int(self.command.motion.num_files)
        self.motion_index_override = shooter_motion_index
        self.aim_mode = aim_mode
        self._reset_count = 0
        self._motion_idx = 0
        self._step = 0
        self._target_locked = False
        self._episode_target_world: torch.Tensor | None = None
        self._action_offset_correction = torch.zeros(1, 29, device=self.device)
        self.reset()

    def reset(self) -> None:
        if self.motion_index_override is None:
            self._motion_idx = self._reset_count % self.motion_count
            self._reset_count += 1
        else:
            self._motion_idx = int(self.motion_index_override) % self.motion_count
        self._step = 0
        self._target_locked = False
        self._episode_target_world = None
        self._action_offset_correction = self._compute_action_offset_correction()

    def _compute_action_offset_correction(self) -> torch.Tensor:
        motion_idx = torch.tensor(self._motion_idx, dtype=torch.long, device=self.device)
        initial_joint_pos = self.command.motion.joint_pos[motion_idx, 0]
        correction = (initial_joint_pos - self.default_joint_pos) / self.action_scale
        return correction.unsqueeze(0)

    def adapt_action(self, action: torch.Tensor) -> torch.Tensor:
        return action + self._action_offset_correction

    def _reference_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        motion = self.command.motion
        motion_idx = torch.tensor(self._motion_idx, dtype=torch.long, device=self.device)
        motion_len = int(motion.file_lengths[motion_idx].item())
        step = min(self._step, max(0, motion_len - 1))
        step_idx = torch.tensor(step, dtype=torch.long, device=self.device)

        joint_pos_ref = motion.joint_pos[motion_idx, step_idx]
        joint_vel_ref = motion.joint_vel[motion_idx, step_idx]
        command = torch.cat([joint_pos_ref, joint_vel_ref], dim=-1)
        motion_ref_ang_vel = motion.body_ang_vel_w[
            motion_idx, step_idx, self.command.motion_anchor_body_index
        ]
        return command, motion_ref_ang_vel

    def _goalkeeper_block_interval(self, raw_state: dict) -> tuple[float, float]:
        goalkeeper = raw_state.get("goalkeeper")
        if goalkeeper is None:
            return (-0.35, 0.35)

        gk_y = float(goalkeeper["root_pos"][1])
        gk_z = float(goalkeeper["root_pos"][2])
        root_quat = _tensor(goalkeeper["root_quat"], self.device)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)
        upright_score = float((-projected_gravity[2]).clamp(-1.0, 1.0).item())

        if gk_z < 0.55 or upright_score < 0.45:
            half_width = 0.95
        else:
            half_width = 0.55
        return (gk_y - half_width, gk_y + half_width)

    def _adaptive_goal_target_world(self, raw_state: dict) -> torch.Tensor:
        block_min, block_max = self._goalkeeper_block_interval(raw_state)
        candidates = (-1.15, -0.9, -0.7, 0.7, 0.9, 1.15)

        def candidate_score(y: float) -> float:
            if block_min <= y <= block_max:
                clearance = -min(y - block_min, block_max - y)
            elif y < block_min:
                clearance = block_min - y
            else:
                clearance = y - block_max
            return clearance + 0.05 * abs(y)

        target_y = max(candidates, key=candidate_score)
        return torch.tensor([-0.5, target_y, 0.11], dtype=torch.float32, device=self.device)

    def _goal_target_world(self, raw_state: dict) -> torch.Tensor:
        target_y = 0.0
        if self.aim_mode == "open" and "goalkeeper" in raw_state:
            gk_y = float(raw_state["goalkeeper"]["root_pos"][1])
            target_y = -0.75 if gk_y > 0.0 else 0.75
        elif self.aim_mode == "adaptive":
            ball_vel = raw_state.get("ball", {}).get("vel", [0.0, 0.0, 0.0])
            ball_speed = float(torch.linalg.norm(_tensor(ball_vel, self.device)).item())
            if not self._target_locked:
                self._episode_target_world = self._adaptive_goal_target_world(raw_state)
                if ball_speed > 0.5 or self._step >= 50:
                    self._target_locked = True
            if self._episode_target_world is not None:
                return self._episode_target_world
        elif self.aim_mode not in ("center", "open"):
            raise ValueError(f"Unsupported aim_mode: {self.aim_mode}")
        return torch.tensor([-0.5, target_y, 0.11], dtype=torch.float32, device=self.device)

    def __call__(self, raw_state: dict) -> torch.Tensor:
        s = raw_state["shooter"]
        ball = raw_state["ball"]

        root_quat = _tensor(s["root_quat"], self.device)
        root_ang_vel = _tensor(s["root_ang_vel"], self.device)
        joint_pos = _tensor(s["joint_pos"], self.device)
        joint_vel = _tensor(s["joint_vel"], self.device)
        ball_pos = _tensor(ball["pos"], self.device)
        root_pos = _tensor(s["root_pos"], self.device)
        last_action = _tensor(s["last_action"], self.device)

        command, motion_ref_ang_vel = self._reference_terms()
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)
        base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel)
        joint_pos_rel = joint_pos - self.default_joint_pos
        ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)
        goal_pos_local = quat_apply(
            quat_inv(root_quat), self._goal_target_world(raw_state) - root_pos
        )

        obs = torch.cat([
            command,              # 58
            projected_gravity,    # 3
            motion_ref_ang_vel,   # 3
            base_ang_vel,         # 3
            joint_pos_rel,        # 29
            joint_vel,            # 29
            last_action,          # 29
            ball_pos_local,       # 3
            goal_pos_local,       # 3
        ])
        self._step += 1
        if obs.numel() != 160:
            raise RuntimeError(f"Shooter obs must be 160-D, got {obs.numel()}")
        return obs.unsqueeze(0)


def compute_goalkeeper_obs(raw_state: dict) -> torch.Tensor:
    """Compute goalkeeper observation tensor from raw state (single frame).

    Default: matches ``eval_goalkeeper_cfg`` per-frame terms (96-D).
    Replace with your own obs terms, scaling, and concatenation order.
    """
    s = raw_state["goalkeeper"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"])
    root_ang_vel = torch.tensor(s["root_ang_vel"])
    joint_pos = torch.tensor(s["joint_pos"])
    joint_vel = torch.tensor(s["joint_vel"])
    ball_pos = torch.tensor(ball["pos"])
    root_pos = torch.tensor(s["root_pos"])
    last_action = torch.tensor(s["last_action"])

    # Projected gravity
    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)

    # Angular velocity with GK scaling (×0.25, matching GK PD gain ratio)
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel) * 0.25

    # Joint positions relative to GK default, GK-specific scaling
    joint_pos_rel = (joint_pos - _GK_DEFAULT_JOINT_POS) * 1.0

    # Joint velocities with GK scaling (×0.05)
    joint_vel_scaled = joint_vel * 0.05

    # Ball position in robot pelvis frame
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        ball_pos_local,         # 3
        base_ang_vel,           # 3
        projected_gravity,      # 3
        joint_pos_rel,          # 29
        joint_vel_scaled,       # 29
        last_action,            # 29
    ])
    return obs.unsqueeze(0)  # (1, 96)


def stack_goalkeeper_history_term_major(history: deque[torch.Tensor]) -> torch.Tensor:
    """Stack GK frame history in the same term-major order as mjlab observations.

    ``GoalkeeperActorCritic`` expects mjlab's history layout as input and then
    transposes it internally. Each frame here is 96-D frame-major, so the API
    server must explicitly rebuild the term-major 960-D layout.
    """
    term_sizes = (3, 3, 3, 29, 29, 29)
    frames = list(history)
    chunks = []
    offset = 0
    for size in term_sizes:
        chunks.extend(frame[:, offset : offset + size] for frame in frames)
        offset += size
    return torch.cat(chunks, dim=-1)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ActResponse(BaseModel):
    action: list[list[float]]  # shape: [1, act_dim]


# ---------------------------------------------------------------------------
# Policy loading  (for model architecture; obs are computed server-side)
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: str, task_id: str, device: str) -> Any:
    """Build env from task config, load checkpoint, return inference policy."""
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()

    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)

    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    history_len = env_cfg.observations["actor"].history_length
    print(f"[INFO] Task: {task_id}")
    print(f"[INFO] Actor obs  ({len(actor_terms)} terms × {history_len} history): {actor_terms}")
    print(f"[INFO] Action dim: {env.num_actions}")

    if task_id == "Eval-Goalkeeper":
        from src.tasks.soccer.config.g1.rl_cfg import (
            GoalkeeperRunner,
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )
        loaded = torch.load(checkpoint_path, map_location=device)
        agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
        runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)

        if "model_state_dict" in loaded and hasattr(runner.alg.actor, "history_encoder"):
            print("[INFO] Detected HIMPPO ActorCritic checkpoint — loading directly.")
            actor_state = {
                k: v
                for k, v in loaded["model_state_dict"].items()
                if not k.startswith("critic.")
            }
            runner.alg.actor.load_state_dict(actor_state, strict=False)
        else:
            runner.load(checkpoint_path, load_cfg={"actor": True})
    else:
        from src.tasks.soccer.config.g1.rl_cfg import (
            SoccerRecurrentRunner,
            unitree_g1_soccer_recurrent_runner_cfg,
        )
        agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
        runner = SoccerRecurrentRunner(
            env, asdict(agent_cfg), log_dir=None, device=device,
        )
        runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Policy loaded from: {checkpoint_path}")
    return policy, env


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    checkpoint_path: str,
    task_id: str,
    device: str,
    shooter_motion_index: int | None = None,
    aim_mode: str = "center",
) -> FastAPI:
    """Build the FastAPI app with a loaded policy and obs computer."""

    policy, env = _load_policy(checkpoint_path, task_id, device)
    policy_device = torch.device(device)
    env_base = env.unwrapped
    is_gk = task_id == "Eval-Goalkeeper"
    history_len = 10 if is_gk else 1
    shooter_obs_computer = None
    if not is_gk:
        shooter_obs_computer = ShooterObsComputer(
            env_base,
            device,
            shooter_motion_index=shooter_motion_index,
            aim_mode=aim_mode,
        )

    # History buffer for goalkeeper's multi-frame observation stack.
    history: deque[torch.Tensor] = deque(maxlen=history_len)
    ball_speed_history: deque[float] = deque(maxlen=max(1, _GK_STATIC_GUARD_HISTORY))
    gk_guard_released = not (is_gk and _GK_STATIC_GUARD_ENABLED)
    gk_guard_action = torch.zeros(1, env.num_actions, device=policy_device)
    if is_gk and _GK_STATIC_GUARD_ACTION == "stand":
        gk_guard_action = _GK_STAND_GUARD_ACTION.to(policy_device).unsqueeze(0)
    if is_gk and _GK_STATIC_GUARD_ENABLED:
        if _GK_STATIC_GUARD_BALL_LOCAL.numel() != 3:
            raise ValueError(
                "GK_STATIC_GUARD_BALL_LOCAL must contain exactly 3 comma-separated values"
            )
        if _GK_STATIC_GUARD_MODE not in {"hold_action", "masked_policy"}:
            raise ValueError(
                "GK_STATIC_GUARD_MODE must be either 'hold_action' or 'masked_policy'"
            )
        print(
            "[INFO] GK static-ball guard enabled: "
            f"speed>{_GK_STATIC_GUARD_SPEED:.3f} m/s over "
            f"{_GK_STATIC_GUARD_HISTORY} frame(s) releases policy; "
            f"mode={_GK_STATIC_GUARD_MODE}; "
            f"hold action={_GK_STATIC_GUARD_ACTION}; "
            f"action scale={_GK_STATIC_GUARD_ACTION_SCALE:.3f}."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[INFO] Server ready — {task_id} policy on {device}")
        yield
        env.close()
        print("[INFO] Server shutting down.")

    app = FastAPI(title=f"CS2810 Phase 2 — {task_id}", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/act", response_model=ActResponse)
    async def act(req: dict):
        nonlocal gk_guard_released
        # Compute per-frame observation from raw state.
        raw_state = req
        gk_guard_active = False
        if is_gk and _GK_STATIC_GUARD_ENABLED and not gk_guard_released:
            ball_vel = torch.tensor(raw_state["ball"]["vel"], dtype=torch.float32)
            ball_speed_history.append(float(torch.linalg.norm(ball_vel).item()))
            if max(ball_speed_history) > _GK_STATIC_GUARD_SPEED:
                gk_guard_released = True
                print(
                    "[INFO] GK static-ball guard released: "
                    f"recent speeds={list(round(v, 3) for v in ball_speed_history)}"
                )
            else:
                gk_guard_active = True

        if is_gk:
            frame = compute_goalkeeper_obs(raw_state)
            if gk_guard_active and _GK_STATIC_GUARD_MODE == "masked_policy":
                frame = frame.clone()
                frame[:, :3] = _GK_STATIC_GUARD_BALL_LOCAL.to(frame.device)
        else:
            assert shooter_obs_computer is not None
            frame = shooter_obs_computer(raw_state)

        # Initialize history buffer on first frame after reset.
        if len(history) == 0:
            for _ in range(history_len):
                history.append(frame.clone())

        history.append(frame)

        # Build stacked observation. GK policies expect mjlab's term-major
        # history layout; shooter history length is one frame.
        if is_gk:
            stacked = stack_goalkeeper_history_term_major(history).to(policy_device)
        else:
            stacked = torch.cat(list(history), dim=-1).to(policy_device)

        if gk_guard_active and _GK_STATIC_GUARD_MODE == "hold_action":
            return ActResponse(action=gk_guard_action.detach().cpu().tolist())

        with torch.inference_mode():
            action = policy({"actor": stacked})
            if not is_gk:
                assert shooter_obs_computer is not None
                action = shooter_obs_computer.adapt_action(action)

        if action.ndim != 2 or action.shape[-1] != env.num_actions:
            raise RuntimeError(
                f"Policy returned action shape {tuple(action.shape)}, "
                f"expected (1, {env.num_actions})"
            )

        if gk_guard_active and _GK_STATIC_GUARD_MODE == "masked_policy":
            action = action * _GK_STATIC_GUARD_ACTION_SCALE

        return ActResponse(action=action.detach().cpu().tolist())

    @app.post("/reset")
    async def reset():
        nonlocal gk_guard_released
        reset_fn = getattr(policy, "reset", None)
        if callable(reset_fn):
            reset_fn()
        if shooter_obs_computer is not None:
            shooter_obs_computer.reset()
        with torch.inference_mode():
            env.reset()
        history.clear()
        ball_speed_history.clear()
        gk_guard_released = not (is_gk and _GK_STATIC_GUARD_ENABLED)
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    checkpoint: str
    """Path to the policy checkpoint (.pt)."""
    port: int = 8000
    """Port to listen on."""
    task: str = "shooter"
    """Task type: 'shooter' or 'goalkeeper'."""
    host: str = "0.0.0.0"
    """Host to bind to."""
    device: str | None = None
    """Torch device (auto-detected if omitted)."""
    shooter_motion_index: int | None = None
    """Optional fixed soccer-standard motion index for shooter servers."""
    aim_mode: str = "center"
    """Shooter target selection: 'center', 'open', or 'adaptive'."""


def main():
    import src.tasks  # noqa: F401  — register eval tasks

    args = tyro.cli(ServerConfig, prog="api_server")

    task_id = "Eval-Shooter" if args.task == "shooter" else "Eval-Goalkeeper"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    app = create_app(
        args.checkpoint,
        task_id,
        device,
        shooter_motion_index=args.shooter_motion_index,
        aim_mode=args.aim_mode,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
