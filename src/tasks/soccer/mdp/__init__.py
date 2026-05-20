"""Soccer task MDP terms.

Self-contained MDP functions for observations, rewards, terminations, and
reset events.  Mirrors the mjlab.tasks.velocity.mdp pattern so that the
soccer task does not depend on the velocity task's MDP module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import (
  sample_uniform,
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


def last_action(
  env: ManagerBasedRlEnv, action_name: str | None = None
) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_ = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity_[:, 2]).abs() > limit_angle


# ---------------------------------------------------------------------------
# Reset events
# ---------------------------------------------------------------------------


def reset_root_state_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  # Pose.
  range_list = [
    pose_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  pose_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )

  # Fixed-base mocap entity path.
  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity "
        f"'{asset_cfg.name}'."
      )
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()
    positions = (
      root_states[:, 0:3]
      + pose_samples[:, 0:3]
      + env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)
    asset.write_mocap_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  # Floating-base entity path.
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  positions = (
    root_states[:, 0:3]
    + pose_samples[:, 0:3]
    + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  if velocity_range is None:
    velocity_range = {}
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


# ---------------------------------------------------------------------------
# Soccer-specific reset: random ball velocity toward goal
# ---------------------------------------------------------------------------


@dataclass
class GoalkeeperBallVelCfg:
  """Configuration for randomizing goalkeeper ball velocity.

  The ball is launched from the penalty spot toward the goal area.
  Velocity is randomized in speed, yaw direction, and pitch angle.

  Attributes:
    speed_min: Minimum ball speed (m/s).
    speed_max: Maximum ball speed (m/s).
    yaw_spread_deg: Maximum yaw spread angle (degrees) from straight-ahead.
      The ball direction is uniformly sampled within ±yaw_spread_deg
      relative to the +x (toward goal) direction.
    pitch_min_deg: Minimum launch elevation angle (degrees).
    pitch_max_deg: Maximum launch elevation angle (degrees).
  """

  speed_min: float = 2.0
  speed_max: float = 4.5
  yaw_spread_deg: float = 18.0
  pitch_min_deg: float = 3.0
  pitch_max_deg: float = 8.0


def reset_ball_with_goal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_cfg: GoalkeeperBallVelCfg,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
  """Reset ball position and apply a randomized velocity toward goal.

  Ball position is reset to its default (from init_state).  Velocity is
  sampled with random direction and speed within the configured ranges so
  every episode presents a different shot.

  Args:
      env: The environment.
      env_ids: Environment IDs to reset. If None, resets all.
      vel_cfg: Velocity randomization configuration.
      ball_cfg: Ball entity configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[ball_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  # Position: reset to default (at penalty spot) + env origins.
  positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]

  # Speed: uniform in [speed_min, speed_max].
  speeds = (
    vel_cfg.speed_min
    + torch.rand(len(env_ids), device=env.device)
    * (vel_cfg.speed_max - vel_cfg.speed_min)
  )

  # Yaw angle: uniform within ±yaw_spread_deg relative to +x (toward goal).
  spread_rad = math.radians(vel_cfg.yaw_spread_deg)
  yaws = (torch.rand(len(env_ids), device=env.device) * 2 - 1) * spread_rad

  # Pitch angle: uniform in [pitch_min_deg, pitch_max_deg].
  pitch_min_rad = math.radians(vel_cfg.pitch_min_deg)
  pitch_max_rad = math.radians(vel_cfg.pitch_max_deg)
  pitches = (
    pitch_min_rad
    + torch.rand(len(env_ids), device=env.device)
    * (pitch_max_rad - pitch_min_rad)
  )

  # Convert spherical to Cartesian: +x toward goal.
  vx = speeds * torch.cos(pitches) * torch.cos(yaws)
  vy = speeds * torch.cos(pitches) * torch.sin(yaws)
  vz = speeds * torch.sin(pitches)

  # Full root velocity: 6-DOF (lin_vel, ang_vel).  Ball has no initial spin.
  ang_vel = torch.zeros(len(env_ids), 3, device=env.device)
  velocities = torch.cat(
    [torch.stack([vx, vy, vz], dim=-1), ang_vel], dim=-1
  )

  # Quaternion unchanged from default.
  orientations = root_states[:, 3:7]

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
