"""Termination functions for the soccer task.

Matches HumanoidSoccer reference implementations exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Basic terminations
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
# Motion-reference terminations (matching HumanoidSoccer)
# ---------------------------------------------------------------------------


def _get_motion_cmd(env: ManagerBasedRlEnv, command_name: str):
  """Resolve motion command: command_manager (training) or env attr (eval)."""
  cmd = env.command_manager.get_term(command_name)
  if cmd is not None:
    return cmd
  return getattr(env, command_name, None)


def bad_anchor_pos_z(
  env: ManagerBasedRlEnv,
  threshold: float = 0.25,
  command_name: str = "motion",
) -> torch.Tensor:
  """Terminate when anchor Z deviates from reference (matching paper)."""
  cmd = _get_motion_cmd(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return torch.abs(cmd.anchor_pos_w[:, -1] - cmd.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.8,
  command_name: str = "motion",
) -> torch.Tensor:
  """Terminate when anchor orientation deviates from reference.

  Uses projected-gravity comparison (matching HumanoidSoccer reference),
  NOT quaternion subtraction. This is yaw-invariant: only pitch/roll deviation
  triggers termination.
  """
  cmd = _get_motion_cmd(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  # World gravity direction (MuJoCo: (0,0,-1) in z-up convention).
  grav = torch.tensor([[0.0, 0.0, -1.0]], device=env.device).expand(env.num_envs, -1)

  motion_proj_grav = quat_apply_inverse(cmd.anchor_quat_w, grav)
  robot_proj_grav = quat_apply_inverse(cmd.robot_anchor_quat_w, grav)
  return (motion_proj_grav[:, 2] - robot_proj_grav[:, 2]).abs() > threshold


def bad_ee_body_pos_z(
  env: ManagerBasedRlEnv,
  threshold: float = 0.25,
  command_name: str = "motion",
  body_names: tuple[str, ...] = (
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
  ),
) -> torch.Tensor:
  """Terminate when any end-effector Z deviates from reference.

  Matches reference bad_motion_body_pos_z_only exactly.
  body_pos_relative_w[..., 2] = body_pos_w world z (yaw doesn't change z).
  robot_body_pos_w[..., 2] = robot world z (mjlab convention).
  Both are in world frame, so direct comparison is correct.
  """
  cmd = _get_motion_cmd(env, command_name)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  body_indexes = [
    i for i, name in enumerate(cmd.cfg.body_names)
    if name in body_names
  ]
  if not body_indexes:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  ref_z = cmd.body_pos_relative_w[:, body_indexes, 2]
  robot_z = cmd.robot_body_pos_w[:, body_indexes, 2]
  error = torch.abs(ref_z - robot_z)
  return torch.any(error > threshold, dim=-1)
