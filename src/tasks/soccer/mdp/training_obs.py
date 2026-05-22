"""Observation functions depending on MultiMotionSoccerCommand.

These parallel mjlab's built-in tracking observations
(mjlab.tasks.tracking.mdp.observations) but cast to
MultiMotionSoccerCommand instead of the single-motion MotionCommand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_inv, subtract_frame_transforms

from .commands import MultiMotionSoccerCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor position in robot base frame (3D)."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w, command.robot_anchor_quat_w,
    command.anchor_pos_w, command.anchor_quat_w,
  )
  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor orientation in robot base frame (6D — first 2 cols of rot matrix)."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w, command.robot_anchor_quat_w,
    command.anchor_pos_w, command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor angular velocity (3D)."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  return command.anchor_ang_vel_w.view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """All robot body positions in robot anchor frame (privileged critic)."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """All robot body orientations as 6D representation (privileged critic)."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def constant_target_point_pos(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Ball position in robot pelvis frame (3D). Observation O^soc_t term."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  target_w = command.target_point_pos + env.scene.env_origins
  delta = target_w - command.robot_pelvis_pos_w
  return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def target_destination_pos_local(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Goal center in robot pelvis frame (3D). Observation O^soc_t term."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  dest_w = command.target_destination_pos + env.scene.env_origins
  delta = dest_w - command.robot_pelvis_pos_w
  return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)
