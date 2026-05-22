"""Shooter evaluation config — matches HumanoidSoccer paper observation space.

Observation order matches Stage II training so trained checkpoints load directly.

  o_t = (o^prop_t, o^ref_t, o^soc_t)

  Actor (160D, same term order as Unitree-G1-Shooter-Stage2):
    command (58) → projected_gravity (3) → motion_ref_ang_vel (3) →
    base_ang_vel (3) → joint_pos (29) → joint_vel (29) → actions (29) →
    target_point_pos (3) → target_destination_pos (3)

Scene: physical goal + ball at penalty spot + ground (robot added by G1 wrapper).

Eval design (matching paper §IV-B):
  - Goal and ball are FIXED (penalty kick scenario).
  - Motion command sets the G1 to the motion's frame-0 pose, then applies
    random position offset (±0.5m xy) and yaw rotation (±0.5 rad) on every reset.
  - The policy must track the motion reference while adjusting to kick the
    stationary ball at (-4, 0, 0.11).
  - 100 trials with different seeds → Success Rate = % of balls entering goal.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.soccer import mdp as soccer_mdp
from src.tasks.soccer.ball import get_ball_cfg
from src.tasks.soccer.goal import get_goal_cfg
from src.tasks.soccer.ground import get_ground_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.soccer_env_cfg import _add_soccer_scene_postproc
from src.tasks.soccer.mdp.commands import MultiMotionSoccerCommandCfg
from src.tasks.soccer.mdp.training_obs import (
  constant_target_point_pos,
  motion_anchor_ang_vel,
  target_destination_pos_local,
)
from src.tasks.soccer.mdp.training_obs import (
  motion_anchor_pos_b,
  motion_anchor_ori_b,
  robot_body_pos_b,
  robot_body_ori_b,
)

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

# Body names matching training configs.
_TRACKING_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)

_EVAL_XY_RANGE = 0.5   # ±m — lateral randomization
_EVAL_YAW_RANGE = 0.5  # ±rad — yaw randomization


def eval_shooter_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Shooter eval config — Stage II-compatible obs ordering.

  Two-phase robot placement:
  1. Command (uniform mode): places G1 at motion frame 0 + small xy/yaw noise.
  2. shift_root_pose event: applies global offset from settings to move the
     robot from motion origin to world position behind the penalty spot.
  Ball is fixed at penalty spot via fixed_ball_pos.
  """
  s = SETTINGS.scene
  ox, oy, oz = s.motion_origin_offset

  # -- Commands: motion places G1 near origin + random noise, then shifts ------
  # -- to world position via motion_origin_offset / motion_yaw_offset. ----------

  commands = {
    "motion": MultiMotionSoccerCommandCfg(
      motion_dir="",
      anchor_body_name="torso_link",
      body_names=_TRACKING_BODY_NAMES,
      entity_name="robot",
      ball_entity_name="ball",
      resampling_time_range=(1e9, 1e9),
      pose_range={
        "x": (-_EVAL_XY_RANGE, _EVAL_XY_RANGE),
        "y": (-_EVAL_XY_RANGE, _EVAL_XY_RANGE),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (-_EVAL_YAW_RANGE, _EVAL_YAW_RANGE),
      },
      velocity_range={},
      joint_position_range=(-0.0, 0.0),
      sampling_mode="uniform",
      fixed_ball_pos=tuple(s.ball_pos),
      motion_origin_offset=(ox, oy, oz),
      motion_yaw_offset=s.motion_yaw_offset,
      debug_vis=False,
    ),
  }

  # -- Observations (same order as Stage II) -----------------------------------

  actor_terms = {
    "command": ObservationTermCfg(
      func=soccer_mdp.generated_commands,
      params={"command_name": "motion"},
    ),
    "projected_gravity": ObservationTermCfg(func=soccer_mdp.projected_gravity),
    "motion_ref_ang_vel": ObservationTermCfg(
      func=motion_anchor_ang_vel,
      params={"command_name": "motion"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=soccer_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(func=soccer_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=soccer_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=soccer_mdp.last_action),
    "target_point_pos": ObservationTermCfg(
      func=constant_target_point_pos,
      params={"command_name": "motion"},
    ),
    "target_destination_pos": ObservationTermCfg(
      func=target_destination_pos_local,
      params={"command_name": "motion"},
    ),
  }

  critic_terms = {
    **actor_terms,
    "motion_anchor_pos_b": ObservationTermCfg(
      func=motion_anchor_pos_b,
      params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=motion_anchor_ori_b,
      params={"command_name": "motion"},
    ),
    "body_pos": ObservationTermCfg(
      func=robot_body_pos_b,
      params={"command_name": "motion"},
    ),
    "body_ori": ObservationTermCfg(
      func=robot_body_ori_b,
      params={"command_name": "motion"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=soccer_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  # -- Actions ----------------------------------------------------------------

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  # -- Events: joint offset reset only (G1/ball placement handled by command) -----

  events = {
    "reset_robot_joints": EventTermCfg(
      func=soccer_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  # -- Terminations (timeout + fell_over only) --------------------------------

  from mjlab.managers.termination_manager import TerminationTermCfg

  terminations = {
    "time_out": TerminationTermCfg(func=soccer_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=soccer_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  # -- Rewards (eval only) ----------------------------------------------------

  from mjlab.managers.reward_manager import RewardTermCfg

  rewards = {
    "is_terminated": RewardTermCfg(func=soccer_mdp.is_terminated, weight=-200.0),
  }

  # -- Assemble ---------------------------------------------------------------

  ep_len = int(1e9) if play else SETTINGS.episode_length_s

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={
        "ground": get_ground_cfg(),
        "ball": get_ball_cfg(pos=tuple(s.ball_pos)),
        "goal": get_goal_cfg(),
      },
      num_envs=1,
      spec_fn=_add_soccer_scene_postproc,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=6.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=48,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=ep_len,
  )
