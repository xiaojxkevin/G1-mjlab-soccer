"""Register evaluation tasks for naive shooter and goalkeeper.

These tasks use observation spaces compatible with Stage II training
so trained checkpoints load directly.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.assets.robots.unitree_g1.g1_constants import FULL_COLLISION, HOME_KEYFRAME
from src.tasks.soccer.config.eval.eval_goalkeeper_cfg import eval_goalkeeper_env_cfg
from src.tasks.soccer.config.eval.eval_shooter_cfg import eval_shooter_env_cfg
from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS


def _yaw_to_quat(yaw: float):
  half = yaw / 2.0
  return (math.cos(half), 0.0, 0.0, math.sin(half))


def _g1_eval_shooter_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Eval shooter with G1 robot at penalty spot + training-compatible obs."""
  cfg = eval_shooter_env_cfg(play=play)

  # Robot at shooter position.
  robot_cfg = get_g1_robot_cfg()
  robot_cfg.init_state = replace(HOME_KEYFRAME, pos=tuple(SETTINGS.scene.shooter_pos),
                                  rot=_yaw_to_quat(0.0))
  robot_cfg.collisions = (FULL_COLLISION,)
  cfg.scene.entities["robot"] = robot_cfg

  # G1 setup.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.viewer.body_name = "torso_link"
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                          entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="ground", entity="ground"),
    fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground,)

  # Inject motion dir for command.
  from src.tasks.soccer.mdp.commands import MultiMotionSoccerCommandCfg
  motion_cmd = cfg.commands["motion"]
  if isinstance(motion_cmd, MultiMotionSoccerCommandCfg):
    motion_cmd.motion_dir = "src/assets/soccer/motions"

  return cfg


register_mjlab_task(
  task_id="Eval-Naive-Shooter",
  env_cfg=_g1_eval_shooter_cfg(play=False),
  play_env_cfg=_g1_eval_shooter_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Eval-Naive-Goalkeeper",
  env_cfg=eval_goalkeeper_env_cfg(play=False),
  play_env_cfg=eval_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)
