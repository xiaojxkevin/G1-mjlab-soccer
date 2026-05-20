"""Ball entity configuration for soccer tasks."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg
from src.tasks.soccer.config.soccer_settings import SETTINGS

BALL_XML: Path = SRC_PATH / "assets" / "soccer" / "ball.xml"
assert BALL_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(BALL_XML))
  s = SETTINGS.ball
  if s.air_drag.enabled:
    body = spec.worldbody.find_child("ball")
    geom = body.geoms[0]
    geom.fluid_ellipsoid = s.radius
    # Keep default fluid_coefs (blunt quadratic = 0.25) — already provides
    # drag comparable to Cd≈0.47 for a 0.1m radius sphere at STP.
  return spec


def get_ball_cfg(
  pos: tuple[float, float, float] | None = None,
) -> EntityCfg:
  """Get ball entity configuration with custom initial position.

  Args:
      pos: World position (x, y, z) of the ball at reset.
           Defaults to the ball pos from settings.yaml.

  Returns:
      EntityCfg configured for the ball.
  """
  if pos is None:
    pos = tuple(SETTINGS.scene.ball_pos)
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=pos),
    spec_fn=get_spec,
  )
