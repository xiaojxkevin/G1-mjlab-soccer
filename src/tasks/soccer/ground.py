"""Ground entity with soccer-specific contact params and skybox."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg

GROUND_XML: Path = SRC_PATH / "assets" / "soccer" / "ground.xml"
assert GROUND_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GROUND_XML))
  return spec


def get_ground_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
  )
