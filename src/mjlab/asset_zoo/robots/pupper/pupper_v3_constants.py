"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
  reflected_inertia,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PUPPER_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "pupper" / "xmls" / "pupper_v3_complete.xml"
)
assert PUPPER_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PUPPER_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PUPPER_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Steadywin GIM4305 actuators

# 4005 brushless motor
# 10:1 planetary gearbox
# ~ 3.5 Nm peak torque
# ~ 1.0 Nm continuous torque with air cooling
# ~ 30 rad/s max speed
ROTOR_INERTIA = 0.00001

# Gearbox.
GEAR_RATIO = 10
KNEE_GEAR_RATIO = 10

ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, GEAR_RATIO),
  velocity_limit=30.1,
  effort_limit=3.5,
)

NATURAL_FREQ = 15 * 2.0 * 3.1415926535  # 15Hz
DAMPING_RATIO = 2.0

STIFFNESS = ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ACTUATOR.reflected_inertia * NATURAL_FREQ

PUPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  joint_names_expr=(r"^leg_(front|back)_(l|r)_(1|2|3)$",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=ACTUATOR.effort_limit,
  armature=ACTUATOR.reflected_inertia,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.05),
  joint_pos={
    "leg_front_r_1": -0.0247,
    "leg_front_r_2": 0.0178,
    "leg_front_r_3": 0.717,

    "leg_front_l_1": 0.0247,
    "leg_front_l_2": -0.0178,
    "leg_front_l_3": -0.717,

    "leg_back_r_1": 0.00434,
    "leg_back_r_2": 0.0151,
    "leg_back_r_3": 0.712,

    "leg_back_l_1": -0.00434,
    "leg_back_l_2": -0.0151,
    "leg_back_l_3": -0.712,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.13),
  joint_pos={
    "leg_front_r_1": 0.0,
    "leg_front_r_2": 0.0,
    "leg_front_r_3": 0.0,

    "leg_front_l_1": 0.0,
    "leg_front_l_2": 0.0,
    "leg_front_l_3": 0.0,

    "leg_back_r_1": 0.0,
    "leg_back_r_2": 0.0,
    "leg_back_r_3": 0.0,

    "leg_back_l_1": 0.0,
    "leg_back_l_2": 0.0,
    "leg_back_l_3": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^leg_(front|back)_(l|r)_3_collision$": 3, ".*_collision": 1},
  priority={r"^leg_(front|back)_(l|r)_3_collision$": 1},
  friction={r"^leg_(front|back)_(l|r)_3_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^leg_(front|back)_(l|r)_3_collision$": 3, ".*_collision": 1},
  priority={r"^leg_(front|back)_(l|r)_3_collision$": 1},
  friction={r"^leg_(front|back)_(l|r)_3_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^leg_(front|back)_(l|r)_3_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

PUPPER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PUPPER_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_pupper_v3_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PUPPER_ARTICULATION,
  )


PUPPER_ACTION_SCALE: dict[str, float] = {}
for a in PUPPER_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  assert e is not None
  for n in names:
    PUPPER_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pupper_v3_robot_cfg())

  viewer.launch(robot.spec.compile())
