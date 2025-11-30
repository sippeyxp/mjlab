"""Unitree Go1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  PUPPER_ACTION_SCALE,
  get_pupper_v3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def pupper_v3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Pupper v3 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_pupper_v3_robot_cfg()}

  foot_names = ("leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3")
  site_names = ("leg_front_r_3_foot_site", "leg_front_l_3_foot_site",
                "leg_back_r_3_foot_site", "leg_back_l_3_foot_site")
  geom_names = tuple(f"{name}_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r"leg_.*_collision",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PUPPER_ACTION_SCALE

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # tuned this number because walking a bit slow and dragging feet a bit
  cfg.rewards["pose"].params["std_standing"] = {
    r"leg_(front|back)_(l|r)_(1|2)": 0.05,
    r"leg_(front|back)_(l|r)_3": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r"leg_(front|back)_(l|r)_(1|2)": 0.3,
    r"leg_(front|back)_(l|r)_3": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"leg_(front|back)_(l|r)_(1|2)": 0.5,
    r"leg_(front|back)_(l|r)_3": 1.0,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["foot_clearance"].params["target_height"] = 0.03
  cfg.rewards["foot_clearance"].weight = -2.0
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.03
  cfg.rewards["foot_swing_height"].weight = -1.0

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0

  cfg.rewards["air_time"].weight = 0.2 # tune this
  cfg.rewards["air_time"].params["threshold_min"] = 0.05
  cfg.rewards["air_time"].params["threshold_max"] = 0.5

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  cfg.rewards["action_rate_l2"].weight = -0.01


  cfg.curriculum["command_vel"].params["velocity_stages"]= [
          {"step": 0, "lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-0.25, 0.25)},
          {"step": 2000 * 24, "lin_vel_x": (-0.6, 0.8), "ang_vel_z": (-0.7, 0.7)},
          {"step": 5000 * 24, "lin_vel_x": (-0.8, 1.2), "ang_vel_z": (-1.0, 1.0)},
        ]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def pupper_v3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Pupper v3 flat terrain velocity configuration."""
  cfg = pupper_v3_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  del cfg.curriculum["terrain_levels"]

  return cfg
