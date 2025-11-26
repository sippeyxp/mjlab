from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pupper_v3_flat_env_cfg,
  pupper_v3_rough_env_cfg,
)
from .rl_cfg import pupper_v3_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pupper-v3",
  env_cfg=pupper_v3_rough_env_cfg(),
  play_env_cfg=pupper_v3_rough_env_cfg(play=True),
  rl_cfg=pupper_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pupper-v3",
  env_cfg=pupper_v3_flat_env_cfg(),
  play_env_cfg=pupper_v3_flat_env_cfg(play=True),
  rl_cfg=pupper_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
