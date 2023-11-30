from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from environment_multi_combo import Environment

from inputs import *


def env_creator(env_config):
    return Environment(env_config)

register_env("MultiPipe", env_creator)


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="MultiPipe", env_config={"num_pipes":num_pipes, "start_pts":start_pts, "end_pts":end_pts}, disable_env_checking=True)
    .multi_agent(
            policies=set([0,1]),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
    
    )
    .build()
    
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")