from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from environment_single import EnvironmentSingle

start_pt = [0,0]
end_pt = [3,3]

def env_creator(env_config):
    return EnvironmentSingle(env_config)

register_env("SinglePipe", env_creator)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="SinglePipe", env_config={"start_pt":start_pt, "end_pt":end_pt})
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")