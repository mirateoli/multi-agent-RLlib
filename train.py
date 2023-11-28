from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import pprint


from environment import Environment

def env_creator(env_config):
    return Environment(env_config)

register_env("my_env", env_creator)
trainer = ppo.PPO(env="my_env")


import argparse

from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v4

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    def env_creator(args):
        return Environment(args)

    env = env_creator({})
    register_env("pipeworld", env_creator)

    config = (
        PPOConfig()
        .environment("pipeworld")
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=2)
        .multi_agent(
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    config.environment(disable_env_checking=True)

    if args.as_test:
        # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 60000}

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space=config,
    ).fit()