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


