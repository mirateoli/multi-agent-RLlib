
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from environment_multi_combo import Environment
from inputs import start_pts, end_pts, num_pipes

import numpy as np
import os

from spaces import agent_action_space, agent_obs_space
import random

import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining


import argparse

def env_creator(env_config):
    return Environment(env_config)

register_env("MultiPipe", env_creator)

env_config = {
    "train": True,
    "num_pipes": num_pipes,
    "start_pts":start_pts,
    "end_pts":end_pts,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)
args, _ = parser.parse_known_args()

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config

hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "num_sgd_iter": lambda: random.randint(1, 30),
    "sgd_minibatch_size": lambda: random.randint(128, 16384),
    "train_batch_size": lambda: random.randint(2000, 160000),
}

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations=hyperparam_mutations,
    custom_explore_fn=explore,
)

# Stop when we've either reached 100 training iterations or reward=300
stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 15}

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        scheduler=pbt,
        num_samples=1 if args.smoke_test else 2,
    ),
    param_space={
        "env": "MultiPipe",
        "env_config": env_config,
        "kl_coeff": 1.0,
        "num_workers": 4,
        "num_cpus": 0,  # number of CPUs to use per trial
        "num_gpus": 1,  # number of GPUs to use per trial
        # These params are tuned from a fixed starting value.
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1e-4,
        # These params start off randomly drawn from a set.
        "num_sgd_iter": tune.choice([10, 20, 30]),
        "sgd_minibatch_size": tune.choice([128, 512, 2048]),
        "train_batch_size": tune.choice([10000, 20000, 40000]),
    },
    run_config=train.RunConfig(stop=stopping_criteria),
)
results = tuner.fit()

import pprint

best_result = results.get_best_result()

print("Best performing trial's final set of hyperparameters:\n")
pprint.pprint(
    {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
)

print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    "episode_len_mean",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

# from ray.rllib.algorithms.algorithm import Algorithm

# loaded_ppo = Algorithm.from_checkpoint(best_result.checkpoint)
# agent = loaded_ppo.get_policy()

# env = Environment(env_config)

#  # run until episode ends
# episode_reward = 0
# terminated = False
# obs, info = env.reset()
# print(obs)
# while not terminated or truncated:
#     action = {}
#     for agent_id, agent_obs in obs.items():
#         action[agent_id] = agent.compute_single_action(agent_obs)
#     obs, reward, terminated, truncated, info = env.step(action)
#     terminated = terminated['__all__']
#     truncated = truncated['__all__']
#     episode_reward += sum(reward.values())
#     # print("agent moved")
#     # print("current position",env.agents.get_position())

# print(env.paths )
# env.render()