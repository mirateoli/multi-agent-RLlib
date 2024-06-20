from ray.tune.registry import register_env

from environment import Environment
# from inputs import start_pts, end_pts, num_pipes

import os

import random
import numpy as np

from ray.rllib.algorithms.dqn import DQNConfig

from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining

import argparse

#uncomment for training
num_pipes = 3
start_pts = np.array([(4,1,4),(4,1,4),(4,1,4)])
end_pts = np.array([(10,11,7),(10,8,4),(8,11,3)])

alg_name = "DQN"
env_name = "MultiPipe"

def env_creator(env_config):
    return Environment(env_config)

register_env(env_name, env_creator)

env_config = {
    "train": True,
    "num_pipes": num_pipes,
    "start_pts":start_pts,
    "end_pts":end_pts,
}


config = (
    DQNConfig()
    .environment(env=env_name, env_config = env_config)
    .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
    .training(
        train_batch_size=200,
        hiddens=[],
        dueling=False,
    )
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .debugging(
        log_level="DEBUG" 
    )  # TODO: change to ERROR to match pistonball example
    .framework(framework="torch")
    .exploration(
        exploration_config={
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.1,
            "final_epsilon": 0.0,
            "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
        }
    )
)

tune.run(
    alg_name,
    name="DQN",
    stop={"timesteps_total": 10000000 if not os.environ.get("CI") else 50000},
    checkpoint_freq=10,
    config=config.to_dict(),
)