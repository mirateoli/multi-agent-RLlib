from ray.tune.registry import register_env
from environment import Environment
import random
import numpy as np
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
import argparse

# Uncomment for training
num_pipes = 3
start_pts = np.array([(4, 1, 4), (4, 1, 4), (4, 1, 4)])
end_pts = np.array([(10, 11, 7), (10, 8, 4), (8, 11, 3)])

def env_creator(env_config):
    return Environment(env_config)

register_env("MultiPipe", env_creator)

env_config = {
    "train": True,
    "num_pipes": num_pipes,
    "start_pts": start_pts,
    "end_pts": end_pts,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)
args, _ = parser.parse_known_args()

# Postprocess the perturbed config to ensure it's still valid
def explore(config):
    if config["train_batch_size"] < config["batch_size"] * 2:
        config["train_batch_size"] = config["batch_size"] * 2
    if config["num_steps_sampled_before_learning_starts"] < 1000:
        config["num_steps_sampled_before_learning_starts"] = 1000
    return config

hyperparam_mutations = {
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "train_batch_size": lambda: random.randint(2000, 160000),
    "exploration_config": {
        "epsilon_timesteps": lambda: random.randint(10000, 50000),
        "final_epsilon": lambda: random.uniform(0.01, 0.1),
    },
    "replay_buffer_config": {
        "capacity": lambda: random.randint(50000, 1000000),
    },
    "num_steps_sampled_before_learning_starts": lambda: random.randint(1000, 10000),
    "target_network_update_freq": lambda: random.randint(100, 5000),
}

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    hyperparam_mutations=hyperparam_mutations,
    custom_explore_fn=explore,
)

# Stop when we've either reached 200 training iterations or reward=100
stopping_criteria = {"training_iteration": 200, "episode_reward_mean": 100}

tuner = tune.Tuner(
    "DQN",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        scheduler=pbt,
        num_samples=1 if args.smoke_test else 2,
    ),
    param_space={
        "env": "MultiPipe",
        "env_config": env_config,
        "num_workers": 4,
        "num_cpus": 0,  # number of CPUs to use per trial
        "num_gpus": 1,  # number of GPUs to use per trial
        "lr": 1e-4,
        "gamma": 0.9,
        "train_batch_size": tune.choice([10000, 20000, 40000, 60000]),
        "exploration_config": {
            "epsilon_timesteps": tune.choice([10000, 20000, 30000]),
            "final_epsilon": tune.choice([0.01, 0.05, 0.1]),
        },
        "replay_buffer_config": {
            "capacity": tune.choice([50000, 100000, 500000]),
        },
        "num_steps_sampled_before_learning_starts": tune.choice([1000, 5000, 10000]),
        "target_network_update_freq": tune.choice([500, 1000, 5000]),
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

