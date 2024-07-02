from ray.tune.registry import register_env

from environment import Environment
# from inputs import start_pts, end_pts, num_pipes

import random
import numpy as np

import ray
from ray import train, tune

from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

#uncomment for training
num_pipes = 3
start_pts = np.array([(4,1,4),(4,1,4),(4,1,4)])
end_pts = np.array([(10,11,7),(10,8,4),(8,11,3)])

ray.init(num_gpus=1)

def env_creator(env_config):
    return Environment(env_config)

register_env("MultiPipe", env_creator)

env_config = {
    "train": True,
    "num_pipes": num_pipes,
    "start_pts":start_pts,
    "end_pts":end_pts,
}

# Stop when we've either reached 100 training iterations
stopping_criteria = {"training_iteration": 100}

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        num_samples=1
    ),
    param_space={
        "env": "MultiPipe",
        "env_config": env_config,
        "kl_coeff": 1.0,
        "num_workers": 2,
        "num_cpus": 1,  # number of CPUs to use per trial
        "num_gpus": 0.05,  # number of GPUs to use per trial
        "lambda": 0.95,
        "lr": tune.grid_search([1e-5, 1e-4]),
        "sgd_minibatch_size": tune.grid_search([64, 128, 256, 512]),
        "train_batch_size": tune.grid_search([4000, 8000, 16000, 32000]),
        # "num_sgd_iter": tune.choice([10, 20, 30]),
        # "gamma": tune.uniform(0.95, 0.999),
        # "lambda": tune.uniform(0.9, 1.0),
        # "clip_param": tune.uniform(0.1, 0.3),
        # "entropy_coeff": tune.loguniform(1e-4, 1e-2),
        # "vf_loss_coeff": tune.loguniform(0.5, 1.0),
        # "kl_target": tune.uniform(0.01, 0.05),
    },
    run_config=train.RunConfig(
        stop=stopping_criteria,
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency = 10
        )
    ),
)
results = tuner.fit()

# Retrieve the best checkpoint and hyperparameters
best_trial = results.get_best_result(metric="episode_reward_mean", mode="max")
best_config = best_trial.config
print("Best trial config: {}".format(best_config))

best_checkpoint = best_trial.checkpoint
print("Best checkpoint: {}".format(best_checkpoint))

