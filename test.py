from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from environment_single import EnvironmentSingle
from ray.rllib.algorithms.callbacks import DefaultCallbacks


import numpy as np
import os
import ray

# start_pt = np.array([4,1,4])
# end_pt = np.array([4,10,4])
start_pt = np.array([0,0,0])
end_pt = np.array([5,5,0])

# Choose what trained model to use based on train_ID
train_ID = "Test18_3D_obs"

checkpoint_dir = os.path.join('C:\\Users\\MDO-Disco\\Documents\\Thesis\\RLlib\\Checkpoints\\',train_ID)

trained_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint")

def env_creator(env_config):
    return EnvironmentSingle(env_config)

register_env("SinglePipe", env_creator)

env_config = {
    "start_pt":start_pt,
    "end_pt":end_pt,
}

config = {
    "env": "SinglePipe",
    "env_config": env_config,
    "num_gpus":1,
    "num_workers": 1,
    "num_envs_per_worker":5,
    "framework": "torch",
    "logger_config":{
        "config":{
            "log_level": "INFO",  # Set the log level
            "log_format": {
                "timesteps_total": "%6.3e",
                "episode_reward_mean": "%8.3f",

            }
        }           
    }
}
# Test one episode
print("TESTING NOW.......")

test_config = config
test_config["explore"]= False

agent = PPO(config=test_config)
agent.restore(trained_checkpoint_path)

env = EnvironmentSingle(env_config)

 # run until episode ends
episode_reward = 0
terminated = False
obs, info = env.reset()
print(obs)
while not terminated:
    action = agent.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    print("agent moved")
    print("current position",env.agent.get_position())

print(env.path)
env.render()

