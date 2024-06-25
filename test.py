from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from environment import Environment
# from inputs import start_pts, end_pts, num_pipes
import select_points_GUI as SPG
import design_spaces as DS

import numpy as np
import os
import time

from spaces import agent_action_space, agent_obs_space

# set start points and end points
num_pipes, key_pts = SPG.select_pts(DS.obstacles)

start_pts = key_pts[::2] # get even indices
end_pts = key_pts[1::2]   # get odd indices

# Choose what trained model to use based on train_ID
train_ID = "MultiAgent_branch_5"

checkpoint_dir = os.path.join('C:\\Users\\MDO-Disco\\Documents\\Thesis\\RLlib\\Checkpoints\\',train_ID)

# trained_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint")

# Obstacle avoidance with 90 deg bends (change input to have num_directions = 6)
# trained_checkpoint_path = "C:\\Users\\MDO-Disco\\ray_results\\PPO_2024-04-24_15-43-54\\PPO_MultiPipe_ff418_00000_0_num_sgd_iter=20,sgd_minibatch_size=2048,train_batch_size=40000_2024-04-24_15-44-00\\checkpoint_000012\\"

# Obstacle avoidance with 90 and 45 deg bends (change input to have num_directions = 18)
#trained_checkpoint_path = "C:\\Users\\MDO-Disco\\ray_results\\PPO_2024-06-07_14-57-19\\PPO_MultiPipe_c7098_00000_0_num_sgd_iter=30,sgd_minibatch_size=512,train_batch_size=40000_2024-06-07_14-57-24\\checkpoint_000026\\"

# FAIL Obstacle avoidance with 90 and 45 deg bends (increased pength penalty by 2x)
# trained_checkpoint_path = "C:\\Users\MDO-Disco\\ray_results\\PPO_2024-06-07_16-35-11\\PPO_MultiPipe_72e58_00001_1_num_sgd_iter=30,sgd_minibatch_size=2048,train_batch_size=60000_2024-06-07_16-35-16\\checkpoint_000010\\"

# Obstacle avoidance with 90 and 45 deg bends (increased iters from 100 to 200)
#trained_checkpoint_path = "C:\\Users\\MDO-Disco\\ray_results\PPO_2024-06-08_15-35-55\\PPO_MultiPipe_55e72_00000_0_num_sgd_iter=10,sgd_minibatch_size=512,train_batch_size=40000_2024-06-08_15-36-00\\checkpoint_000050\\"

# Obstacle avoidance with 90 and 45 deg bends (200 iters, obs reward -5)
# trained_checkpoint_path = "C:\\Users\MDO-Disco\\ray_results\\PPO_2024-06-10_16-57-28\\PPO_MultiPipe_0f3bd_00001_1_num_sgd_iter=10,sgd_minibatch_size=2048,train_batch_size=40000_2024-06-10_16-57-33\\checkpoint_000038\\"

# Obstacle avoidance with 90 and 45 deg bends (200 iters, obs reward -10)
# trained_checkpoint_path = "C:\\Users\\MDO-Disco\\ray_results\PPO_2024-06-12_12-37-01\\PPO_MultiPipe_024da_00000_0_num_sgd_iter=10,sgd_minibatch_size=512,train_batch_size=10000_2024-06-12_12-37-07\\checkpoint_000034\\"

# Branching and obstacle avoidance and 90 and 45 deg bends
trained_checkpoint_path = "C:\\Users\\MDO-Disco\\ray_results\PPO_2024-06-20_16-02-03\\PPO_MultiPipe_f9b86_00001_1_num_sgd_iter=20,sgd_minibatch_size=2048,train_batch_size=40000_2024-06-20_16-02-09\\checkpoint_000025"

def env_creator(env_config):
    return Environment(env_config)

register_env("MultiPipe", env_creator)

# set "train" to False if you don't want to test specified start and end points
env_config = {
    "train": False,
    "num_pipes": num_pipes,
    "start_pts":start_pts,
    "end_pts":end_pts,
}

config = {
    "env": "MultiPipe",
    "env_config": env_config,
    "train_batch_size":20000,
    "num_sgd_iter": 30,
    "sgd_minibatch_size": 2048,
    "lr": 0.0001,
    "lambda":  0.95,
    "clip_param": 0.2,
    "num_gpus":1,
    "num_workers": 1,
    "num_envs_per_worker":1,
    "framework": "torch",
    "observation_space": agent_obs_space,
    "action_space": agent_action_space,
    }                 


# Test one episode
print("TESTING NOW.......")



test_config = config
test_config["explore"]= False
test_config["entropy_coeff"]= 0

agent = PPO(config=test_config)
agent.restore(trained_checkpoint_path)

env = Environment(env_config)

 # run until episode ends
episode_reward = 0
terminated = False
obs, info = env.reset()
print(obs)
start_time = time.time()
while not terminated or truncated:
    action = {}
    for agent_id, agent_obs in obs.items():
        action[agent_id] = agent.compute_single_action(agent_obs)
    obs, reward, terminated, truncated, info = env.step(action)
    terminated = terminated['__all__']
    truncated = truncated['__all__']
    episode_reward += sum(reward.values())
    # print("agent moved")
    # print("current position",env.agents.get_position())

print(env.paths )

end_time = time.time()
elapsed_time = end_time - start_time
print("Time [s]: ", elapsed_time)
env.render()

