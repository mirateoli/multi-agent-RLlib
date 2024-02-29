from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from environment_multi_combo import Environment
from inputs import start_pts, end_pts, num_pipes

import numpy as np
import os

from spaces import agent_action_space, agent_obs_space

train_ID = "MultiAgent_13_Diag"

checkpoint_dir = os.path.join('C:\\Users\\MDO-Disco\\Documents\\Thesis\\RLlib\\Checkpoints\\',train_ID)
# Create the directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

def env_creator(env_config):
    return Environment(env_config)

register_env("MultiPipe", env_creator)

env_config = {
    "num_pipes": num_pipes,
    "start_pts":start_pts,
    "end_pts":end_pts,
}

config = {
    "env": "MultiPipe",
    "env_config": env_config,
    "train_batch_size":4000,
    "lr": 5e-5,
    "entropy_coeff": 0.1,
    "num_gpus":1,
    "num_workers": 1,
    "num_envs_per_worker":1,
    "framework": "torch",
    "observation_space": agent_obs_space,
    "action_space": agent_action_space,
    }         

trainer = PPO(config=config)

# trainer.train()

for i in range(80):
    result = trainer.train()
    # print(pretty_print(result))
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint = trainer.save_checkpoint(checkpoint_path)  # Save checkpoint to specified directory
    print(f"Iteration {i+1}: {pretty_print(result)}") 

# Save the final trained model to the specified directory
trained_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint")
os.makedirs(trained_checkpoint_path, exist_ok=True)
trained_checkpoint = trainer.save_checkpoint(trained_checkpoint_path)
print(f"Final model saved at:", trained_checkpoint_path)

# Test one episode
print("TESTING NOW.......")

test_config = config
test_config["explore"]= False

agent = PPO(config=test_config)
agent.restore(trained_checkpoint_path)

env = Environment(env_config)

 # run until episode ends
episode_reward = 0
terminated = False
obs, info = env.reset()
print(obs)
while not terminated:
    action = {}
    for agent_id, agent_obs in obs.items():
        action[agent_id] = agent.compute_single_action(agent_obs)
    obs, reward, terminated, truncated, info = env.step(action)
    terminated = terminated['__all__']
    episode_reward += sum(reward.values())
    # print("agent moved")
    # print("current position",env.agents.get_position())

print(env.paths )
env.render()