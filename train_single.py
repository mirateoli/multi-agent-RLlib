from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from environment_single import EnvironmentSingle
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import numpy as np
import os
import ray

start_pt = np.array([0,0,0])
end_pt = np.array([18,14,19])

train_ID = "Test5_3D"

class CustomCallbacks(DefaultCallbacks):
    def log_route(info):
        pipe_routes = info["env"].get_route()
        print("Agent locations: {}".format(pipe_routes))
        return pipe_routes

checkpoint_dir = os.path.join('C:\\Users\\MDO-Disco\\Documents\\Thesis\\RLlib\\Checkpoints\\',train_ID)
# Create the directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)



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
    "num_workers": 1,
    "framework": "torch",
    "callbacks": CustomCallbacks,
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

trainer = PPO(config=config)

for i in range(30):
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

