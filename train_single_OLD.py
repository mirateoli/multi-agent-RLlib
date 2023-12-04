from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from environment_single import EnvironmentSingle
import numpy as np
import os
import ray

start_pt = np.array([0,0])
end_pt = np.array([7,7])

train_ID = "Test1"

checkpoint_dir = os.path.join('C:\\Users\\MDO-Disco\\Documents\\Thesis\\RLlib\\Checkpoints\\',train_ID)
# Create the directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

def env_creator(env_config):
    return EnvironmentSingle(env_config)

register_env("SinglePipe", env_creator)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="SinglePipe", env_config={"start_pt":start_pt, "end_pt":end_pt})
    .build()
)

for i in range(10):
    result = algo.train()
    # print(pretty_print(result))
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = algo.save_checkpoint(checkpoint_path)  # Save checkpoint to specified directory
    print(f"Iteration {i+1}: {result}")

# Save the final trained model to the specified directory
trained_checkpoint_path = algo.save_checkpoint(os.path.join(checkpoint_dir, "final_checkpoint"))
print(f"Final model saved at: {trained_checkpoint_path}")

