from train_single import trained_checkpoint_path
from ray.tune.registry import register_env
from environment_single import EnvironmentSingle
import numpy as np
from train_single import *
import vedo

start_pt = np.array([0,0])
end_pt = np.array([7,7])

algo.restore(trained_checkpoint_path)

policy = algo.get_policy()

test_env = EnvironmentSingle(config={"start_pt":start_pt, "end_pt":end_pt})

obs = test_env.reset()
done = False
total_reward = 0

while not done:
    action = policy.compute_single_action(obs)
    obs, reward, done, _ = test_env.step(action)
    total_reward += reward
    print(total_reward)
    test_env.render()  # Optionally render the environment