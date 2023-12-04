from train_single import trained_checkpoint_path
from ray.tune.registry import register_env
from environment_single import EnvironmentSingle
from ray.rllib.evaluation import RolloutWorker
import numpy as np
from train_single import trained_checkpoint_path, env_config, config
from ray.rllib.algorithms.ppo import PPO
import vedo
import os

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

def log_route(info):
    pipe_routes = info["env"].get_route()
    print("Agent locations: {}".format(pipe_routes))
    return pipe_routes

start_pt = np.array([0,0])
end_pt = np.array([7,7])

agent = PPO(config=config)
agent.restore(trained_checkpoint_path)

env = EnvironmentSingle(env_config)

 # run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    print("agnet moved")
    print("current path",env.path)

print(env.path)
