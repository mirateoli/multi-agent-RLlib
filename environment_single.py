from typing import Dict

import ray
import gymnasium as gym

import vedo

from ray.rllib.env import EnvContext
from ray.rllib.utils import check_env

from inputs import *
from agent import *
from spaces import *

class EnvironmentSingle(gym.Env):
    
    def __init__(self,config: EnvContext):
        super().__init__()
        self.start = config["start_pt"]
        self.goal = config["end_pt"]
        self.agent = PipeAgent(self.start,self.goal)

        self.observation_space = agent_obs_space
        self.action_space = agent_action_space

        

    def reset(self,*, seed=None, options=None):

        self.agent.initialize() 

        observation = {
            'agent_location': self.agent.get_position(),
            'goal_position': self.agent.goal
        }        
        # print(observations)
        info = {}
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info, = {}, {}, {}, {}, {}
        
        observation = {
            'agent_location': self.agent.move(action),
            'goal_position': self.agent.goal
        }
        if (self.agent.position == self.agent.goal).all():
            reward = 10
            terminated = True
            truncated = False
        else:
            reward = -0.1
            terminated = False
            truncated = False

        return observation, reward, terminated, truncated, info


    def render(self):
        pass

    def close(self):
        pass

# env = EnvironmentSingle(start_pts[0],end_pts[0])
# # print(env.action_space)
# # print(env.observation_space)

# check_env(env)



