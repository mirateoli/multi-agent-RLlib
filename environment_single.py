from typing import Dict

import ray
import gymnasium as gym

from vedo import *

from ray.rllib.env import EnvContext
from ray.rllib.utils import check_env

from inputs import *
from agent import *
from spaces import *

class EnvironmentSingle(gym.Env):

    metadata = {
        "render.modes": ["rgb_array"],
    }
    
    def __init__(self,config: EnvContext):
        super().__init__()
        self.start = config["start_pt"]
        self.goal = config["end_pt"]
        self.agent = PipeAgent(self.start,self.goal)

        self.observation_space = agent_obs_space
        self.action_space = agent_action_space

        self.path = [self.start] # list to store all locations of 

        self.maxsteps = 10000

        

    def reset(self,*, seed=None, options=None):

        self.path = [self.start] # reset path to empty list
        self.agent.initialize() 

        self.maxsteps = 10000

        observation = {
            'agent_location': self.agent.get_position(),
            'goal_position': self.agent.goal
        }        
        # print(observations)
        info = {}
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info, = {}, {}, {}, {}, {}

        self.maxsteps -= 1
        
        observation = {
            'agent_location': self.agent.move(action),
            'goal_position': self.agent.goal
        }

        if self.maxsteps <= 0:
            terminated = True
            truncated = True
        elif (self.agent.position == self.agent.goal).all():
            reward = 10
            terminated = True
            truncated = False
        else:
            reward = -0.1
            terminated = False
            truncated = False

        self.path.append(self.agent.position)

        return observation, reward, terminated, truncated, info


    def render(self):
        pts = self.path
        ln = Line(pts)
        ln.color("red5").linewidth(5)
        show(Points(pts),ln,axes=1).close()

    def get_route(self):
        return self.path


# start_pt = np.array([0,0,0])
# end_pt = np.array([7,7,5])

# env = EnvironmentSingle(config={"start_pt":start_pt, "end_pt":end_pt})
# env.reset()
# env.step(action=2)
# print(env.path)
# # env.render()
# env.step(action=2)
# print(env.path)
# env.render()
# env.step(action=0)
# print(env.path)
# env.render()
# # print(env.action_space)
# # print(env.observation_space)

# check_env(env)



