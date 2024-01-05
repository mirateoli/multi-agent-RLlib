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

        self.maxsteps = 5000

        try: obstacles
        except NameError: obstacles = None

        self.obstacles = obstacles

        if self.obstacles is not None:
            self.obs_ranges = {
                "x" : (obstacles[0], obstacles[1]),
                "y" : (obstacles[2], obstacles[3]),
                "z" : (obstacles[4], obstacles[5]),
            }
        else:
            self.obs_ranges = None


    def reset(self,*, seed=None, options=None):

        self.path = [self.start] # reset path to empty list
        self.agent.initialize() 

        self.maxsteps = 5000

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
        elif self.obs_ranges is not None:
            if (self.agent.position[0] in range(self.obs_ranges["x"][0],self.obs_ranges["x"][1])) and\
                (self.agent.position[1] in range(self.obs_ranges["y"][0],self.obs_ranges["y"][1])) and\
                (self.agent.position[2] in range(self.obs_ranges["z"][0],self.obs_ranges["z"][1])):
                reward = -10
                terminated = False
                truncated = False
                # print("Agent moved through obstacle")

        else:
            reward = -0.05
            terminated = False
            truncated = False

        self.path.append(self.agent.position)

        return observation, reward, terminated, truncated, info


    def render(self):
        pts = self.path
        ln = Line(pts)
        ln.color("red5").linewidth(5)
        if self.obstacles is not None:
            bounding_box = self.obstacles.tolist()
            print(bounding_box)
            box = Box(size=bounding_box)
            box.color('g4')
            show(Points(pts),ln,box,axes=1).close()
        else:
            show(Points(pts),ln,axes=1).close()

    def get_route(self):
        return self.path


# start_pt = np.array([0,0,0])
# end_pt = np.array([7,7,5])

# env = EnvironmentSingle(config={"start_pt":start_pt, "end_pt":end_pt})
# env.reset()
# env.step(action=2)
# print(env.path)
# env.render()
# env.step(action=2)
# print(env.path)
# env.render()
# env.step(action=0)
# print(env.path)
# env.render()
# # print(env.action_space)
# # print(env.observation_space)

# check_env(env)



