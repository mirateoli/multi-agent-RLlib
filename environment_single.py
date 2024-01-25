from typing import Dict

# generate random integer values
from numpy.random import seed
from numpy.random import randint

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

        self.train = config["train"]

        # if training, randomize start and end points
        if self.train:
            self.start = randint(0, grid_size, 3)
            self.goal = randint(0, grid_size, 3)
        # if testing, use defined start and end points
        else:
            self.start = config["start_pt"]
            self.goal = config["end_pt"]

        self.agent = PipeAgent(self.start,self.goal)
        
        self.last_action = None
        self.bends = 0

        self.observation_space = agent_obs_space
        self.action_space = agent_action_space

        self.path = [self.start] # list to store all locations of 

        self.maxsteps = 10000


        # self.obstacles = obstacles
        # # print(self.obstacles)

        # if self.obstacles is not None:
        #     self.obs_ranges = {
        #         "x" : (obstacles[0], obstacles[1]),
        #         "y" : (obstacles[2], obstacles[3]),
        #         "z" : (obstacles[4], obstacles[5]),
        #     }
        # else:
        #     self.obs_ranges = None

            
    def reset(self,*, seed=None, options=None):

        # if training, randomize start and end points
        if self.train:
            self.start = randint(0, grid_size, 3)
            self.goal = randint(0, grid_size, 3)
        # if testing, use defined start and end points
        else:
            self.start = self.start
            self.goal = self.goal

        self.agent = PipeAgent(self.start,self.goal)

        self.last_action = None
        self.bends = 0

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

        # check if max steps reached
        if self.maxsteps <= 0:
            reward = -0.1
            terminated = True
            truncated = False
        # check if agent reached its goal
        elif (self.agent.position == self.agent.goal).all():
            reward = 10
            terminated = True
            truncated = False
        # check if agent changed directions (bend)
        elif self.last_action != None:
            if (np.cross(actions_key[self.last_action], 
                         actions_key[action])).any() != 0:
                self.bends += 1
                reward = -1
            else:
                reward = -0.1
        # elif self.obs_ranges is not None:
        #     if (self.agent.position[0] in range(self.obs_ranges["x"][0],self.obs_ranges["x"][1]+1)) and\
        #         (self.agent.position[1] in range(self.obs_ranges["y"][0],self.obs_ranges["y"][1]+1)) and\
        #         (self.agent.position[2] in range(self.obs_ranges["z"][0],self.obs_ranges["z"][1]+1)):
        #         reward = -0.25
        #         observation["agent_location"] = self.agent.reverse(action) #make agent go back to position before going into obstacle
        #         terminated = False
        #         truncated = False
        #         # print("Agent moved through obstacle")
            # else:
            #     reward = -0.1
            #     terminated = False
            #     truncated = False
        # if no other criteria met, give agent penalty for taking a step
        else:
            reward = -0.1
            terminated = False
            truncated = False

        self.path.append(self.agent.position) # add to list of path locations
        self.last_action = action #set last_action to current action 

        return observation, reward, terminated, truncated, info


    def render(self):
        pts = self.path
        key_pts = Points([self.start, self.goal])
        key_pts.color("blue").ps(10)
        ln = Line(pts)
        ln.color("red5").linewidth(5)
        # if self.obstacles is not None:
        # bounding_box = self.obstacles.tolist()
        # box = Box(size=bounding_box)
        # box.color('g4')
        # box.opacity(0.5)
        show(key_pts, Points(pts),ln, axes=1).close()
        # else:
        #     show(Points(pts),ln,axes=1).close()
        print(self.bends)

    def get_route(self):
        return self.path


# start_pt = np.array([0,0,0])
# end_pt = np.array([5,5,0])

# env = EnvironmentSingle(config=None)
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
# env.step(action=2)
# print(env.path)
# env.render()
# env.step(action=1)
# print(env.path)
# env.render()
# env.step(action=3)
# print(env.path)
# env.render()
# # print(env.action_space)
# # print(env.observation_space)

# check_env(env)



