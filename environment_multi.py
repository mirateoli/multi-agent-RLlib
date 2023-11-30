from typing import Dict

import ray

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext
from ray.rllib.utils import check_env

from inputs import *
from agent import *
from spaces import *

class Environment(MultiAgentEnv):
    
    def __init__(self, env_config: EnvContext):
        super().__init__()
        self.num_agents = num_pipes
        self.agents = [PipeAgent(start_pts[i],end_pts[i]) for i in range(num_pipes)]
        self._agent_ids = set(range(num_pipes))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = agent_obs_space
        self.action_space = agent_action_space
        self.resetted = False

        self.disable_env_checking=True
        

    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)

        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        for agent in self.agents:
            agent.initialize() 
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[i] = {
                'agent_location': agent.get_position(),
                'goal_position': agent.goal
            }
        
        # print(observations)

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action_dict):
        observation, reward, terminated, truncated, info, = {}, {}, {}, {}, {}
        

        for i, agent in enumerate(self.agents):
            observation[i] = {
                'agent_location': agent.move(action_dict[i]),
                'goal_position': agent.goal
            }
            if (agent.position == agent.goal).all():
                reward[i] = 10
                terminated[i] = True
                truncated[i] = False
            else:
                reward[i] = -0.1
                terminated[i] = False
                truncated[i] = False
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return observation, reward, terminated, truncated, info


    def render(self):
        pass

    def close(self):
        pass

# env = Environment()
# # print(env._agent_ids)
# # ids = env.get_agent_ids()
# # print(ids)

# # print(env.action_space)
# # print(env.observation_space)


# print(env.action_space)
# print(env.action_space.sample())
# print(env.action_space_sample())

# print(env.observation_space)
# print(env.observation_space_sample())

# check_env(env)


# ray.rllib.utils.check_env(env)
# obs,infos = env.reset()
# print(obs)

# actions = {0:2,1:2}
# obs = env.step(actions)
# print(obs)
