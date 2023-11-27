from typing import Dict

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from inputs import *
from agent import *
from spaces import *

class Environment(MultiAgentEnv):
    def __init__(self,env_config: EnvContext):
        self.num_agents = num_pipes
        self.agents = [PipeAgent(start_pts[i],end_pts[i]) for i in range(num_pipes)]
        self._agent_ids = set(range(num_pipes))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = agent_obs_space
        self.action_space = agent_action_space
        self.resetted = False

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
        observations, reward, done, info, = {}, {}, {}, {}

        for i, agent in enumerate(self.agents):
            observations[i] = {
                'agent_location': agent.move(action_dict[i]),
                'goal_position': agent.goal
            }
            if (agent.position == agent.goal).all():
                reward[i] = 10
            else:
                reward[i] = -0.1
        return observations, reward, done, info


    def render(self):
        pass

# env = Environment()
# obs,infos = env.reset()
# print(obs)

# actions = {0:2,1:2}
# obs = env.step(actions)
# print(obs)
