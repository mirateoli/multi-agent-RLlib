from typing import Dict

import ray

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext
from ray.rllib.utils import check_env

from inputs import *
from agent import *
from spaces import *

from environment_single import EnvironmentSingle

class Environment(MultiAgentEnv):
    
    def __init__(self, config: EnvContext):
        super().__init__()
        self.num_pipes = config["num_pipes"]
        self.start_pts = config["start_pts"]
        self.end_pts = config["end_pts"]
        self.num_agents = self.num_pipes
        self.agents = [EnvironmentSingle(config={"start_pt":self.start_pts[i], "end_pt":self.end_pts[i]}) for i in range(self.num_agents)]
        self._agent_ids = set(range(num_pipes))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.resetted = False


    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)

        infos = {}
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        for agent in self.agents:
            agent.agent.initialize() 
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[i] = {
                'agent_location': agent.agent.get_position(),
                'goal_position': agent.goal
            }
        
        # print(observations)

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action_dict):
        observations, rewards, terminateds, truncateds, info, = {}, {}, {}, {}, {}
        
        agent_ids = action_dict.keys()

        for agent_id in agent_ids:
            print(self.agents[agent_id])
            # self.agents[agent_id].move(action_dict[agent_id])


        observations = {i: self.get_observation(i) for i in agent_ids}
        rewards = {i: self.get_reward(i) for i in agent_ids}
        terminateds = {i: self.is_terminated(i) for i in agent_ids}
        truncateds = {i: False for i in agent_ids}

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(terminateds.values())

        # for i, agent in enumerate(self.agents):
        #     print(i)
        #     # print(action_dict[i])
        #     observation[i] = {
        #         'agent_location': agent.agent.move(action_dict[i]),
        #         'goal_position': agent.goal
        #     }
        #     if (agent.agent.position == agent.goal).all():
        #         reward[i] = 10
        #         terminated[i] = True
        #         truncated[i] = False
        #     else:
        #         reward[i] = -0.1
        #         terminated[i] = False
        #         truncated[i] = False
        # terminated["__all__"] = len(self.terminateds) == len(self.agents)
        # truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return observations, rewards, terminateds, truncateds, info


    def render(self):
        pass

    def close(self):
        pass

    def get_observation(self, agent_id):
        return {
                'agent_location': self.agents[agent_id].get_position(),
                'goal_position': self.agents[agent_id].goal
                }
    def get_reward(self, agent_id):
        if(self.agents[agent_id].position == self.agents[agent_id].goal).all():
            reward = 10

        else:
            reward = -0.1

        return reward
    
    def is_terminated(self, agent_id):
        if(self.agents[agent_id].position == self.agents[agent_id].goal).all():
            terminated = True
        else:
            terminated = False
        return terminated


env = Environment(config={"num_pipes":num_pipes, "start_pts":start_pts, "end_pts":end_pts})

obs = env.reset()
print(obs)

while True:
    obs, rew, done, info = env.step(
        {1: env.action_space.sample(), 2: env.action_space.sample()}
    )
    # time.sleep(0.1)
    env.render()
    if any(done.values()):
        break

