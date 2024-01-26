from typing import Dict

import ray

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext
from ray.rllib.utils import check_env

from vedo import *

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
        self.agents = [PipeAgent(self.start_pts[i], self.end_pts[i]) for i in range(self.num_agents)]
        # self.agents = [EnvironmentSingle(config={"start_pt":self.start_pts[i], "end_pt":self.end_pts[i]}) for i in range(self.num_agents)]
        self._agent_ids = set(range(num_pipes))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = agent_obs_space
        self.action_space = agent_action_space
        self.resetted = False
        self.paths = {i: np.array([self.start_pts[i]]) for i in range(self.num_agents)}
         

        print(self.paths)
        print(self.paths[0])


    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)

        infos = {}
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
        observations, rewards, terminateds, truncateds, info, = {}, {}, {}, {}, {}

        for i, action in action_dict.items():
            print(i)
            print(action)
            self.agents[i].move(action)


        observations = {i: self.get_observation(i) for i in self._agent_ids}
        rewards = {i: self.get_reward(i) for i in self._agent_ids}
        terminateds = {i: self.is_terminated(i) for i in self._agent_ids}
        truncateds = {i: False for i in self._agent_ids}

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(terminateds.values())

        self.paths = {i: np.vstack((self.paths[i],self.agents[i].get_position())) for i in range(self.num_agents)}
        print(self.paths)

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
        pts = {i: self.paths[i] for i in range(self.num_agents)}
        key_pts = {i: Points([self.start_pts[i],self.end_pts[i]]) for i in range(self.num_agents)}                    
        key_pts[0].color("blue").ps(10)
        key_pts[1].color("orange").ps(10)
        ln = {i: Line(pts[i]) for i in range(self.num_agents)}
        ln[0].color("red5").linewidth(5)
        ln[1].color("yellow").linewidth(5)
        show(key_pts[0], key_pts[1], Points(pts[0]),Points(pts[1]),ln[0], ln[1],axes=1).close()
        # print(ln[0])
        # show(ln[0])

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

env.reset()

while True:
    obs, rew, terminateds, truncateds, info = env.step(
        {0: env.action_space.sample(), 1: env.action_space.sample()}
    )
    # time.sleep(0.1)
    
    if any(terminateds.values()):
        break

env.render()