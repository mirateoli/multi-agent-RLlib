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
        self._agent_ids = set(range(num_pipes))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = agent_obs_space
        self.action_space = agent_action_space

        self.paths = {i: np.array([self.start_pts[i]]) for i in range(self.num_agents)}


    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)

        # print("Environment Reset")
        self.maxsteps = 100
        info = {}
        for agent in self.agents:
            agent.initialize() 
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[i] = {
                'agent_location': agent.get_position(),
                'goal_position': agent.goal
            }

        info = {agent: {} for agent in self.agents}
        return observations, info

    def step(self, action_dict):
        observations, rewards, terminateds, truncateds, info, = {}, {}, {}, {}, {}

        self.maxsteps -= 1

        for i, action in action_dict.items():
            self.agents[i].move(action)


        observations = {i: self.get_observation(i) for i in self._agent_ids}
        rewards = {i: self.get_reward(i) for i in self._agent_ids}
        terminateds_check = {i: self.is_terminated(i) for i in self._agent_ids} # changed to try to debug
        truncateds = {i: False for i in self._agent_ids}

        terminateds["__all__"] = all(terminateds_check.values())
        truncateds["__all__"] = all(terminateds.values())

        self.paths = {i: np.vstack((self.paths[i],self.agents[i].get_position())) for i in range(self.num_agents)}

        # print("Observations:",observations,"\nTerminateds:", terminateds, "\nSteps left:",self.maxsteps)
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
        if self.maxsteps < 0:
            terminated = True
        elif(self.agents[agent_id].position == self.agents[agent_id].goal).all():
            terminated = True
        else:
            terminated = False
        return terminated
    


# env = Environment(config={"num_pipes":num_pipes, "start_pts":start_pts, "end_pts":end_pts})
# env.reset()

# obs, info = env.reset()
# print(obs)
# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 2, 1: 0}
#     )
# print(terminateds)
# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 2, 1: 0}
#     )
# print(terminateds)
# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 2, 1: 1}
#     )
# print(terminateds)
# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 2, 1: 0}
#     )
# print(terminateds)
# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 2, 1: 0}
#     )
# print(terminateds)
# while True:
#     obs, rew, terminateds, truncateds, info = env.step(
#         {0: env.action_space.sample(), 1: env.action_space.sample()}
#     )
#     # time.sleep(0.1)
    
#     if any(terminateds.values()):
#         break

# env.render()