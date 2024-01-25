from typing import Dict

import ray


from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.env import EnvContext
from ray.rllib.utils import check_env
from ray.tune.registry import register_env

from inputs import *
from agent import *
from spaces import *

from environment_single import EnvironmentSingle

start_pt = np.array([2,0,3])
end_pt = np.array([9,3,2])

def env_creator(env_config):
    return EnvironmentSingle(env_config)

register_env("SinglePipe", env_creator)

env_config = {
    "start_pt":start_pt,
    "end_pt":end_pt,
}

env_creator

MultiPipe = make_multi_agent("SinglePipe")

MultiPipe = MultiPipe({"num_agents": 2})

# register_env("multi_agent_pendulum", lambda _: MultiPipe({"num_agents": 2}))

# env = MultiPipe({"num_agents": 2})