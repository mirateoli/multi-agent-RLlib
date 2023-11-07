from typing import Dict

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

class Environment(MultiAgentEnv):
    def __init__(self, config: EnvContext):
        self.observation_space = None
        self.is_use_visualization = config['is_use_visualization']

        self.visualization = None

    def reset(self):
        

        