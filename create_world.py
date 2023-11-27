from inputs import *
from agent import *

class TargetSystem:
    def __init__(self):
        for agent in num_pipes:
            self.agents[agent] = PipeAgent()