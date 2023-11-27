from typing import Optional
import numpy as np
from inputs import X, Y, Z, start_pts, end_pts, x_length, y_length, z_length

# set N, S, E, W to action keys for ease of understanding
N = 0
S = 1
E = 2
W = 3

actions_key = {
    0 : (0,1),  # North
    1 : (0,-1), # South
    2 : (1,0),  # East
    3 : (-1,0), # West
}


class PipeAgent:
    def __init__(self, start_pt, end_pt):
        self.position: Optional[np.darray] = None
        self.start_pt = start_pt
        self.end_pt = end_pt

    def initialize(self) -> None:
        # Pipe should be initialized to its start point

        self.position = self.start_pt
        self.goal = self.end_pt
        return self.position,self.goal

    def get_position(self):
        return self.position
    
    def move(self, direction: int) -> None:
        # 0,0 is top left for rendering, y down is positive
        
        # move agent to new position based on action
        self.position = np.add(self.position, actions_key[direction])

        # don't let agent move out of field
        if self.position[X] > x_length:
            self.position[X] = x_length
        elif self.position[X] < 0.0:
            self.position[X] = 0.0
        if self.position[Y] > y_length:
            self.position[Y] = y_length
        elif self.position[Y] < 0.0:
            self.position[Y] = 0.0

        return self.position

