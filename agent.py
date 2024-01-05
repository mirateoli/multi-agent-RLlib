from typing import Optional
import numpy as np
from inputs import X, Y, Z, start_pts, end_pts, x_length, y_length, z_length

# set N, S, E, W to action keys for ease of understanding
N = 0
S = 1
E = 2
W = 3

actions_key = {
    0 : (0,1,0),  # North
    1 : (0,-1,0), # South
    2 : (1,0,0),  # East
    3 : (-1,0,0), # West
    4 : (0,0,1),  # Up
    5 : (0,0,-1),  # Down
    # 6 : (1,1,0),  # North-East
    # 7 : (1,-1,0), # North-West
    # 8 : (-1,1,0), # South-East
    # 9 : (-1,-1,0), # South-West
    # 10: (0,1,1),   # Up-North
    # 11: (0,1,-1),  # Down-North
    # 12: (0,-1,1),   # Up-South
    # 13: (0,-1,-1),  # Down-South
    # 14: (1,0,1),   # Up-East
    # 15: (1,0,-1),  # Down-East
    # 16: (-1,0,1),   # Up-West
    # 17: (-1,0,-1),  # Down-West
    # 18: (1,1,1),   # Up-North-East
    # 19: (1,1,-1),  # Down-North-East
    # 20: (-1,1,1),   # Up-North-West
    # 21: (-1,1,-1),  # Down-North-West
    # 22: (1,-1,1),   # Up-South-East
    # 23: (1,-1,-1),  # Down-South-East
    # 24: (-1,-1,1),   # Up-South-West
    # 25: (-1,-1,-1),  # Down-South-West
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
        if self.position[X] >= x_length:
            self.position[X] = x_length
        elif self.position[X] < 0.0:
            self.position[X] = 0.0
        if self.position[Y] >= y_length:
            self.position[Y] = y_length
        elif self.position[Y] < 0.0:
            self.position[Y] = 0.0
        if self.position[Z] >= z_length:
            self.position[Z] = z_length
        elif self.position[Z] < 0.0:
            self.position[Z] = 0.0

        return self.position

