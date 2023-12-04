import numpy as np

x_length = 8
y_length = 8
z_length = 8

grid_size = 8

grid_density = 1

step_size = 1/grid_density

X = 0
Y = 1
Z = 2

end_pts = np.array([(4,0),(0,3)])
start_pts = np.array([(3,0),(0,0)])

num_pipes = 2

num_directions = 6

step_reward = -0.001
arrival_reward = 1.0
collision_reward = -0.5

