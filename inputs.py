import numpy as np

x_length = 4
y_length = 4
z_length = 4

X = 0
Y = 1
Z = 2

end_pts = np.array([(3,3),(0,3)])
start_pts = np.array([(3,0),(0,0)])

num_pipes = 2

num_directions = 4

step_reward = -0.001
arrival_reward = 1.0
collision_reward = -0.5