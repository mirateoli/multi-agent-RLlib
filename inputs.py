import numpy as np


# obstacle_box = np.array([(2,2,2),(2,2,6),(2,6,6),(6,6,6),(6,2,2),(6,2,6),(2,6,2),(6,6,2)])

# obstacles = np.array([2,6,2,9,2,6])  # bounding box (xmin,ymin,zmin,xmax,ymax,zmax)

x_length = 6
y_length = 6
z_length = 6

grid_size = 7

grid_density = 1

step_size = 1/grid_density

X = 0
Y = 1
Z = 2

end_pts = np.array([(4,0),(0,3)])
start_pts = np.array([(3,0),(0,0)])

num_pipes = 2

num_directions = 18

step_reward = -0.001
arrival_reward = 1.0
collision_reward = -0.5

# print(obstacles)