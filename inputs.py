import numpy as np


# obstacle_box = np.array([(2,2,2),(2,2,6),(2,6,6),(6,6,6),(6,2,2),(6,2,6),(2,6,2),(6,6,2)])
obstacles = np.array([1,8,1,5,1,5])  # bounding box (xmin,xmax,ymin,ymax,zmin,zmax)

# try: obstacles
# except NameError: obstacles = None

x_length = 12
y_length = 12
z_length = 12

grid_size = 12

grid_density = 1

step_size = 1/grid_density

X = 0
Y = 1
Z = 2

start_pts = np.array([(0,0,0),(6,0,0),(12,0,0)])
end_pts = np.array([(12,12,12),(6,12,12),(12,12,12)])


num_pipes = 3

num_directions = 18

step_reward = -0.001
arrival_reward = 1.0
collision_reward = -0.5

# print(obstacles)