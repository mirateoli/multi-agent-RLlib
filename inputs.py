import numpy as np


# obstacle_box = np.array([(2,2,2),(2,2,6),(2,6,6),(6,6,6),(6,2,2),(6,2,6),(2,6,2),(6,6,2)])

# obstacles = np.array([1,8,1,5,1,5])  # bounding box (xmin,xmax,ymin,ymax,zmin,zmax)

# try: obstacles
# except NameError: obstacles = None

x_length = 6
y_length = 6
z_length = 6

grid_size = 6

grid_density = 1

step_size = 1/grid_density

X = 0
Y = 1
Z = 2

start_pts = np.array([(0,0,0),(3,0,0),(5,0,0)])
end_pts = np.array([(0,5,5),(3,5,5),(5,5,5)])


num_pipes = 3

num_directions = 18

step_reward = -0.001
arrival_reward = 1.0
collision_reward = -0.5

# print(obstacles)