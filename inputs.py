import numpy as np


# obstacle_box = np.array([(2,2,2),(2,2,6),(2,6,6),(6,6,6),(6,2,2),(6,2,6),(2,6,2),(6,6,2)])
obstacles = np.array([(2,6,2,8,0,5),(9,12,9,12,0,6)])  # bounding box (xmin,xmax,ymin,ymax,zmin,zmax)
# obstacles = np.array([(2,6,2,8,0,5)])
# obstacles = None

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

start_pts = np.array([(1,4,1),(4,1,3)])
end_pts = np.array([(10,10,6),(10,9,4)])


num_pipes = 2

num_directions = 6



# print(obstacles)