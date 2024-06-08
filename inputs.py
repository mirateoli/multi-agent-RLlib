import numpy as np
# import select_points_GUI as SPG
import design_spaces as DS



# obstacles = np.array([(2,6,2,8,0,5),(9,12,9,12,0,6)])  # bounding box (xmin,xmax,ymin,ymax,zmin,zmax)

obstacles = DS.obstacles

# num_pipes, key_pts = SPG.select_pts(obstacles)

# start_pts = key_pts[::2] # get even indices
# end_pts = key_pts[1::2]   # get odd indices

x_length = DS.length
y_length = DS.width
z_length = DS.height

grid_size = max(x_length, y_length, z_length)

num_directions = 18



# start_pts = np.array([(4,1,4),(4,1,4),(4,1,4)])
# end_pts = np.array([(10,11,7),(10,8,4),(8,11,3)])


# print(obstacles)