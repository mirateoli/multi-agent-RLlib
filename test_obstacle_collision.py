import numpy as np
import random

from design_spaces import obstacles

def get_outside_coordinates(bounding_box, max_x, max_y, max_z):
    # this one has different x y and z max values
    outside_coords = []
    
    for x in range(max_x):
        for y in range(max_y):
            for z in range(max_z):
                is_outside = True
                for obstacle in obstacles:
                    xmin, xmax, ymin, ymax, zmin, zmax = obstacle
                    if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                        is_outside = False
                        break
                
                if is_outside:
                    outside_coords.append((x, y, z))
    
    return outside_coords

def coordinates_outside_bboxes(obstacles, max_value):
    """
    Generate coordinates outside the given 3D bounding boxes within the range [0, max_value].

    Parameters:
    - obstacles: numpy array of shape (N, 6) where each row is (xmin, xmax, ymin, ymax, zmin, zmax)
    - max_value: maximum value for x, y, and z coordinates

    Returns:
    - List of tuples representing coordinates outside the bounding boxes
    """
    outside_coordinates = []
    
    # Generate all coordinates in the range [0, max_value]
    for x in range(max_value + 1):
        for y in range(max_value + 1):
            for z in range(max_value + 1):
                outside = True
                # Check if the coordinate is inside any of the bounding boxes
                for bbox in obstacles:
                    x_min, x_max, y_min, y_max, z_min, z_max = bbox
                    if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                        outside = False
                        break
                if outside:
                    outside_coordinates.append((x, y, z))
    
    return outside_coordinates

# Example usage

max_value = 20

max_x = 20  # Example maximum value for x
max_y = 20  # Example maximum value for y
max_z = 16  # Example maximum value for z

# outside_coords = coordinates_outside_bboxes(obstacles, max_value)
# print(outside_coords)

# start_pt = [(1,1,1)]
# end_pt = [(1,1,1)]

# while (start_pt == end_pt):
#     print("same points selected")
#     end_pt = random.sample(outside_coords,1)

# print('found two unique points')

# print("start:", start_pt, "\n end:", end_pt)

obstacles = np.array([(2, 6, 2, 8, 0, 5), (9, 14, 8, 12, 0, 4), (0, 2, 12, 18, 0, 3), (18, 20, 3, 8, 0, 9)])

outside_coords = get_outside_coordinates(obstacles, max_x, max_y, max_z)
print(outside_coords)

start_pt = random.sample(outside_coords,1) * 3
end_pt = random.sample(outside_coords,3)

while (start_pt == end_pt):
    print("same points selected")
    end_pt = random.sample(outside_coords,1)

print('found two unique points')

print("start:", start_pt, "\n end:", end_pt)

coordinate_to_check = (4,4,4)
if coordinate_to_check in outside_coords:
    print(f"{coordinate_to_check} is in the list of outside coordinates.")
else:
    print(f"{coordinate_to_check} is not in the list of outside coordinates.")