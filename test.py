
import numpy as np

obstacles = np.array([2,6,2,9,2,6])

obs_ranges = {
    "x" : [obstacles[0], obstacles[1]],
    "y" : [obstacles[2], obstacles[3]],
    "z" : [obstacles[4], obstacles[5]],
}

position = [4,2,4]

# print(obs_ranges["x"][0])

if (position[0] in range(obs_ranges["x"][0],obs_ranges["x"][1])) and\
            (position[1] in range(obs_ranges["y"][0],obs_ranges["y"][1])) and\
            (position[2] in range(obs_ranges["z"][0],obs_ranges["z"][1])):
                        print("True")