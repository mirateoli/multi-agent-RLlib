
import numpy as np

from gymnasium import spaces

from inputs import num_pipes, num_directions, grid_size
# Observation spaces

agent_self_space = spaces.Box(
    shape=(3,),
    dtype=float,
    low=0.0,
    high=grid_size
)

# # might delete agent_other_space, not sure if this is useful since we actually want to know all 
# # locations that agents have passed through
# agent_others_space = spaces.Box(
#     shape=(num_pipes-1,2),
#     dtype=float,
#     low=0.0,
#     high=grid_size
# )

agent_goal_space = spaces.Box(
    shape=(3,),
    dtype=float,
    low=0.0,
    high=grid_size
)

agent_distance_to_goal_space = spaces.Box(
    shape=(1,),
    dtype=float,
    low=0.0,
    high = grid_size*grid_size,
)

agent_obs_space = spaces.Dict({
    'agent_location': agent_self_space,
    'goal_position': agent_goal_space,
    'distance_to_goal': agent_distance_to_goal_space
})

agent_action_space = spaces.Discrete(num_directions)