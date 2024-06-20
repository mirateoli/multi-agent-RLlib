from environment import Environment

from agent import *
from spaces import *
import design_spaces as DS
import select_points_GUI as SPG

num_pipes, key_pts = SPG.select_pts(DS.obstacles)

start_pts = key_pts[::2] # get even indices
end_pts = key_pts[1::2]   # get odd indices

env = Environment(config={"train":False,"num_pipes":num_pipes, "start_pts":start_pts, "end_pts":end_pts})

obs, info = env.reset()
print(obs)


obs, rew, terminateds, truncateds, info = env.step(
        {0: 0, 1: 0}
    )
print(rew)

obs, rew, terminateds, truncateds, info = env.step(
        {0: 0, 1: 0}
    )
print(rew)

obs, rew, terminateds, truncateds, info = env.step(
        {0: 2, 1: 0}
    )
print(rew)

obs, rew, terminateds, truncateds, info = env.step(
        {0: 2, 1: 1}
    )

print(rew)

env.render()
