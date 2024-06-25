from numpy.random import randint

import copy

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from vedo import *

# from inputs import *
from agent import *
from spaces import *
import design_spaces as DS


class Environment(MultiAgentEnv):
    
    def __init__(self, config: EnvContext):
        super().__init__()

        
        self.train = config["train"]
        self.num_pipes = config["num_pipes"]
        self.num_agents = self.num_pipes

        # if training, randomize start and end points
        if self.train:
            # self.start_pts = randint(0, grid_size, size=(self.num_agents, 3))
            self.start_pts =config["start_pts"] # same start points if doing branching
            self.end_pts = randint(0, grid_size, size=(self.num_agents, 3))
        # if testing, use defined start and end points
        else:
            self.start_pts = config["start_pts"]
            self.end_pts = config["end_pts"]

        self.obstacles = DS.obstacles

        if self.obstacles is not None:
            self.obs_ranges = {
                    "x" : [],
                    "y" : [],
                    "z" : [],
                }
            for obstacle in self.obstacles:
                self.obs_ranges['x'].append((obstacle[0], obstacle[1]))
                self.obs_ranges['y'].append((obstacle[2], obstacle[3]))
                self.obs_ranges['z'].append((obstacle[4], obstacle[5]))
                
        else:
            self.obs_ranges = None

        self.agents = [PipeAgent(self.start_pts[i], self.end_pts[i]) for i in range(self.num_agents)]
        self._agent_ids = set(range(self.num_pipes))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = agent_obs_space
        self.action_space = agent_action_space

        self.paths = {i: np.array([self.start_pts[i]]) for i in range(self.num_agents)}


    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)

        # if training, randomize start and end points
        if self.train:
            self.start_pts = randint(0, grid_size, size=(self.num_agents, 3))
            self.end_pts = randint(0, grid_size, size=(self.num_agents, 3))
        # if testing, use defined start and end points
        else:
            self.start_pts = self.start_pts
            self.end_pts = self.end_pts

        self.agents = [PipeAgent(self.start_pts[i], self.end_pts[i]) for i in range(self.num_agents)]

        self.active_agents = copy.deepcopy(self._agent_ids)
        self.paths = {i: np.array([self.start_pts[i]]) for i in range(self.num_agents)}

        # print("Environment Reset")
        self.maxsteps = 100
        info = {}
        for agent in self.agents:
            agent.initialize() 
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[i] = {
                'agent_location': agent.get_position(),
                'goal_position': agent.goal,
                'distance_to_goal': agent.distance_to_goal()
            }

        info = {agent: {} for agent in self.agents}
        return observations, info

    def step(self, action_dict):
        observations, rewards, terminateds, truncateds, info, = {}, {}, {}, {}, {}

        self.maxsteps -= 1

        for i, action in action_dict.items():
            self.agents[i].move(action)


        observations = {i: self.get_observation(i) for i in self.active_agents}
        rewards = {i: self.get_reward(i) for i in self._agent_ids}
        terminateds = {i: self.is_terminated(i) for i in self.active_agents} # changed to try to debug
        truncateds = {i: self.maxsteps <= 0 for i in self._agent_ids}

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        #TODO only add to path if agent was active
        for i in self.active_agents:
            self.paths[i] = np.vstack((self.paths[i],self.agents[i].get_position()))


        # remove agent from active agents if terminated is true
        # Remove agent from active_agents if terminated is true
        self.active_agents = [agent_id for agent_id in self.active_agents if not terminateds[agent_id]]


        # print("Observations:",observations,"\nTerminateds:", terminateds, "\nSteps left:",self.maxsteps)
        return observations, rewards, terminateds, truncateds, info


    def render(self):
        plotter = Plotter()

        pts = {i: self.paths[i] for i in range(self.num_agents)}
        plot_pts = {i: Points(pts[i]) for i in range(self.num_agents)}

        key_pts = {i: Points([self.start_pts[i],self.end_pts[i]]) for i in range(self.num_agents)}                    

        ln = {i: Line(pts[i]) for i in range(self.num_agents)}
        # ln[0].color("red5").linewidth(10)
        # ln[1].color("green").linewidth(10)
        # ln[2].color("blue").linewidth(10)

        txt = ''

        for pts in plot_pts.values():
            pts.color("black").ps(11)
            plotter.add(pts)

        for pts in key_pts.values():
            pts.color("yellow").ps(12)
            plotter.add(pts)

        for i, lns in enumerate(ln.values()):
            txt += 'Length of line ' + str(i) +': ' +str(lns.length()) + '\n'
            lns.color("red").linewidth(10)
            plotter.add(lns)

        plotter.add(Text2D(txt))

        DS.room(plotter, x_length+1, y_length+1, z_length+1)

        if self.obstacles is not None:
            for obstacle in self.obstacles:
                bounding_box = obstacle.tolist()
                box = Box(size=bounding_box)
                box.color(c=(135,206,250))
                box.opacity(0.9)
                plotter.add(box)
            plotter.show(axes=1)
            # show(key_pts[0], key_pts[1],Points(pts[0]),Points(pts[1]),ln[0], ln[1],box,axes=1).close()
        else:
            plotter.show(axes=1).close()

    def close(self):
        pass

    def get_observation(self, agent_id):
        return {
                'agent_location': self.agents[agent_id].get_position(),
                'goal_position': self.agents[agent_id].goal,
                'distance_to_goal': self.agents[agent_id].distance_to_goal()
                }
    def get_reward(self, agent_id):
        reward = -0.5 # penalty for every step
        if(self.agents[agent_id].position == self.agents[agent_id].goal).all():
            reward += 50 # reward for reaching goal
            reward += - 1 * self.path_length(self.paths[agent_id]) # penalty for path length
        else:
            reward += - 0.1 * np.abs(np.linalg.norm(self.agents[agent_id].distance_to_goal())) # reward for how far from goal
        if self.obs_ranges is not None:
            for i in range(self.obstacles.shape[0]):
                if (self.agents[agent_id].position[0] in range(self.obs_ranges["x"][i][0],self.obs_ranges["x"][i][1]+1)) and\
                    (self.agents[agent_id].position[1] in range(self.obs_ranges["y"][i][0],self.obs_ranges["y"][i][1]+1)) and\
                    (self.agents[agent_id].position[2] in range(self.obs_ranges["z"][i][0],self.obs_ranges["z"][i][1]+1)):
                    reward += -2 # penalty for moving through obstacle
                    # print(agent_id,"collided with obstacle")
        
        # check for pipe collision
        if self.path_collision(agent_id):
            # print("Pipe ",agent_id,"collided")
            reward += 5 # positive if branching
        
        return reward
    
    def is_terminated(self, agent_id):
        if(self.agents[agent_id].position == self.agents[agent_id].goal).all():
            terminated = True
        else:
            terminated = False
        return terminated
    
    def remove_agent(self, agent_id):
        self.active_agents.remove(agent_id)

    def path_length(self, path):
        pts = Points(path)
        ln = Line(pts)
        return ln.length()
    
    def path_collision(self, agent_id):
        for i in range(self.num_agents):
            if i != agent_id:
                if (self.agents[agent_id].position == self.agents[i].position).all():
                    return True
                else:
                    return False
            else:
                continue

    def output_file(self, file_path):
        # Open the file in append mode ('a') or write mode ('w')
        # 'a' mode: appends to the end of the file if it exists, or creates a new file if it doesn't
        # 'w' mode: opens the file for writing, or creates a new file if it doesn't exist
        with open(file_path, 'a') as file:
            # Write some content to the file
            file.write('Hello, world!\n')
            
# env = Environment(config={"train":False,"num_pipes":num_pipes, "start_pts":start_pts, "end_pts":end_pts})

# obs, info = env.reset()
# print(obs)


# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 0, 1: 0}
#     )
# print(rew)

# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 0, 1: 0}
#     )
# print(rew)

# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 2, 1: 0}
#     )
# print(rew)

# obs, rew, terminateds, truncateds, info = env.step(
#         {0: 2, 1: 1}
#     )

# print(rew)

# env.render()
