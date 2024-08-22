import copy
import random

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from vedo import *

# from inputs import *
from agent import *
from spaces import *
import design_spaces as DS


class Environment(MultiAgentEnv):

    def __init__(self, config: EnvContext):
        # Initialize the environment and set up the basic parameters
        super().__init__()

        # Load obstacles from the dataset and find the free coordinates (i.e., those not occupied by obstacles)
        self.obstacles = DS.obstacles
        self.free_coords = self.find_free_coords(self.obstacles, DS.length, DS.width, DS.height)
        
        # Check if the environment is in training mode
        self.train = config["train"]
        self.num_pipes = config["num_pipes"]
        self.num_agents = self.num_pipes  # The number of agents equals the number of pipes

        # Randomize start and end points for training, or use fixed points for testing
        if self.train:
            self.start_pts = random.sample(self.free_coords, 3)
            self.end_pts = random.sample(self.free_coords, 3)
        else:
            self.start_pts = config["start_pts"]
            self.end_pts = config["end_pts"]

        # Initialize obstacle ranges in x, y, and z axes if obstacles are present
        if self.obstacles is not None:
            self.obs_ranges = {
                "x": [],
                "y": [],
                "z": [],
            }
            for obstacle in self.obstacles:
                self.obs_ranges['x'].append((obstacle[0], obstacle[1]))
                self.obs_ranges['y'].append((obstacle[2], obstacle[3]))
                self.obs_ranges['z'].append((obstacle[4], obstacle[5]))
        else:
            self.obs_ranges = None

        # Initialize agents with their start and end points
        self.agents = [PipeAgent(self.start_pts[i], self.end_pts[i]) for i in range(self.num_agents)]
        self._agent_ids = set(range(self.num_pipes))  # Set of agent IDs
        self.terminateds = set()  # Track terminated agents
        self.truncateds = set()   # Track truncated agents
        self.observation_space = agent_obs_space  # Define observation space for agents
        self.action_space = agent_action_space  # Define action space for agents

        # Initialize paths dictionary to store the paths taken by each agent
        self.paths = {i: np.array([self.start_pts[i]]) for i in range(self.num_agents)}

        # Initialize a variable to store the last actions of the agents
        self.last_actions = None

    def reset(self, *, seed=None, options=None):
        # Reset the environment to its initial state
        super().reset(seed=seed)

        # Reinitialize start and end points depending on the mode (training or testing)
        if self.train:
            self.start_pts = random.sample(self.free_coords, 3)
            self.end_pts = random.sample(self.free_coords, 3)
        else:
            self.start_pts = self.start_pts
            self.end_pts = self.end_pts

        # Reinitialize agents, active agents, and paths
        self.agents = [PipeAgent(self.start_pts[i], self.end_pts[i]) for i in range(self.num_agents)]
        self.active_agents = copy.deepcopy(self._agent_ids)
        self.paths = {i: np.array([self.start_pts[i]]) for i in range(self.num_agents)}

        # Reset agent parameters and observations
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

        # Reset action tracking variables
        self.actions = None
        self.last_actions = None
        self.bends = {i: 0 for i in range(self.num_agents]}

        return observations, info

    def step(self, action_dict):
        # Take a step in the environment based on the actions provided by the agents
        observations, rewards, terminateds, truncateds, info = {}, {}, {}, {}, {}

        self.maxsteps -= 1
        self.actions = action_dict

        # Move each agent based on its respective action
        for i, action in action_dict.items():
            self.agents[i].move(action)

        # Collect observations, rewards, and check termination and truncation statuses
        observations = {i: self.get_observation(i) for i in self.active_agents}
        rewards = {i: self.get_reward(i) for i in self._agent_ids}
        terminateds = {i: self.is_terminated(i) for i in self.active_agents}
        truncateds = {i: self.maxsteps <= 0 for i in self._agent_ids}

        # Check if all agents have either terminated or truncated
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        # Update paths for agents that are still active
        for i in self.active_agents:
            self.paths[i] = np.vstack((self.paths[i], self.agents[i].get_position()))

        # Remove agents that have terminated from the list of active agents
        self.active_agents = [agent_id for agent_id in self.active_agents if not terminateds[agent_id]]

        # Update the last actions performed
        self.last_actions = self.actions

        return observations, rewards, terminateds, truncateds, info

    def render(self):
        # Render the environment, visualizing the paths taken by the agents and the obstacles
        plotter = Plotter()

        # Prepare paths and key points (start and end points) for plotting
        pts = {i: self.paths[i] for i in range(self.num_agents)}
        plot_pts = {i: Points(pts[i]) for i in range(self.num_agents)}
        key_pts = {i: Points([self.start_pts[i], self.end_pts[i]]) for i in range(self.num_agents)}
        ln = {i: Line(pts[i]) for i in range(self.num_agents)}

        txt = ''

        # Add agent paths to the plot
        for pts in plot_pts.values():
            pts.color("black").ps(11)
            plotter.add(pts)

        # Add start and end points to the plot
        for pts in key_pts.values():
            pts.color("yellow").ps(12)
            plotter.add(pts)

        # Add lines representing the paths taken by the agents
        for i, lns in enumerate(ln.values()):
            txt += 'Length of line ' + str(i) + ': ' + str(lns.length()) + '\n'
            lns.color("red").linewidth(10)
            plotter.add(lns)

        plotter.add(Text2D(txt))

        # Add the room to the plot, including any obstacles
        DS.room(plotter, x_length + 1, y_length + 1, z_length + 1)

        if self.obstacles is not None:
            for obstacle in self.obstacles:
                bounding_box = obstacle.tolist()
                box = Box(size=bounding_box)
                box.color(c=(135, 206, 250))
                box.opacity(0.9)
                plotter.add(box)
            plotter.show(axes=1)
        else:
            plotter.show(axes=1).close()

    def close(self):
        pass

    def get_observation(self, agent_id):
        # Get the observation for a specific agent, including its current location, goal, and distance to goal
        return {
            'agent_location': self.agents[agent_id].get_position(),
            'goal_position': self.agents[agent_id].goal,
            'distance_to_goal': self.agents[agent_id].distance_to_goal()
        }

    def get_reward(self, agent_id):
        # Calculate the reward for a specific agent based on its actions and environment interactions

        # No reward if the agent has already terminated
        if agent_id not in self.active_agents:
            reward = 0
        else:
            reward = -0.5  # Penalty for each step taken
            if (self.agents[agent_id].position == self.agents[agent_id].goal).all():
                reward += 50  # Reward for reaching the goal
                reward -= 1 * self.path_length(self.paths[agent_id])  # Penalty for the path length taken
            else:
                reward -= 0.1 * np.abs(np.linalg.norm(self.agents[agent_id].distance_to_goal()))  # Reward for reducing distance to goal
            
            # Additional penalty if the agent moves through an obstacle
            if self.obs_ranges is not None:
                for i in range(self.obstacles.shape[0]):
                    if (self.agents[agent_id].position[0] in range(self.obs_ranges["x"][i][0], self.obs_ranges["x"][i][1] + 1)) and \
                       (self.agents[agent_id].position[1] in range(self.obs_ranges["y"][i][0], self.obs_ranges["y"][i][1] + 1)) and \
                       (self.agents[agent_id].position[2] in range(self.obs_ranges["z"][i][0], self.obs_ranges["z"][i][1] + 1)):
                        reward -= 5  # Penalty for colliding with an obstacle

            # Check for a pipe bend (a change in direction), and apply a penalty if so
            if self.last_actions is not None:
                if (np.cross(actions_key[self.last_actions[agent_id]], actions_key[self.actions[agent_id]])).any() != 0:
                    self.bends[agent_id] += 1
                    reward -= 2

            # Uncomment the following section to add a reward/penalty for pipe collisions in branched systems
            # if self.path_collision(agent_id):
            #     reward += 5  # Positive reward if branching is desired
        
        return reward
    
    def is_terminated(self, agent_id):
        # Determine if the agent has reached its goal (i.e., if it has terminated)
        return (self.agents[agent_id].position == self.agents[agent_id].goal).all()

    def remove_agent(self, agent_id):
        # Remove an agent from the set of active agents
        self.active_agents.remove(agent_id)

    def path_length(self, path):
        # Calculate the length of the path taken by an agent
        pts = Points(path)
        ln = Line(pts)
        return ln.length()
    
    def path_collision(self, agent_id):
        # Check if the path of one agent collides with another agent's path
        for i in range(self.num_agents):
            if i != agent_id:
                if (self.agents[agent_id].position == self.agents[i].position).all():
                    return True
        return False

    def find_free_coords(self, obstacles, max_x, max_y, max_z):
        """
        Generate coordinates outside the given 3D bounding boxes within the range [0, max_x/y/z].

        Parameters:
        - obstacles: numpy array of shape (N, 6) where each row is (xmin, xmax, ymin, ymax, zmin, zmax)
        - max_x/y/z: maximum values for x, y, and z coordinates

        Returns:
        - List of tuples representing coordinates outside the bounding boxes
        """
        outside_coords = []
        
        # Iterate through the entire grid space and check for points outside of any obstacle
        for x in range(max_x):
            for y in range(max_y):
                for z in range(max_z):
                    is_outside = True
                    for obstacle in obstacles:
                        xmin, xmax, ymin, ymax, zmin, zmax = obstacle
                        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                            is_outside = False
                            break
                    
                    # If the point is outside all obstacles, add it to the list
                    if is_outside:
                        outside_coords.append((x, y, z))
        
        return outside_coords

        

# # test
# num_pipes = 2
# start_pts = np.array([(4,1,4),(5,1,7)])
# end_pts = np.array([(10,11,7),(10,8,4)])

# env = Environment(config={"train":True,"num_pipes":num_pipes, "start_pts":start_pts, "end_pts":end_pts})

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

# obs, info = env.reset()
# print(obs)
