import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
import copy
import json
import zmq
from multiprocessing.pool import ThreadPool
from PathTrajectory import PathTrajectory
from VelocityTuner import VelocityTuner
from TrajectorySmoothing import TrajectorySmoothing

from numba import njit

"""
Planner Helpers
"""

# RRT params
MAX_ITER = 300
STEER_LENGTH = 0.3
TERMINATE_LENGTH = 0.2
ETA = 0.6
X_SEARCH_LIM = 10.0
Y_SEARCH_LIM = 10.0
# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

# -3.14159, 3.14159, 0.00582316


def get_laser_scan_obstacles():
    """
    Returns list of grid points that the occupancy grid views as occupied in the
    global frame. 
    """
    laser_scan_obst = []
    return laser_scan_obst


def get_occupancy_grid():
    occupancy_grids = np.zeros((520, 220), dtype=int)
    # set the occupancy grid according to knowledge about the levine hall
    occupancy_grids[:, 0:2] = 2  # right wall
    occupancy_grids[:, 197:] = 2  # left wall
    occupancy_grids[0:6, :] = 2  # bottom wall
    occupancy_grids[491:, :] = 2  # upper wall
    occupancy_grids[24:473, 22:178] = 2  # inner section
    return occupancy_grids


# class defining the Node object in the RRT tree.
# More fields could be added to this struct if more info needed.
# You can choose to use this or not
class Node:
    def __init__(self, x_pos, y_pos, parent, is_root):
        self.x = x_pos
        self.y = y_pos
        self.cost = None
        self.parent = parent  # index of parent node
        self.is_root = is_root


class RRT:
    def __init__(self, obs, rrt_star):

        self.rrt_star = rrt_star
        self.x_curr = obs[0]
        self.y_curr = obs[1]
        self.theta_curr = obs[2] % (2 * math.pi)
        self.start = Node(self.x_curr, self.y_curr, None, True)
        self.start.cost = 0.0
        self.goal_pt = self.calc_goal_pt()
        self.search_area = []

    # convert global frame to grid frame
    # grid size: 0.05m * 0.05m
    def convert_frame(self, x_global, y_global, x_off=14.50, y_off=0.70):
        x_grid = x_global + x_off
        y_grid = y_global + y_off
        x_grid_int = int(round(x_grid / 0.05))
        y_grid_int = int(round(y_grid / 0.05))
        if x_grid_int > 100000:
            x_grid_int = 0
        if y_grid_int > 100000:
            y_grid_int = 0
        return [x_grid_int, y_grid_int]

    # The scan callback, update your occupancy grid and the goal_point here
    # each point scanned results in a square of 0.6m * 0.6m blocked around it
    # Args:
    #    obs: observation of the space
    #    goal_pts: set goal points
    # Returns:
    #
    #    @njit(fastmath=False, cache=True)
    def update_grids(self, obs):
        global occupancy_grids
        global laser_scan_obst
        scan_msg = np.array(obs[3])
        self.goal_pt = self.calc_goal_pt()
        angle_min = -3.14159
        angle_increment = 0.00582316
        # angle_max = 3.14159
        rear_to_lidar = 0.29275
        self.x_curr = obs[0]
        self.y_curr = obs[1]
        self.theta_curr = obs[2]
        x_lidar = self.x_curr + rear_to_lidar * math.cos(self.theta_curr)
        y_lidar = self.y_curr + rear_to_lidar * math.sin(self.theta_curr)

        # update occupancy grid = 1 for current car
        self.update_curr_car_occupancy_grid(self.x_curr, self.y_curr, self.theta_curr)

        # update occupancy grid based on current car's laser scan.
        # actual obstacles are set to 2. only set an index to 2 if the
        # given index is not set to 1 already
        for i in range(scan_msg.shape[0]):
            # only update 90 degrees section in front of car
            if (
                angle_min + angle_increment * i >= -0.78540 and angle_min + angle_increment * i <= 0.78540
            ):
                if (not math.isinf(scan_msg[i])) and (not math.isnan(scan_msg[i])):
                    distance = scan_msg[i]
                    local_angle = angle_min + angle_increment * i
                    global_angle = local_angle + self.theta_curr
                    x_obstacle = x_lidar + distance * math.cos(global_angle)
                    y_obstacle = y_lidar + distance * math.sin(global_angle)
                    grid_coordinates = self.convert_frame(x_obstacle, y_obstacle)
                    for j in range(
                        max(grid_coordinates[0] - 4, 0),
                        min(grid_coordinates[0] + 5, occupancy_grids.shape[0]),
                    ):
                        for k in range(
                            max(grid_coordinates[1] - 4, 0),
                            min(grid_coordinates[1] + 5, occupancy_grids.shape[1]),
                        ):
                            if occupancy_grids[j, k] != 1:
                                occupancy_grids[j, k] = 2
                                laser_scan_obst += [x_obstacle, y_obstacle]  # used for rendering

    # update occupancy grid = 1 for current car
    # also used for updating occupancy grid for current car path
    def update_curr_car_occupancy_grid(self, x, y, theta):
        global occupancy_grids
        global laser_scan_obst
        for i in range(-5, 12):
            for j in range(-5, 6):
                x_car = (
                    x
                    + i * 0.05 * math.cos(theta)
                    + j * 0.05 * math.sin(theta)
                )
                y_car = (
                    y
                    + i * 0.05 * math.sin(theta)
                    + j * 0.05 * math.cos(theta)
                )
                laser_scan_obst += [x_car, y_car]  # used for rendering
                grid_coordinates = self.convert_frame(x_car, y_car)
                occupancy_grids[grid_coordinates[0], grid_coordinates[1]] = 1

    def calc_goal_pt(self):
        # calculate goal points

        if self.x_curr <= 6.00 and self.y_curr <= 2.34:
            goal_point = [self.x_curr + 7.50, -0.145]
        elif self.x_curr > 6.00 and self.y_curr <= 6.15:
            goal_point = [9.575, self.y_curr + 2.30]
        elif self.x_curr >= -11.26 and self.y_curr > 6.15:
            goal_point = [self.x_curr - 2.30, 8.65]
        elif self.x_curr < -11.26 and self.y_curr > 2.34:
            goal_point = [-13.79, self.y_curr - 2.30]
        return goal_point

    # The RRT main loop happens here
    # Args:
    #
    # Returns:
    #    @njit(fastmath=False, cache=True)
    def find_path(self):
        # goal_coordinate = self.convert_frame(self.goal_pt[0], self.goal_pt[1])
        # if occupancy_grids[goal_coordinate[0], goal_coordinate[1]] != 0:
        self.start = Node(self.x_curr, self.y_curr, None, True)
        self.start.cost = 0.0
        tree = [self.start]  # tree as a list
        path = []  # vector to store the final path
        # each loop creates a new sample in the space, generate up to MAX_ITERATION samples due to on-board computation constraints
        for i in range(MAX_ITER):
            sampled_pt = self.sample()  # sample the free space
            nearest_node_idx = self.nearest(
                tree, sampled_pt
            )  # get the tree's nearest point
            new_node = self.steer(
                tree[nearest_node_idx], sampled_pt
            )  # steer the tree toward the sampled point, get new point
            new_node.parent = nearest_node_idx  # set the parent of the new point
            if (
                self.check_collision(tree[nearest_node_idx], new_node) is False
            ):  # collision checking for connecting the new point to the tree
                # if algorithm RRT* star is chosen, the block in the if statement is performed
                if self.rrt_star is True:
                    near_set = self.near(
                        tree, new_node
                    )  # set of points in the neighborhood of the new point
                    # find the points in the neighborhood through which minimum cost can be obtained to reach the new point
                    min_cost = self.cost(tree, tree[nearest_node_idx]) + self.line_cost(
                        new_node, tree[nearest_node_idx]
                    )
                    for j in range(len(near_set)):
                        if (
                            self.check_collision(tree[near_set[j]], new_node) is False
                            and (
                                self.cost(tree, tree[near_set[j]])
                                + self.line_cost(new_node, tree[near_set[j]])
                            )
                            < min_cost
                        ):
                            new_node.parent = near_set[j]
                            min_cost = self.cost(
                                tree, tree[near_set[j]]
                            ) + self.line_cost(new_node, tree[near_set[j]])
                    # rewire the tree to get lower cost for other points in the neighborhood
                    new_node.cost = min_cost
                    for j in range(len(near_set)):
                        if self.check_collision(
                            tree[near_set[j]], new_node
                        ) is False and (
                            min_cost + self.line_cost(new_node, tree[near_set[j]])
                            < self.cost(tree, tree[near_set[j]])
                        ):
                            tree[near_set[j]].parent = len(tree)
                            tree[near_set[j]].cost = min_cost + self.line_cost(
                                new_node, tree[near_set[j]]
                            )
                tree.append(new_node)  # add the new point to the tree
                if self.is_goal(new_node):  # check if the goal point is reached
                    path = self.backtrack_path(
                        tree, new_node
                    )  # return the generated path
                    break
                # print(i)
        return np.array(path)

    def sample(self):
        # This method returns a sampled point from the free space
        # You should restrict so that it only samples a small region
        # of interest around the car's current position
        # Args:
        #
        # Returns:
        #     sampled_point ([x, y] ): the sampled point in free space
        # x_dist = np.random.uniform(self.x_curr, self.x_curr + 5.0)
        # y_dist = np.random.uniform(self.y_curr - 2.50, self.y_curr + 2.50)
        # rotm = np.array(
        #     [
        #         [np.cos(self.theta_curr), -np.sin(self.theta_curr), 0],
        #         [np.sin(self.theta_curr), np.cos(self.theta_curr), 0],
        #         [0, 0, 1],
        #     ]
        # )
        # trans1 = np.array([[1, 0, -self.x_curr], [0, 1, -self.y_curr], [0, 0, 1]])
        # trans2 = np.array([[1, 0, self.x_curr], [0, 1, self.y_curr], [0, 0, 1]])
        # sample = np.dot(
        #     trans2, np.dot(rotm, np.dot(trans1, np.array([x_dist, y_dist, 1])))
        # )
        if self.x_curr <= 6.00 and self.y_curr <= 2.34:
            x_limit_top = self.x_curr + 8.6
            x_limit_bot = self.x_curr
            y_limit_left = 0.37
            y_limit_right = -0.66
        elif self.x_curr > 6.00 and self.y_curr <= 6.15:
            x_limit_top = 10.03
            x_limit_bot = 8.50
            y_limit_left = self.y_curr + 4.50
            y_limit_right = self.y_curr
        elif self.x_curr >= -11.26 and self.y_curr > 6.15:
            x_limit_top = self.x_curr
            x_limit_bot = self.x_curr - 2.50
            y_limit_left = 9.15
            y_limit_right = 8.15
        elif self.x_curr < -11.26 and self.y_curr > 2.34:
            x_limit_top = -13.32
            x_limit_bot = -14.26
            y_limit_left = self.y_curr
            y_limit_right = self.y_curr - 2.50
        x_dist = np.random.uniform(x_limit_bot, x_limit_top)
        y_dist = np.random.uniform(y_limit_right, y_limit_left)
        self.search_area = [x_limit_top, y_limit_left,
                            x_limit_top, y_limit_right,
                            x_limit_bot, y_limit_right,
                            x_limit_bot, y_limit_left,
                            x_limit_top, y_limit_left]
        return [x_dist, y_dist]

    # This method returns the nearest node on the tree to the sampled point
    # Args:
    #     tree (list of node): the current RRT tree
    #     sampled_point ([x, y]): the sampled point in free space
    # Returns:
    #     nearest_node (int): index of nearest node on the tree
    #    @njit(fastmath=False, cache=True)
    def nearest(self, tree, sampled_pt):
        nearest_node = 0
        min_dist = (tree[0].x - sampled_pt[0]) ** 2 + (tree[0].y - sampled_pt[1]) ** 2
        for i in range(len(tree)):
            newmin = (tree[i].x - sampled_pt[0]) ** 2 + (tree[i].y - sampled_pt[1]) ** 2
            if newmin < min_dist:
                min_dist = newmin
                nearest_node = i
        return nearest_node

    # The function steer:(x,y)->z returns a point such that z is “closer”
    # to y than x is. The point z returned by the function steer will be
    # such that z minimizes ||z−y|| while at the same time maintaining
    # ||z−x|| <= max_expansion_dist, for a prespecified max_expansion_dist > 0

    # basically, expand the tree towards the sample point (within a max dist)

    # Args:
    #    nearest_node (Node): nearest node on the tree to the sampled point
    #    sampled_point ([x, y]): the sampled point in free space
    # Returns:
    #    new_node (Node): new node created from steering
    def steer(self, nearest_node, sampled_pt):
        act_dist = math.sqrt(
            (nearest_node.x - sampled_pt[0]) ** 2
            + (nearest_node.y - sampled_pt[1]) ** 2
        )
        x = nearest_node.x + STEER_LENGTH / act_dist * (sampled_pt[0] - nearest_node.x)
        y = nearest_node.y + STEER_LENGTH / act_dist * (sampled_pt[1] - nearest_node.y)
        new_node = Node(x, y, None, False)
        return new_node

    # This method returns a boolean indicating if the path between the
    # nearest node and the new node created from steering is collision free
    # Args:
    #    nearest_node (Node): nearest node on the tree to the sampled point
    #    new_node (Node): new node created from steering
    # Returns:
    #    collision (bool): true if in collision, false otherwise
    #    @njit(fastmath=False, cache=True)
    def check_collision(self, nearest_node, new_node):
        global occupancy_grids
        collision = False
        for i in range(10):
            coordinate = self.convert_frame(
                nearest_node.x + i * 0.01 * (new_node.x - nearest_node.x),
                nearest_node.y + i * 0.01 * (new_node.y - nearest_node.y),
            )
            x_coordinate = nearest_node.x + i * 0.01 * (new_node.x - nearest_node.x)
            y_coordinate = nearest_node.y + i * 0.01 * (new_node.y - nearest_node.y)
            alpha = math.atan2(y_coordinate - self.y_curr, x_coordinate - self.x_curr)
            x_diff = math.sqrt(
                (y_coordinate - self.y_curr) ** 2 + (x_coordinate - self.x_curr) ** 2
            ) * math.cos(alpha - self.theta_curr)
            y_diff = math.sqrt(
                (y_coordinate - self.y_curr) ** 2 + (x_coordinate - self.x_curr) ** 2
            ) * math.sin(alpha - self.theta_curr)
            if (
                x_diff >= -0.15
                and x_diff <= 0.45
                and y_diff >= -0.15
                and y_diff <= 0.15
            ):
                continue
            elif occupancy_grids[coordinate[0], coordinate[1]] != 0:
                collision = True
                break
        return collision

    # This method checks if the latest node added to the tree is close
    # enough (defined by goal_threshold) to the goal so we can terminate
    # the search and find a path
    # Args:
    #   latest_added_node (Node): latest addition to the tree
    # Returns:
    #   close_enough (bool): true if node close enough to the goal
    def is_goal(self, node):
        close_enough = False
        if (
            math.sqrt((node.x - self.goal_pt[0]) ** 2 + (node.y - self.goal_pt[1]) ** 2)
            <= TERMINATE_LENGTH
        ):
            close_enough = True
        return close_enough

    # This method traverses the tree from the node that has been determined
    # as goal
    # Args:
    #   latest_added_node (Node): latest addition to the tree that has been
    #      determined to be close enough to the goal
    # Returns:
    #   path (list of nodes): the vector that represents the order of
    #      of the nodes traversed as the found path
    #    @njit(fastmath=False, cache=True)
    def backtrack_path(self, tree, latest_node):
        found_path = []
        next_node = tree[latest_node.parent]
        while next_node.is_root is False:
            # theta = self.calculate_theta(next_node.x,
            #                              next_node.y,
            #                              tree[next_node.parent].x,
            #                              tree[next_node.parent].y)
            found_path.append([next_node.x, next_node.y])
            next_node = tree[next_node.parent]
        found_path.append([tree[0].x, tree[0].y])
        return found_path

    """
    This method calculates the estimated angle that the car will be at for a 
    given segment in the path. Used to 
    Args: 
        (x1, y1) - global coords of first point
        (x2, y2) - global coords of second point
    Returns:
        theta - angle between two points (in radians)
    """

    def calculate_theta(self, x1, y1, x2, y2):
        return np.arctan2((y2 - y1), (x2 - x1))

    # This method returns the set of Nodes in the neighborhood of a
    # node.
    # Args:
    #   tree (list of nodes): the current tree
    #   node (Node): the node to find the neighborhood for
    # Returns:
    #   neighborhood (list of index): the index of the nodes in the neighborhood

    def near(self, tree, new_node):
        neighborhood = []
        for i in range(len(tree)):
            if (
                math.sqrt((tree[i].x - new_node.x) ** 2 + (tree[i].y - new_node.y) ** 2)
                < ETA
            ):
                neighborhood.append(i)
        return neighborhood

    # This method returns the cost associated with a node
    # Args:
    #    tree (list of nodes): the current tree
    #    node (Node): the node the cost is calculated for
    # Returns:
    #    cost (double): the cost value associated with the node
    def cost(self, tree, new_node):
        cost = 0
        cost = new_node.cost
        return cost

    # This method returns the cost of the straight line path between two nodes
    # Args:
    #    node1 (Node): the Node at one end of the path
    #    node2 (Node): the Node at the other end of the path
    # Returns:
    #    cost (double): the cost value associated with the path
    def line_cost(self, node1, node2):
        cost = 0
        cost = math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        return cost


def get_goal_point(goal_car_curr_pos):
    r = 2
    return [
        goal_car_curr_pos[0] + r * math.cos(goal_car_curr_pos[2]),
        goal_car_curr_pos[1] + r * math.sin(goal_car_curr_pos[2]),
    ]


if __name__ == "__main__":

    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.60,
        "vgain": 0.90338203837889,
    }

    # start up the master node server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    laptime = 0.0
    start = time.time()

    pool = ThreadPool(processes=2)

    while True:

        # receive state of cars from car node
        message = socket.recv()

        # deserialize json messages
        obs = json.loads(message)

        rrt_1 = RRT(obs[0], True)
        rrt_2 = RRT(obs[1], True)

        occupancy_grids = get_occupancy_grid()
        laser_scan_obst = get_laser_scan_obstacles()

        rrt_2.update_grids(obs[1])
        rrt_1.update_grids(obs[0])

        trajectory_1 = None
        trajectory_2 = None

        trajectory_1 = rrt_1.find_path()
        trajectory_2 = rrt_2.find_path()
        
#        trajectory_1 = np.array([[ 2.22374142e+00, -3.79893641e-03],
#         [ 1.63769408e+00,  3.30800844e-02],
#         [ 1.07068519e+00, -2.75612004e-02],
#         [ 6.65643282e-01, -1.51545069e-01],
#         [ 7.94652784e-02, -2.07459981e-01],
#         [-2.18877129e-01, -2.38952966e-01],
#         [-8.13397358e-01, -2.63592846e-01],
#         [-1.38573766e+00, -3.71340052e-01],
#         [-1.96368019e+00, -5.18964665e-01],
#         [-2.37448779e+00, -4.63992759e-01],
#         [-2.96652200e+00, -5.43226038e-01],
#         [-3.54476297e+00, -3.99039045e-01],
#         [-3.86796226e+00, -2.06865596e-01],
#         [-4.45580315e+00, -1.74883870e-01],
#         [-4.99744705e+00, -7.54591859e-08]])
#        trajectory_2 = np.array([[ 4.09344206e+00, -1.75239803e-01],
#         [ 3.79392378e+00, -1.58245731e-01],
#         [ 3.20005712e+00, -1.61349176e-01],
#         [ 2.61035169e+00, -2.22532707e-01],
#         [ 2.07353467e+00, -1.28643557e-01],
#         [ 1.77660251e+00, -8.58499525e-02],
#         [ 1.19250716e+00,  2.99515409e-02],
#         [ 5.94109720e-01,  6.82018614e-02],
#         [-5.42762556e-03,  5.20939999e-02],
#         [-6.03895385e-01,  6.63778575e-02],
#         [-1.20257923e+00,  7.74802401e-02],
#         [-1.80084216e+00,  6.52300186e-02],
#         [-2.39992338e+00,  3.78900126e-02],
#         [-2.99872352e+00,  1.88647965e-08]])
        
        car1_vel = 2.5
        car2_vel = 2.5

        msg = {
            "goal_pts": [rrt_1.goal_pt, rrt_2.goal_pt],
            "trajectory": [trajectory_1.tolist(), trajectory_2.tolist()],
            "laser_scan_obst": laser_scan_obst,
            "search_area": [rrt_1.search_area, rrt_2.search_area]
        }

        if len(trajectory_1) > 1 and len(trajectory_2) > 1:
            
            # smooth the trajectories
            # set car 1 to have a velocity of 2.0 m/s and car 2 to a velocity of 1.0 m/s
            smoothedTraj_1 = TrajectorySmoothing(np.flip(trajectory_1,0), car1_vel)
            smoothedTraj_2 = TrajectorySmoothing(np.flip(trajectory_2,0), car2_vel)
            
            # calc max time along path between two smoothed trajectories
            max_time = max(smoothedTraj_1.total_time, smoothedTraj_2.total_time)
            time_step = 0.02
                        
            # calculate s valued points along path
            pathTraj_1 = PathTrajectory(smoothedTraj_1.get_total_path(time_step), 
                                        np.array([[0,0],[smoothedTraj_1.total_time, 1]]), smoothedTraj_1.total_time)
            pathTraj_2 = PathTrajectory(smoothedTraj_2.get_total_path(time_step), 
                                        np.array([[0,0],[smoothedTraj_2.total_time, 1]]), smoothedTraj_2.total_time)

            velocityTuner = VelocityTuner(pathTraj_1, pathTraj_2, max_time, time_step)
            velocityTuner.tune_velocities()
            pathTraj_2 = velocityTuner.pathTraj_2
            pathTrajvelcoity_1, pathTrajvelocity_2 = velocityTuner.calculate_trajectory(car1_vel)

            # msg["time_s"] = [pathTraj_1.time_s.tolist(), pathTraj_2.time_s.tolist()]
            # msg["path_length"] = [pathTraj_1.total_path_length, pathTraj_2.total_path_length]
            msg["trajectory_velocity"] = [pathTrajvelcoity_1.tolist(), pathTrajvelocity_2.tolist()]
            msg["trajectory"] = [pathTraj_1.path.tolist(), pathTraj_2.path.tolist()]

        # send calculate path back to car
        socket.send_string(
            json.dumps(msg)
        )

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)
