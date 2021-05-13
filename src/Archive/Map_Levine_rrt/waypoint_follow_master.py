import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
import copy
import json
import zmq

from numba import njit

"""
Planner Helpers
"""

# RRT params
MAX_ITER = 100
STEER_LENGTH = 0.3
TERMINATE_LENGTH = 0.2
ETA = 0.6
X_SEARCH_LIM = 10.0
Y_SEARCH_LIM = 10.0
# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

# -3.14159, 3.14159, 0.00582316

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
    def __init__(self, obs, goal_pt, rrt_star):

        self.rrt_star = rrt_star
        self.x_curr = obs["x_curr"]
        self.y_curr = obs["y_curr"]
        self.theta_curr = obs["theta_curr"] % (2 * math.pi)
        self.oppx_curr = obs["oppx_curr"]
        self.oppy_curr = obs["oppy_curr"]
        self.opptheta_curr = obs["opptheta_curr"] % (2 * math.pi)
        self.start = Node(self.x_curr, self.y_curr, None, True)
        self.start.cost = 0.0
        self.goal_pt = goal_pt
        # create an occupancy grid
        self.occupancy_grids_prior = np.ones((500, 200), dtype=bool)
        # set the occupancy grid according to knowledge about the levine hall
        # for i in range(0, 2):  # right wall
        #     for j in range(0, self.occupancy_grids_prior.shape[0]):
        #         self.occupancy_grids_prior[j, i] = False
        # for i in range(197, self.occupancy_grids_prior.shape[1]):  # left wall
        #     for j in range(0, self.occupancy_grids_prior.shape[0]):
        #         self.occupancy_grids_prior[j, i] = False
        # for i in range(0, self.occupancy_grids_prior.shape[1]):  # bot wall
        #     for j in range(0, 6):
        #         self.occupancy_grids_prior[j, i] = False
        # for i in range(0, self.occupancy_grids_prior.shape[1]):  # upper wall
        #     for j in range(491, self.occupancy_grids_prior.shape[0]):
        #         self.occupancy_grids_prior[j, i] = False
        # for i in range(22, 178):  # inner parts
        #     for j in range(24, 473):
        #         self.occupancy_grids_prior[j, i] = False
        self.occupancy_grids_prior[:, 0:2] = False
        self.occupancy_grids_prior[:, 197:] = False
        self.occupancy_grids_prior[0:6, :] = False
        self.occupancy_grids_prior[491:, :] = False
        self.occupancy_grids_prior[24:473, 22:178] = False

        self.occupancy_grids = self.occupancy_grids_prior

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
    def update_grids(self, obs, goal_pts):
        scan_msg = np.array(obs["scans"])
        self.goal_pt = goal_pts
        self.occupancy_grids = copy.deepcopy(self.occupancy_grids_prior)
        angle_min = -3.14159
        angle_increment = 0.00582316
        angle_max = 3.14159
        rear_to_lidar = 0.29275
        self.x_curr = obs["x_curr"]
        self.y_curr = obs["y_curr"]
        self.theta_curr = obs["theta_curr"]
        x_lidar = self.x_curr + rear_to_lidar * math.cos(self.theta_curr)
        y_lidar = self.y_curr + rear_to_lidar * math.sin(self.theta_curr)
        for i in range(scan_msg.shape[0]):
            if (
                angle_min + angle_increment * i >= -0.78540
                and angle_min + angle_increment * i <= 0.78540
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
                        min(grid_coordinates[0] + 5, self.occupancy_grids.shape[0]),
                    ):
                        for k in range(
                            max(grid_coordinates[1] - 4, 0),
                            min(grid_coordinates[1] + 5, self.occupancy_grids.shape[1]),
                        ):
                            self.occupancy_grids[j, k] = False

                    # self.occupancy_grids[
                    #     int(max(grid_coordinates[0] - 3, 0)) : int(
                    #         min(grid_coordinates[0] + 4, self.occupancy_grids.shape[0])
                    #     ),
                    #     int(max(grid_coordinates[1] - 3, 0)) : int(
                    #         min(grid_coordinates[1] + 4, self.occupancy_grids.shape[1])
                    #     ),
                    # ] = False

                # self.occupancy_grids[
                #     int(max(grid_coordinates[0] - 3, 0)) : int(
                #         min(grid_coordinates[0] + 4, self.occupancy_grids.shape[0])
                #     ),
                #     int(max(grid_coordinates[1] - 3, 0)) : int(
                #         min(grid_coordinates[1] + 4, self.occupancy_grids.shape[1])
                #     ),
                # ] = False

    # The RRT main loop happens here
    # Args:
    #
    # Returns:
    def find_path(self):
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
                self.check_collision(tree[nearest_node_idx], new_node) == False
            ):  # collision checking for connecting the new point to the tree
                # if algorithm RRT* star is chosen, the block in the if statement is performed
                if self.rrt_star == True:
                    near_set = self.near(
                        tree, new_node
                    )  # set of points in the neighborhood of the new point
                    # find the points in the neighborhood through which minimum cost can be obtained to reach the new point
                    min_cost = self.cost(tree, tree[nearest_node_idx]) + self.line_cost(
                        new_node, tree[nearest_node_idx]
                    )
                    for j in range(len(near_set)):
                        if (
                            self.check_collision(tree[near_set[j]], new_node) == False
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
                        ) == False and (
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
        if self.x_curr <= 7.00 and self.y_curr <= 2.34:
            x_limit_top = self.x_curr + 2.50
            x_limit_bot = self.x_curr
            y_limit_left = 0.37
            y_limit_right = -0.66
        elif self.x_curr > 7.00 and self.y_curr <= 6.15:
            x_limit_top = 10.03
            x_limit_bot = 9.12
            y_limit_left = self.y_curr + 2.50
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
        return [x_dist, y_dist]

    # This method returns the nearest node on the tree to the sampled point
    # Args:
    #     tree (list of node): the current RRT tree
    #     sampled_point ([x, y]): the sampled point in free space
    # Returns:
    #     nearest_node (int): index of nearest node on the tree
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
    def check_collision(self, nearest_node, new_node):
        collision = False
        for i in range(100):
            coordinate = self.convert_frame(
                nearest_node.x + i * 0.01 * (new_node.x - nearest_node.x),
                nearest_node.y + i * 0.01 * (new_node.y - nearest_node.y),
            )
            if self.occupancy_grids[coordinate[0], coordinate[1]] == False:
                collision = True
                break
        return collision

    def get_opp_car_pts(self):
        car_pts_arr = np.array(
            [
                [self.oppx_curr + CAR_LENGTH, self.oppy_curr + CAR_WIDTH],
                [self.oppx_curr - CAR_LENGTH, self.oppy_curr + CAR_WIDTH],
                [self.oppx_curr - CAR_LENGTH, self.oppy_curr - CAR_WIDTH],
                [self.oppx_curr + CAR_LENGTH, self.oppy_curr - CAR_WIDTH],
            ]
        )
        rotm = np.array(
            [
                [np.cos(self.theta_curr), -np.sin(self.theta_curr)],
                [np.sin(self.theta_curr), np.cos(self.theta_curr)],
            ]
        )
        new_car_pts_arr = []
        for i in range(0, 4):
            new_car_pts_arr.append(np.dot(rotm, car_pts_arr[i]))

        return new_car_pts_arr

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
    def backtrack_path(self, tree, latest_node):
        found_path = []
        next_node = tree[latest_node.parent]
        while next_node.is_root == False:
            found_path.append([next_node.x, next_node.y])
            next_node = tree[next_node.parent]
        found_path.append([tree[0].x, tree[0].y])
        return found_path

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


#    return [10,0]


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

    # receive state of cars from car node
    message = socket.recv()

    # deserialize json messages
    obs = json.loads(message)

    rrt_1 = RRT(obs, [0, 0], True)

    empty = [0, 0, 0, 0]

    socket.send_string(json.dumps(empty))

    while True:

        # receive state of cars from car node
        message = socket.recv()

        # deserialize json messages
        obs = json.loads(message)

        # calculate goal points
        if obs["x_curr"] <= 7.00 and obs["y_curr"] <= 2.34:
            goal_point = [obs["x_curr"] + 2.30, -0.145]
        elif obs["x_curr"] > 7.00 and obs["y_curr"] <= 6.15:
            goal_point = [9.575, obs["y_curr"] + 2.30]
        elif obs["x_curr"] >= -11.26 and obs["y_curr"] > 6.15:
            goal_point = [obs["x_curr"] - 2.30, 8.65]
        elif obs["x_curr"] < -11.26 and obs["y_curr"] > 2.34:
            goal_point = [-13.79, obs["y_curr"] - 2.30]

        rrt_1.update_grids(obs, goal_point)
        trajectory_1 = rrt_1.find_path()

        # send calculate path back to car
        socket.send_string(json.dumps(trajectory_1.tolist()))

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)
