import gym
import numpy as np
import yaml
from argparse import Namespace
import math

# RRT params
MAX_ITER = 100
STEER_LENGTH = 0.3
TERMINATE_LENGTH = 0.1
ETA = 0.6
X_SEARCH_LIM = 10.0
Y_SEARCH_LIM = 10.0
# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuit:
    def __init__(self, path, wb):
        self.wheelbase = wb
        self.path = path
        self.max_reacquire = 20.
        
    def get_current_waypoint(self, lookahead_distance, position, theta):
        wpts = np.vstack((self.path[0], self.path[1])).T
        nearest_point, nearest_dist, t, i, = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=False)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            current_waypoint[0:2] = wpts[i2,:]
            current_waypoint[2] = 1.0 #speed
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i,:], 1.0)
        else:
            return None
        
    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point = self.get_current_waypoint(lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            return 4.0, 0.0
        
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed
        
        return speed, steering_angle
        
# class defining the Node object in the RRT tree.
# More fields could be added to this struct if more info needed.
# You can choose to use this or not
class Node:
    def __init__(self, x_pos, y_pos, parent, is_root):
        self.x = x_pos
        self.y = y_pos
        self.cost = None
        self.parent = parent    # index of parent node
        self.is_root = is_root
        
class RRT:
    def __init__(self, obs, goal_pt, rrt_star):
        
        self.rrt_star = rrt_star
        self.x_curr = obs['poses_x'][0]
        self.y_curr = obs['poses_y'][0]
        self.theta_curr = obs['poses_theta'][0]%(2*math.pi)
        self.oppx_curr = obs['poses_x'][1]
        self.oppy_curr = obs['poses_y'][1]
        self.opptheta_curr = obs['poses_theta'][1]%(2*math.pi)
        self.start = Node(self.x_curr, self.y_curr, None, True)
        self.goal_pt = goal_pt
        
    # The RRT main loop happens here
    # Args:
    # 
    # Returns:
    def find_path(self):
        tree = [self.start] # tree as a list
        path = []  # vector to store the final path
        # each loop creates a new sample in the space, generate up to MAX_ITERATION samples due to on-board computation constraints
        for i in range(0,MAX_ITER):
            sampled_pt = self.sample() # sample the free space
            nearest_node_idx = self.nearest(tree, sampled_pt) # get the tree's nearest point
            new_node = self.steer(tree[nearest_node_idx], sampled_pt) # steer the tree toward the sampled point, get new point
            new_node.parent = nearest_node_idx # set the parent of the new point
            if (self.check_collision(tree[nearest_node_idx], new_node) == False): # collision checking for connecting the new point to the tree
                # if algorithm RRT* star is chosen, the block in the if statement is performed
                if(self.rrt_star == True):
                    near_set = self.near(tree, new_node) # set of points in the neighborhood of the new point
                    # find the points in the neighborhood through which minimum cost can be obtained to reach the new point
                    min_cost = self.cost(tree, tree[nearest_node_idx]) + self.line_cost(new_node, tree[nearest_node_idx])
                    for j in range(len(near_set)):
                        if (self.check_collision(tree[near_set[j]]) == False and (self.cost(tree, tree[near_set[j]]) + self.line_cost(new_node, tree[near_set[j]])) < min_cost):
                            new_node.parent = near_set[j]
                            min_cost = self.cost(tree, tree[near_set[j]]) + self.line_cost(new_node, tree[near_set[j]])
                    # rewire the tree to get lower cost for other points in the neighborhood
                    new_node.cost = min_cost
                    for j in range(len(near_set)):
                        if (self.check_collision(tree[near_set[j]], new_node) == False 
                            and (min_cost + self.line_cost(new_node, tree[near_set[j]]) < 
                                 self.cost(tree, tree[near_set[j]]))):
                            tree[near_set[j]].parent = len(tree)
                            tree[near_set[j]].cost = min_cost + self.line_cost(new_node, tree[near_set[j]])
                tree.append(new_node) # add the new point to the tree
                if self.is_goal(new_node): # check if the goal point is reached
                    path = self.backtrack_path(tree, new_node) # return the generated path
                    break
        return np.array(path)
            
    
    def sample(self):
            # This method returns a sampled point from the free space
            # You should restrict so that it only samples a small region
            # of interest around the car's current position
            # Args:
            # 
            # Returns:
            #     sampled_point ([x, y] ): the sampled point in free space
        x_dist = np.random.uniform(self.x_curr, self.x_curr + 2.5)
        y_dist = np.random.uniform(self.y_curr - 0.5, self.y_curr + 0.5)
        rotm = np.array([[np.cos(self.theta_curr), -np.sin(self.theta_curr), 0],
                          [np.sin(self.theta_curr), np.cos(self.theta_curr), 0],
                          [0, 0, 1]])
        trans1 = np.array([[1, 0, -self.x_curr],
                           [0, 1, -self.y_curr],
                           [0, 0, 1]])
        trans2 = np.array([[1, 0, self.x_curr],
                           [0, 1, self.y_curr],
                           [0, 0, 1]])
        sample = np.dot(trans2, np.dot(rotm, np.dot(trans1, np.array([x_dist, y_dist, 1]))))
        return [sample[0], sample[1]]
    
    # This method returns the nearest node on the tree to the sampled point
    # Args:
    #     tree (list of node): the current RRT tree
    #     sampled_point ([x, y]): the sampled point in free space
    # Returns:
    #     nearest_node (int): index of nearest node on the tree
    def nearest(self, tree, sampled_pt):
        nearest_node = 0
        min_dist = (tree[0].x - sampled_pt[0])**2 + (tree[0].y - sampled_pt[1])**2
        for i in range(len(tree)):
            newmin = (tree[i].x - sampled_pt[0])**2 + (tree[i].y - sampled_pt[1])**2
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
        act_dist = math.sqrt((nearest_node.x - sampled_pt[0])**2 + (nearest_node.y - sampled_pt[1])**2)
        x = nearest_node.x + STEER_LENGTH / act_dist * (sampled_pt[0] - nearest_node.x)
        y = nearest_node.y + STEER_LENGTH / act_dist * (sampled_pt[1] - nearest_node.y)
        new_node = Node(x,y,None, False)
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
        A = [nearest_node.x, nearest_node.y]
        B = [new_node.x, new_node.y]
        D = None
        car_pts_arr = self.get_opp_car_pts()
        for i in range(0,4):
            C = car_pts_arr[i]
            if i == 3:
                D = car_pts_arr[0]
            else:
                D = car_pts_arr[i+1]
            collision = self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)
        return collision
        
    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[0]) * (C[0]-A[0])
    
    def get_opp_car_pts(self):
        car_pts_arr = np.array([[self.oppx_curr+CAR_LENGTH, self.oppy_curr+CAR_WIDTH],
                                [self.oppx_curr-CAR_LENGTH, self.oppy_curr+CAR_WIDTH],
                                [self.oppx_curr-CAR_LENGTH, self.oppy_curr-CAR_WIDTH],
                                [self.oppx_curr+CAR_LENGTH, self.oppy_curr-CAR_WIDTH]])
        rotm = np.array([[np.cos(self.theta_curr), -np.sin(self.theta_curr)],
                          [np.sin(self.theta_curr), np.cos(self.theta_curr)]])
        new_car_pts_arr = []
        for i in range(0,4):
            new_car_pts_arr.append(np.dot(rotm, car_pts_arr[i]))
        
        return new_car_pts_arr    

    # This method checks if the latest node added to the tree is close
    # enough (defined by goal_threshold) to the goal so we can terminate
    # the search and find a path
    # Args:
    #   latest_added_node (Node): latest addition to the tree
    # Returns:
    #   close_enough (bool): true if node close enough to the goal
    def is_goal(self,node):
        close_enough = False
        if (math.sqrt((node.x-self.goal_pt[0])**2 + (node.y-self.goal_pt[1])**2) <= TERMINATE_LENGTH):
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
        while(next_node.is_root == False):
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
            if (math.sqrt((tree[i].x - new_node.x)**2 + (tree[i].y - new_node.y)**2) < ETA):
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
        cost = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
        return cost
        
        
    
def get_goal_point(goal_car_curr_pos):
    r = 2
    return [goal_car_curr_pos[0]+r*math.cos(goal_car_curr_pos[2]),
            goal_car_curr_pos[1]+r*math.sin(goal_car_curr_pos[2])]
#    return [10,0]
    


if __name__ == '__main__':
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}

    with open('map0_example.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    racecar_env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)
    obs, step_reward, done, info = racecar_env.reset(np.array([[conf.sx, conf.sy, conf.stheta],
                                                               [conf.oppx, conf.oppy, conf.opptheta]]))
    racecar_env.render()
    
    lap_time = 0.
    steer = 0.00
    speed = 1.0
    
    goal_pt = None
    
    while not done:
        
        goal_pt = [obs['poses_x'][1], obs['poses_y'][1]]
        rrt = RRT(obs, goal_pt, False)
        path = rrt.find_path()
        obs, step_reward, done, info = racecar_env.step(np.array([[steer, speed],
                                                                  [0.0, 0.5]]))
        if len(path) == 0:
            continue
        planner = PurePursuit(path, 0.17145+0.15875)
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                    work['tlad'], work['vgain'])
        print(speed, steer)
        
        racecar_env.update_path(path)
        
        lap_time += step_reward
        racecar_env.render(mode='human')