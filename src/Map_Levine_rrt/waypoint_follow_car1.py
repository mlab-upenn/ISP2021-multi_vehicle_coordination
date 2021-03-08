import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
import zmq
import json

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


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - 2.0 * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
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
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - 2.0 * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
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


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
    # speed = lookahead_point[2]
    # real_distance = math.sqrt(
    #     (lookahead_point[0] - position[0]) * (lookahead_point[0] - position[0])
    #     + (lookahead_point[1] - position[1]) * (lookahead_point[1] - position[1])
    # )
    # lookahead_angle = np.arctan2(
    #     lookahead_point[1] - position[1], lookahead_point[0] - position[0]
    # )
    # del_y = real_distance * np.sin(lookahead_angle - pose_theta)
    # steering_angle = 2.00 * del_y / (real_distance * real_distance)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, wb, follow_master, conf=None):
        self.wheelbase = wb
        if follow_master:
            self.conf = conf
            self.load_waypoints(conf)
        self.max_reacquire = 20.0
        self.follow_master = follow_master

    def load_waypoints(self, conf):
        # load waypoints
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        )

    def update_paths(self, trajectory):
        if np.size(trajectory) != 0:
            self.waypoints = trajectory

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        if self.follow_master:
            wpts = np.vstack(
                (
                    waypoints[:, self.conf.wpt_xind],
                    waypoints[:, self.conf.wpt_yind],
                )
            ).T
        else:
            wpts = np.vstack(
                (
                    np.flip(waypoints[:, 0]),
                    np.flip(waypoints[:, 1]),
                )
            ).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 == None:
                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = 2.0
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], 2.0)
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        if self.follow_master == True:
            lookahead_point = self._get_current_waypoint(
                self.waypoints, lookahead_distance, position, pose_theta
            )
        else:
            lookahead_point = np.empty((3,))
            for i in range(self.waypoints.shape[0] - 1, 0, -1):
                if (
                    math.sqrt(
                        (self.waypoints[i, 0] - position[0]) ** 2
                        + (self.waypoints[i, 1] - position[1]) ** 2
                    )
                    >= lookahead_distance
                ):
                    lookahead_point[0] = self.waypoints[i, 0]
                    lookahead_point[1] = self.waypoints[i, 1]
                    lookahead_point[2] = 4.0
                    break
        # print("waypoints matrix", self.waypoints)
        # print("lookahead", lookahead_point)
        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase
        )
        # if abs(steering_angle) < 0.5236:
        speed = vgain * speed
        # else:
        #     speed = 0.30 * vgain * speed

        return speed, steering_angle


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
        "tlad": 0.70,
        "vgain": 0.90338203837889,
    }

    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to hello world server…")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    with open("config_example_map.yaml") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make(
        "f110_gym:f110-v0", map=conf.map_path, map_ext=conf.map_ext, num_agents=2
    )
    obs, step_reward, done, info = env.reset(
        np.array(
            [[conf.sx, conf.sy, conf.stheta], [conf.oppx, conf.oppy, conf.opptheta]]
        )
    )
    env.render()
    planner_1 = PurePursuitPlanner(0.17145 + 0.15875, False, conf)
    planner_2 = PurePursuitPlanner(0.17145 + 0.15875, True, conf)

    laptime = 0.0
    start = time.time()

    while not done:

        # serialize the observation data into JSON
        socket.send_string(
            json.dumps(
                {
                    "x_curr": obs["poses_x"][0],
                    "y_curr": obs["poses_y"][0],
                    "theta_curr": obs["poses_theta"][0],
                    "oppx_curr": obs["poses_x"][1],
                    "oppy_curr": obs["poses_y"][1],
                    "opptheta_curr": obs["poses_theta"][1],
                    "scans": (obs["scans"][0]).tolist(),
                }
            )
        )

        # receive the path from the master node
        message = socket.recv()

        # deserialize it
        trajectory_1 = json.loads(message)

        if np.size(trajectory_1) != 0:
            env.update_path(trajectory_1)
        # # print("position", obs["poses_x"][0], obs["poses_y"][0])
        # # print("goal_point", goal_point)
        # # print("generate path", trajectory_1)

        # convert 1d list to 2d np array
        nptraj_1 = np.array(trajectory_1)
        nptraj_1 = np.reshape(nptraj_1, (-1, 2))

        planner_1.update_paths(nptraj_1)
        # print(111)
        speed_1, steer_1 = planner_1.plan(
            obs["poses_x"][0],
            obs["poses_y"][0],
            obs["poses_theta"][0],
            work["tlad"],
            work["vgain"],
        )
        # print(222)
        speed_2, steer_2 = planner_2.plan(
            obs["poses_x"][1],
            obs["poses_y"][1],
            obs["poses_theta"][1],
            work["tlad"],
            work["vgain"],
        )
        obs, step_reward, done, info = env.step(
            np.array([[steer_1, speed_1], [steer_2, speed_2]])
        )
        laptime += step_reward
        env.render(mode="human")
    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)
