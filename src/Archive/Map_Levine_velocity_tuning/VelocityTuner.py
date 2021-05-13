#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 21:03:56 2021

@author: meam520
"""
import cvxpy as cp
from PathTrajectory import PathTrajectory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class VelocityTuner:
    def __init__(self, pathTraj_1, pathTraj_2, max_time, time_step, car1_vel, car2_vel):
        self.pathTraj_1 = pathTraj_1
        self.pathTraj_2 = pathTraj_2
        self.max_time = int(max_time)  # there is no need to consider the max time, min time is enough
        self.time_step = time_step
        self.vec_plan_space = []
        self.min_obstacle = []
        self.max_obstacle = []
        self.car1_vel = car1_vel
        self.car2_vel = car2_vel

    def tune_velocities(self):
        """
        Tunes the velocities
        """

        c1_pos = self.pathTraj_1.constant_velocity_trajectory(self.time_step, self.max_time, self.car1_vel)

        # iterate thru time
        for i in range(0, int(self.max_time / self.time_step)):

            curr_c1_pos = c1_pos[i, :]

            found_collision = []

            c2_traj = self.pathTraj_2

            for t2 in np.arange(0.0, self.max_time, self.time_step):
                curr_c2_pos = c2_traj.update(t2)

                # calculate if the two cars collide
                if (np.linalg.norm(curr_c1_pos - curr_c2_pos) < 0.5):

                    # if they collide, then find the time parameterized value along the path
                    # that corresponds to this time
                    s = c2_traj.s

                    # append to array for plotting
                    self.vec_plan_space.append([i * self.time_step, s])
                    found_collision.append(s)
            if len(found_collision) > 0:
                found_collision = np.array(found_collision)
                self.min_obstacle.append(found_collision[0])
                self.max_obstacle.append(found_collision[-1])
            else:
                self.min_obstacle.append(None)
                self.max_obstacle.append(None)
            c2_traj.restore()

        self.optimize_velocity_path()

    def optimize_velocity_path(self):
        """
        Apply the optimization solver
        """
        n = int(self.max_time / self.time_step)
        v_targeted = cp.Variable(n - 1, nonneg=True)
        y = cp.Variable(n, boolean=True)
        v_true = cp.Variable(n, nonneg=True)
        s = cp.Variable(n, nonneg=True)
        # delta = cp.Variable(n - 1, boolean=True)
        # z = cp.Variable(n - 1, nonneg=True)

        M_s = float(self.max_time / self.time_step)
        # m_delta = - self.car2_vel
        # M_delta = self.car2_vel
        constraints = [
            v_true[0] == 0.0,
            s[0] == 0.0,
            s <= np.ones(n),
            v_true <= np.full(n, self.car2_vel),
        ]
        for i in range(n - 1):
            constraints += [v_targeted[i] <= self.car2_vel]
            constraints += [s[i + 1] == s[i] + v_true[i] * self.time_step / self.pathTraj_2.total_path_length]
            constraints += [v_true[i + 1] == v_true[i] + 2.0 * 9.51 /
                            20.0 * (v_targeted[i] - v_true[i]) * self.time_step]
            constraints += [v_targeted[i] >= v_true[i]]  # use this if the hybrid approach does not work
            # constraints += [v_true[i + 1] == v_true[i] + 2.0 * 9.51 /
            #                 5.0 * (v_targeted[i] - v_true[i]) * self.time_step + (2.0 * 9.51 /
            #                 20.0 - 2.0 * 9.51 / 5.0) * self.time_step * z[i]]
            # constraints += [-m_delta * delta[i] <= v_targeted[i] - v_true[i] - m_delta,
            #                 -(M_delta + 0.00000001) * delta[i] <= - (v_targeted[i] - v_true[i]) - 0.00000001]
            # constraints += [z[i] <= M_delta * delta[i],
            #                 z[i] >= m_delta * delta[i]]
            # constraints += [z[i] <= v_targeted[i] - v_true[i] - m_delta * (1 - delta[i]),
            #                 z[i] >= v_targeted[i] - v_true[i] - M_delta * (1 - delta[i])]

        for i in range(len(self.min_obstacle)):
            if self.min_obstacle[i] is not None:
                constraints += [self.max_obstacle[i] - s[i] <= M_s * (1 - y[i]),
                                s[i] - self.min_obstacle[i] <= M_s * y[i]]

        objective = cp.Maximize(cp.sum(s))
        prob = cp.Problem(objective,
                          constraints)
        prob.solve(verbose=True)
        s_2 = np.array(s.value).reshape((n, 1))
        t_2 = np.arange(0, self.max_time, self.time_step).reshape((n, 1))
        time_fxn_2 = np.hstack((t_2, s_2))

        # reset the pathTraj_2's velocity tuning to be time_fxn_2
        self.pathTraj_2 = PathTrajectory(self.pathTraj_2.path, time_fxn_2, self.pathTraj_2.total_time)
        self.target_velocity = np.array(v_targeted.value).reshape((n - 1, 1))

        self.vec_plan_space = np.array(self.vec_plan_space)
        # time-s figure
        if len(self.vec_plan_space) > 0:
            plt.figure(2)
            plt.plot(self.vec_plan_space[:, 0], self.vec_plan_space[:, 1], 'b.')
            plt.plot(np.arange(0, float(self.max_time), self.time_step), s.value)
            plt.axis([0, self.max_time, 0, 1])
            plt.xlabel('Time (sec)')
            plt.ylabel('Percent Along Path')
            plt.title('Velocity Planning Obstacles for Car 2')
            plt.show()
            plt.figure(3)
            plt.plot(np.arange(0, float(self.max_time - self.time_step), self.time_step), v_targeted.value)
            plt.plot(np.arange(0, float(self.max_time), self.time_step), v_true.value)
            plt.xlabel('Time (sec)')
            plt.ylabel('Target and true velocity')
            plt.show()
        else:
            print("No collisions along path")

    def calculate_trajectory(self, car1_vel):
        """
        Calculate the trajectory with velocities for each vehicle
        """
        pathTrajxy_1 = self.pathTraj_1.path
#        velocity_1 = np.full((pathTrajxy_1.shape[0], 1), 4.0)
        velocity_1 = np.full((pathTrajxy_1.shape[0], 1), car1_vel)
        pathTrajvelocity_1 = np.hstack((pathTrajxy_1, velocity_1))
#        x_1 = []
#        y_1 = []
#        v_1 = []
        x_2 = []
        y_2 = []
        v_2 = []
        t_arr = []
        iterator = 0

        for t in np.arange(0.0, self.max_time, self.time_step):
            #            x_2_t, y_2_t, v_2_t = self.pathTraj_2.determine_waypoint(t, self.pathTraj_1.total_path_length)
            #            x_1_t, y_1_t, v_1_t = self.pathTraj_1.determine_waypoint(t)
            x_2_t, y_2_t, v_2_t = self.pathTraj_2.determine_waypoint(t)
            if iterator < self.target_velocity.shape[0]:
                v_2_t = self.target_velocity[iterator, 0]
            else:
                v_2_t = 0.0
#            x_1.append(x_1_t)
#            y_1.append(y_1_t)
#            v_1.append(v_1_t)
            # v_2_t = v_2_t * self.max_time
            x_2.append(x_2_t)
            y_2.append(y_2_t)
            v_2.append(v_2_t)
            t_arr.append(t)
            iterator = iterator + 1
#        x_1 = np.array(x_1).reshape((-1, 1))
#        y_1 = np.array(y_1).reshape((-1, 1))
#        v_1 = np.array(v_1).reshape((-1, 1))
        x_2 = np.array(x_2).reshape((-1, 1))
        y_2 = np.array(y_2).reshape((-1, 1))
        v_2 = np.array(v_2).reshape((-1, 1))
        t_arr = np.array(t_arr).reshape((-1, 1))
#        pathTrajvelocity_1 = np.hstack((x_1, y_1, v_1, t_arr))
        pathTrajvelocity_2 = np.hstack((x_2, y_2, v_2, t_arr))

        return pathTrajvelocity_1, pathTrajvelocity_2

    def animate(self):
        fig, ax = plt.subplots()
        ax.axis([0, 6, 0, 6])
        ax.set_aspect("equal")
        self.point, = ax.plot(0, 1, marker="o", markersize=42.28)
        self.point2, = ax.plot(0, 1, marker="o", markersize=42.28)
        # ani = FuncAnimation(fig, self.update, interval=50, repeat=False,
        #                     frames=np.linspace(0, self.max_time, self.max_time * 100, endpoint=False))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cars Avoiding Collision')

        plt.show()

    def line(self, t):

        return np.array(self.pathTraj_1.update(t))

    # animation function for car 2
    def line2(self, t):
        return np.array(c2_traj.pathTraj_2.update(t))

    # used for running the animation function

    def update(self, t):

        # get x,y positions of each car at time t
        x, y = self.line(t)
        list1 = [[x], [y]]
        x, y = self.line2(t)
        list2 = [[x], [y]]

        self.point.set_data(list1[0], list1[1])
        self.point2.set_data(list2[0], list2[1])
        self.point.set_label('Car 1')
        self.point2.set_label('Car 2')
        plt.legend(loc="upper right")
        return [self.point, self.point2]
