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


class VelocityTuner:
    def __init__(self, pathTraj_1, pathTraj_2, pathTraj_3, max_time, time_step, car1_vel, car2_vel, car3_vel):
        self.pathTraj_1 = pathTraj_1
        self.pathTraj_2 = pathTraj_2
        self.pathTraj_3 = pathTraj_3
        self.pathTrajectories = np.array([self.pathTraj_1, self.pathTraj_2, self.pathTraj_3])
        self.max_time = int(max_time)  # there is no need to consider the max time, min time is enough
        self.time_step = time_step
        self.vec_plan_space = []
        self.min_obstacle = []
        self.max_obstacle = []        
        self.car1_vel = car1_vel
        self.car2_vel = car2_vel
        self.car3_vel = car3_vel
        self.car_velocities = np.array([self.car1_vel, self.car2_vel, self.car3_vel])
        self.target_velocities = []

    def tune_velocities(self):
        """
        Tunes the velocities
        """

        for i in range(1, 3):
                        
            # find positions of higher priority vehicles
            higher_car_pos = self.pathTrajectories[0].constant_velocity_trajectory(self.time_step, self.max_time, self.car_velocities[0])
            for j in range(1, i):
                higher_car_pos = np.hstack((higher_car_pos,self.pathTrajectories[j].constant_velocity_trajectory(self.time_step, self.max_time, self.car_velocities[j])))
            
            
            for t1 in range(0, int(self.max_time / self.time_step)):
                
                # get all positions of higher priority vehicles at current time
                curr_higher_car_pos = higher_car_pos[t1,:].reshape(int(higher_car_pos.shape[1]/2), 2)
                
                found_collision = []
                
                for t2 in np.arange(0.0, self.max_time, self.time_step):
                    
                    # current car's position
                    curr_car_pos = self.pathTrajectories[i].update(t2)
                    
                    # loop through all higher priority cars' positions at current time
                    # and check if there is a collision with current car's position
                    isCollided = False
                    
                    for pos in curr_higher_car_pos:                        
                        if (np.linalg.norm(curr_car_pos - pos) < 0.7):
                            isCollided = True
                            break
                    
                    if isCollided:
                        
                        # if collision occurs, then find time parameterized value along
                        # path that corresponds to this time
                        s = self.pathTrajectories[i].s
                        
                        # append to array for plotting
                        self.vec_plan_space.append([t1 * self.time_step, s])
                        found_collision.append(s)
                        
                if len(found_collision) > 0:
#                    found_collision = np.array(found_collision)
                    self.min_obstacle.append(found_collision[0])
                    self.max_obstacle.append(found_collision[-1])
                else:
                    self.min_obstacle.append(None)
                    self.max_obstacle.append(None)
                    
                self.pathTrajectories[i].restore()
            
            self.optimize_velocity_path(i)
        
            self.min_obstacle = []
            self.max_obstacle = []
                

    def optimize_velocity_path(self, car_idx):
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
            v_true <= np.full(n, self.car_velocities[car_idx]),
        ]
        for i in range(n - 1):
            constraints += [v_targeted[i] <= self.car_velocities[car_idx]]
            constraints += [s[i + 1] == s[i] + v_true[i] * self.time_step / self.pathTrajectories[car_idx].total_path_length]
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
        self.pathTrajectories[car_idx] = PathTrajectory(self.pathTrajectories[car_idx].path, time_fxn_2, self.pathTrajectories[car_idx].total_time)
        self.target_velocities.append(np.array(v_targeted.value).reshape((n - 1, 1)))

        self.vec_plan_space = np.array(self.vec_plan_space)
        # time-s figure
#        if len(self.vec_plan_space) > 0:
#            plt.figure(2)
#            plt.plot(self.vec_plan_space[:, 0], self.vec_plan_space[:, 1], 'b.')
#            plt.plot(np.arange(0, float(self.max_time), self.time_step), s.value)
#            plt.axis([0, self.max_time, 0, 1])
#            plt.xlabel('Time (sec)')
#            plt.ylabel('Percent Along Path')
#            plt.title('Velocity Planning Obstacles for Car '+str(car_idx+1))
#            plt.show()
#            plt.figure(3)
#            plt.plot(np.arange(0, float(self.max_time - self.time_step), self.time_step), v_targeted.value)
#            plt.plot(np.arange(0, float(self.max_time), self.time_step), v_true.value)
#            plt.xlabel('Time (sec)')
#            plt.ylabel('Target and true velocity')
#            plt.show()
#        else:
#            print("No collisions along path")
        self.vec_plan_space = []

    def calculate_trajectory(self, car1_vel):
        """
        Calculate the trajectory with velocities for each vehicle
        """
        pathTrajxy_1 = self.pathTraj_1.path
        velocity_1 = np.full((pathTrajxy_1.shape[0], 1), car1_vel)
        pathTrajvelocity_1 = np.hstack((pathTrajxy_1, velocity_1))
        x_2 = []
        y_2 = []
        v_2 = []
        x_3 = []
        y_3 = []
        v_3 = []
        t_arr = []
        iterator = 0

        for t in np.arange(0.0, self.max_time, self.time_step):
            x_2_t, y_2_t, v_2_t = self.pathTraj_2.determine_waypoint(t)
            x_3_t, y_3_t, v_3_t = self.pathTraj_3.determine_waypoint(t)
            if iterator < self.target_velocities[0].shape[0]:
                v_2_t = self.target_velocities[0][iterator, 0]
            else:
                v_2_t = 0.0
            if iterator < self.target_velocities[1].shape[0]:
                v_3_t = self.target_velocities[1][iterator, 0]
            else:
                v_3_t = 0.0
            
            x_2.append(x_2_t)
            y_2.append(y_2_t)
            v_2.append(v_2_t)
            x_3.append(x_3_t)
            y_3.append(y_3_t)
            v_3.append(v_3_t)
            t_arr.append(t)
            iterator = iterator + 1
            
        x_2 = np.array(x_2).reshape((-1, 1))
        y_2 = np.array(y_2).reshape((-1, 1))
        v_2 = np.array(v_2).reshape((-1, 1))
        x_3 = np.array(x_3).reshape((-1, 1))
        y_3 = np.array(y_3).reshape((-1, 1))
        v_3 = np.array(v_3).reshape((-1, 1))
        t_arr = np.array(t_arr).reshape((-1, 1))
        pathTrajvelocity_2 = np.hstack((x_2, y_2, v_2, t_arr))
        pathTrajvelocity_3 = np.hstack((x_3, y_3, v_3, t_arr))

        return pathTrajvelocity_1, pathTrajvelocity_2, pathTrajvelocity_3
