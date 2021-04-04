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
    def __init__(self, pathTraj_1, pathTraj_2, max_time):
        self.pathTraj_1 = pathTraj_1
        self.pathTraj_2 = pathTraj_2
        self.max_time = int(max_time)
        self.vec_plan_space = []
        self.min_obstacle = []
        self.max_obstacle = []
    
    def tune_velocities(self):
        """
        Tunes the velocities
        """
        
        # iterate thru time
        for t in np.arange(0.0, self.max_time, 0.01):
            
            curr_c1_pos = self.pathTraj_1.update(t)
        
            found_collision = []
            
            c2_traj = self.pathTraj_2
        
            for t2 in np.arange(0.0, self.max_time, 0.01):
                curr_c2_pos = c2_traj.update(t2)
        
                # calculate if the two cars collide
                if (np.linalg.norm(curr_c1_pos - curr_c2_pos) < 1):
        
                    # if they collide, then find the time parameterized value along the path
                    # that corresponds to this time
                    s = c2_traj.s
        
                    # append to array for plotting
                    self.vec_plan_space.append([t, s])
            if len(found_collision) > 0:
                found_collision = np.array(found_collision)
                self.min_obstacle.append(found_collision[0])
                self.max_obstacle.append(found_collision[-1])
            else:
                self.min_obstacle.append(None)
                self.max_obstacle.append(None)
        
        self.optimize_velocity_path()
    
    def optimize_velocity_path(self):
        """
        Apply the optimization solver
        """
        n = self.max_time*100
        s = cp.Variable(n, nonneg=True)
        y = cp.Variable(self.max_time*100, integer=True)
        
        M = float(self.max_time*100)
        constraints = [
            s[0] == 0.0,
            s <= np.ones(n),
            y >= np.zeros(self.max_time*100),
            y <= np.ones(self.max_time*100)
        ]
        for i in range(n - 1):
            constraints += [s[i] - s[i + 1] <= 0, s[i + 1] - s[i] <= 0.01]
        
        for i in range(self.max_time*100):
            if self.min_obstacle[i] != None:
                constraints += [self.max_obstacle[i] - s[i] <= M * (1 - y[i]), 
                                s[i] - self.min_obstacle[i] <= M * y[i]]
        
        objective = cp.Maximize(cp.sum(s))
        prob = cp.Problem(objective,
                          constraints)
        prob.solve()
        s_2 = np.array(s.value).reshape((self.max_time*100, 1))
        t_2 = np.arange(0, self.max_time, 0.01).reshape((self.max_time*100, 1))
        time_fxn_2 = np.hstack((t_2, s_2))
        
        # reset the pathTraj_2's velocity tuning to be time_fxn_2
        self.pathTraj_2 = PathTrajectory(self.pathTraj_2.path, time_fxn_2)
    
    def animate(self):
        fig, ax = plt.subplots()
        ax.axis([0, 6, 0, 6])
        ax.set_aspect("equal")
        self.point, = ax.plot(0, 1, marker="o", markersize=42.28)
        self.point2, = ax.plot(0, 1, marker="o", markersize=42.28)
        ani = FuncAnimation(fig, self.update, interval=50, repeat=False,
                        frames=np.linspace(0, self.max_time, self.max_time*100, endpoint=False))
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
    
    
    