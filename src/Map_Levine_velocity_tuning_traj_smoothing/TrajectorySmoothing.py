#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:14:06 2021

@author: meam520
"""


import numpy as np
import matplotlib.pyplot as plt

waypoints = np.array([[ 3.90381196, -0.0209113],
                 [ 3.4175844 , -0.17185454],
                 [ 3.11886669 ,-0.14414657],
                 [ 2.55273275 , 0.04153676],
                 [ 2.08106487, -0.22191367],
                 [ 1.48739343, -0.24855908],
                 [ 0.89204041, -0.27761634],
                 [ 0.29382866, -0.23940197],
                 [-0.29543927, -0.14085241],
                 [-0.88951877, -0.11933159],
                 [-1.44345253, -0.28084456],
                 [-1.68420664, -0.32390108],
                 [-2.10195192, -0.05172465],
                 [-2.70052411, -0.01772543],
                 [-3.        ,  0.        ]])

waypoints = np.flip(waypoints, 0)

class TrajectorySmoothing:
    def __init__(self, waypoints, speed):
        """
        Smooths a given set of waypoints using piecewise polynomial
        Args:
            waypoints: list of waypoints to do trajectory smoothing on
            speed: speed at which to traverse along smoothed path
        """
        self.waypoints = waypoints

        # total # of segments along path
        self.num_segments = np.shape(waypoints)[0]-1

        # rate of travel along path
        self.speed = speed

        # LH-side A matrix
        self.A = np.zeros(((self.num_segments+1)*6, (self.num_segments+1)*6))

        # RH-side b vector for constraints
        self.b = np.zeros((self.num_segments+1)*6)

        # vector of polynomial constants we are solving for
        self.c = np.zeros(((self.num_segments+1)*6, 2))

        self.prev_total = 0
        self.curr_segment = 0
        self.curr_segment_final_time = self.get_duration(0)
        self.curr_segment_start_time = 0
        self.total_time = 0
        self.calc_total_time()

        # iterate thru x, y axes to solve piecewise polynomial
        for k in range(0, 2):
            self.construct_end_point_boundary_cond(k)
            self.construct_intermediary_cond(k)
            self.construct_regular_cont_cond()
            self.construct_higher_cont_cond()
            self.solve(k)

    def get_total_path(self, time_step):
        """
        Get the total path 
        
        Args:
            time_step: duration of time between sampled points (i.e. 0.01 sec)
        
        Returns:
            n x 2 array of points along path
        """
        new_path = []
        l = np.arange(0.0, self.total_time, time_step)
        for t in l:
            new_path.append(self.update(t))
        return np.array(new_path)

    def update(self, t):
        """
        Call this fxn to get the x,y coordinate along the path at time t
        If t > total path time, then it will just return the final position along path

        Args:
            t: current instant in time (sec)

        Returns:
            [x,y] coordinates at time t along curve
        """

        if t > self.total_time:
            return self.waypoints[self.num_segments]
        self.get_curr_segment(t)
        new_t = t - self.curr_segment_start_time
        return self.get_xy_coords(self.curr_segment, new_t)

    def get_curr_segment(self, t):
        """
        Gets the current segment along the path at the given overall time t

        Args:
            t: Current total time along path

        Returns:
            None
        """
        if t > self.curr_segment_final_time:
            if self.curr_segment < (self.num_segments-1):
                self.curr_segment += 1
                self.curr_segment_final_time += self.get_duration(self.curr_segment)
                self.curr_segment_start_time += self.get_duration(self.curr_segment-1)

    def calc_total_time(self):
        for i in range(self.num_segments-1):
            self.total_time += self.get_duration(i)


    def get_xy_coords(self, curr_seg, t):
        """
        Calculate x, y coord at given time t

        Args:
            curr_seg: current segment along path
            t: current time along current segment's path (resets to 0 at each segment start)

        Returns:

        """
        x = self.c[curr_seg*6, 0]*t**5 + self.c[curr_seg*6+1, 0]*t**4 + self.c[curr_seg*6+2, 0]*t**3 + \
                self.c[curr_seg*6+3, 0]*t**2 + self.c[curr_seg*6+4, 0]*t + self.c[curr_seg*6+5, 0]
        y = self.c[curr_seg * 6, 1] * t ** 5 + self.c[curr_seg * 6 + 1, 1] * t ** 4 + self.c[
            curr_seg * 6 + 2, 1] * t ** 3 + \
            self.c[curr_seg * 6 + 3, 1] * t ** 2 + self.c[curr_seg * 6 + 4, 1] * t + self.c[curr_seg * 6 + 5, 1]
        return [x, y]

    def solve(self, k):
        """
        Solve linear system for given axis (either x or y)

        Args:
            k: coordinate axis index (0 = x, 1 = y)

        Returns:
            None
        """
        self.c[:,k] = np.linalg.solve(self.A, self.b.T)

    def construct_higher_cont_cond(self):
        """
        Construct higher order continuity constraints (jerk)

        Returns:
            None
        """
        for i in range(self.num_segments):
            t = self.get_duration(i)
            self.A[self.prev_total + 1, 6*i:6*i+3] = [60*t**2, 24*t, 6]
            self.A[self.prev_total + 2, 6*i:6*i+2] = [120*t, 24]
            self.A[self.prev_total + 1, 6*(i+1)+2] = -6
            self.A[self.prev_total + 2, 6*(i+1)+1] = -24

            self.b[self.prev_total + 1] = 10
            self.b[self.prev_total + 2] = 10

            self.prev_total = self.prev_total + 2

    def construct_regular_cont_cond(self):
        """
        Construct regular continuity constraints (velocity and acceleration)

        Returns:
            None
        """
        for i in range(self.num_segments):
            t = self.get_duration(i)
            self.A[self.prev_total + 1, 6*i:6*i+5] = [5*t**4, 4*t**3, 3*t**2, 2*t, 1]
            self.A[self.prev_total + 2, 6*i:6*i+4] = [20*t**3, 12*t**2, 6*t, 2]
            self.A[self.prev_total + 1, 6*(i+1)+4] = -1
            self.A[self.prev_total + 2, 6*(i+1)+3] = -2

            self.b[self.prev_total + 1] = 0
            self.b[self.prev_total + 2] = 0

            self.prev_total = self.prev_total + 2

    def construct_intermediary_cond(self, k):
        """
        Sets intermediary boundary conditions in A and b matrices

        Args:
            k: index of x or y axis (will be 0 or 1)

        Returns:
            Nothing
        """
        for i in range(self.num_segments):
            t = self.get_duration(i)
            self.A[6+i*2,i*6:i*6+6] = [t**5, t**4, t**3, t**2, t, 1]
            self.A[7+i*2,6*(i+1)+5] = 1

            self.b[6+i*2] = self.waypoints[i+1, k]
            self.b[7+i*2] = self.waypoints[i+1, k]

            self.prev_total = self.prev_total + 2

    def construct_end_point_boundary_cond(self, k):
        """
        Construct the end point boundary conditions
        Only applies to 1st segment and last segment
        :param k : current axis (x=0, y=1, z=2)
        :return: none
        """
        # starting point BCs for pos, vel, accel
        self.A[0,5] = 1
        self.A[1,4] = 1
        self.A[2,3] = 2

        # end point BCs for pos, vel, accel
        t = self.get_duration(self.num_segments-1) # total duration for last segment
        self.A[3,-6:] = [t**5, t**4, t**3, t**2, t, 1]
        self.A[4,-6:-1] = [5*t**4, 4*t**3, 3*t**2, 2*t, 1]
        self.A[5,-6:-2] = [20*t**3, 12*t**2, 6*t, 2]

        self.b[0:6] = [self.waypoints[0,k], 0, 0, self.waypoints[self.num_segments, k], 0, 0]

        self.prev_total = 5


    def get_duration(self, curr_segment_idx):
        """
        calculates total time to traverse current segment assuming constant speed
        :param curr_segment_idx:
        :return: time of current segment
        """
        dist = np.linalg.norm(self.waypoints[curr_segment_idx+1] - self.waypoints[curr_segment_idx])
        return dist/self.speed

#traj = traj_gen(waypoints, 3)
#
#new_path = []
#l = np.arange(0.0, 3, 0.02)
#for t in l:
#    new_path.append(traj.update(t))
#new_path = np.array(new_path)
#fig = plt.figure()
#plt.axis([-4,4,-2,2])
#plt.plot(new_path[:,0], new_path[:,1], 'r.')
#plt.plot(waypoints[:,0], waypoints[:,1], 'b*')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

