#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:26:34 20211

@author: meam520
"""
import numpy as np
import matplotlib.pyplot as plt


class PathTrajectory:
    def __init__(self, path, time_s, total_time):
        self.path = path        # list of waypoints
        self.time_s = time_s    # list of [time , s] values that correspond to
        # waypoint at given index

        # the cumulative s value for the current path segment
        self.path_s, self.total_path_length = self.calc_total_path_length(self.path)

        # current percent along the total path traveled
        self.s = 0

        # current velocity in m/s
        self.v = 0

        # the current segment idx in the path_s array
        self.curr_path_seg = 0

        # the current segment idx in the time_s array
        self.curr_s_seg = 0

        # the current vector for current segment in path
        self.curr_vec = np.array(path[1] - path[0])

        # velocity of the first vehicle
        # self.velocity_car_1 = 3.0

        # total time it takes to traverse path
        self.total_time = total_time
        
    def update(self, t):
        """
        Returns the current position along the path for given t value

        Args:
            IN: t - current time in seconds
            OUT: pos - current [x,y] value
        """

        # increment current segment idx in time_s array
        if t > self.time_s[self.curr_s_seg + 1, 0]:

            if self.curr_s_seg + 1 < len(self.time_s) - 1:
                self.curr_s_seg += 1
            else:
                t = self.time_s[self.curr_s_seg + 1, 0]

        # increment current segment idx in path array
        if self.s > self.path_s[self.curr_path_seg + 1]:
            if self.curr_path_seg + 1 < len(self.path_s) - 1:
                self.curr_path_seg += 1
            self.curr_vec = np.array(self.path[self.curr_path_seg + 1] -
                                     self.path[self.curr_path_seg])

        # calc speed in units of s/time
        time_s_speed = np.array((self.time_s[self.curr_s_seg + 1, 1] -
                                 self.time_s[self.curr_s_seg, 1]) /
                                (self.time_s[self.curr_s_seg + 1, 0] -
                                 self.time_s[self.curr_s_seg, 0]))

        # calculate current value of s along path we are
        self.s = (t - self.time_s[self.curr_s_seg, 0]) * time_s_speed + \
            self.time_s[self.curr_s_seg, 1]

        # calc total length of current segment in units of s
        curr_path_seg_total_s_len = self.path_s[self.curr_path_seg + 1] - \
            self.path_s[self.curr_path_seg]

        # convert current s value into position units
        pos = self.path[self.curr_path_seg] + (self.s - self.path_s[self.curr_path_seg])\
            / curr_path_seg_total_s_len * self.curr_vec

        return pos

    def constant_velocity_trajectory(self, d_t, max_time, v):
        """
        Returns the positions along the path discretized at the given d_t value, following constant target velocity v and the underlying lower-level pd controller

        Args:
            IN: d_t - discrete time steps
                max_time - maximum time for the simulation
                v - constant target velocity
            OUT: pos - n * 2 np array of positions
        """

        pos = []
        s_array = []

        for t in np.arange(0.0, max_time, d_t):
            # increment current segment idx in path array
            if self.s > self.path_s[self.curr_path_seg + 1]:
                if self.curr_path_seg + 1 < len(self.path_s) - 1:
                    self.curr_path_seg += 1
                self.curr_vec = np.array(self.path[self.curr_path_seg + 1] -
                                         self.path[self.curr_path_seg])

            # calc speed in units of s/time
            time_s_speed = self.v / self.total_path_length
            if self.v <= v:
                self.v = self.v + 2.0 * 9.51 / 20.0 * (v - self.v) * d_t
            else:
                self.v = self.v + 2.0 * 9.51 / 5.0 * (v - self.v) * d_t

            # calculate current value of s along path we are
            self.s += time_s_speed * d_t

            # calc total length of current segment in units of s
            curr_path_seg_total_s_len = self.path_s[self.curr_path_seg + 1] - \
                self.path_s[self.curr_path_seg]

            # convert current s value into position units
            position = self.path[self.curr_path_seg] + (self.s - self.path_s[self.curr_path_seg])\
                / curr_path_seg_total_s_len * self.curr_vec

            pos.append([position[0], position[1]])
            s_array.append(self.s)

        pos = np.array(pos)

#        plt.figure(1)
#        plt.plot(np.arange(0.0, max_time, d_t), s_array)
#        plt.xlabel('Time (sec)')
#        plt.ylabel('Percent Along Path')
#        plt.show()

        return pos

#    def determine_waypoint(self, t, path_length_vehicle_1):
    def determine_waypoint(self, t):
        """
        Returns the current position and velocity along the path for given t value

        Args:
            IN: t - current time in seconds
            OUT: x, y, v - current x, y, v value
        """
        # increment current segment idx in time_s array
        if t > self.time_s[self.curr_s_seg + 1, 0]:

            if self.curr_s_seg + 1 < len(self.time_s) - 1:
                self.curr_s_seg += 1
            else:
                t = self.time_s[self.curr_s_seg + 1, 0]

        # increment current segment idx in path array
        if self.s > self.path_s[self.curr_path_seg + 1]:
            if self.curr_path_seg + 1 < len(self.path_s) - 1:
                self.curr_path_seg += 1
            self.curr_vec = np.array(self.path[self.curr_path_seg + 1] -
                                     self.path[self.curr_path_seg])

        # calc speed in units of s/time
        time_s_speed = np.array((self.time_s[self.curr_s_seg + 1, 1] -
                                 self.time_s[self.curr_s_seg, 1]) /
                                (self.time_s[self.curr_s_seg + 1, 0] -
                                 self.time_s[self.curr_s_seg, 0]))

        # calculate current value of s along path we are
        self.s = (t - self.time_s[self.curr_s_seg, 0]) * time_s_speed + \
            self.time_s[self.curr_s_seg, 1]

        # calc total length of current segment in units of s
        curr_path_seg_total_s_len = self.path_s[self.curr_path_seg + 1] - \
            self.path_s[self.curr_path_seg]

        # convert current s value into position units
        pos = self.path[self.curr_path_seg] + (self.s - self.path_s[self.curr_path_seg])\
            / curr_path_seg_total_s_len * self.curr_vec

        # calculate current velocity of vehicle
#        vel = time_s_speed * self.velocity_car_1 * self.total_path_length / path_length_vehicle_1
        vel = time_s_speed * self.total_path_length / self.total_time

        return pos[0], pos[1], vel

    def calc_total_path_length(self, path):
        """
        Calculates the aggregate s value for the given segment

        Args:

        In:
            path - n x 2 array of path points

        Out:
            path_s - (n-1) x 1 list of s vals for given segment at idx

        """
        path_s = np.array([0])
        total_path_length = 0
        for i in range(len(path) - 1):
            total_path_length += np.linalg.norm(path[i + 1] - path[i])
        for i in range(len(path) - 1):
            path_s = np.append(path_s, np.linalg.norm(path[i + 1] - path[i]) /
                               total_path_length + path_s[i])
        return path_s, total_path_length

    def restore(self):
        """
        restore the parameters for future calculations
        """
        self.s = 0
        self.v = 0
        self.curr_path_seg = 0
        self.curr_s_seg = 0
        self.curr_vec = np.array(self.path[1] - self.path[0])


# pos_plot = []
#
#
# pos_plot = np.array(pos_plot)
# plt.figure()
# plt.plot(pos_plot[:,0], pos_plot[:,1], 'r.')
# plt.axis([2, 6, 2, 6])
# plt.show()
