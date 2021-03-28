import numpy as np
import matplotlib.pyplot as plt

# instantiate car1 and car2 waypoints
c1_pos = np.array([[3, 2],
                  [4, 5]])
c2_pos = np.array([[2, 3],
                  [5, 4]])
c1_size = 0.5
c2_size = 0.5

# set car1 velocity profile waypoints
# [time, percent along path]
c1_vel = np.array([[0, 0],
                  [1, 1]])

# set max time
max_time = 1.0

# path functions for each car
c1_fxn = np.array(c1_pos[1]-c1_pos[0])
c2_fxn = np.array(c2_pos[1]-c2_pos[0])

# arrays for plotting
vec_plan_space = []
real_space_path_c1 = []
real_space_path_c2 = []
real_space_collided = []


# iterate thru time
for t in np.arange(0.0, max_time, 0.01):

    # calculate the scaled time for car 1's current segment
    c1_seg_time = (c1_vel[1,1]-c1_vel[0,1])/(c1_vel[1,0]-c1_vel[0,0])*t
    curr_c1_pos = (c1_pos[0]+c1_fxn*c1_seg_time)

    for t2 in np.arange(0.0, max_time, 0.01):
        curr_c2_pos = (c2_pos[0]+c2_fxn*t2)

        # calculate if the two cars collide
        if (np.linalg.norm(curr_c1_pos-curr_c2_pos) < 1):

            # add collision points to array for plotting
            real_space_collided.append(curr_c1_pos)
            real_space_collided.append(curr_c2_pos)

            # if they collide, then find the time parameterized value along the path
            # that corresponds to this time
            c2_dist_from_start = np.linalg.norm(curr_c2_pos - c2_pos[0])
            s = c2_dist_from_start/np.linalg.norm(c2_pos[1]-c2_pos[0])

            # append to array for plotting
            vec_plan_space.append([t, s])
        else:
            # if they don't collide, just add them to the real space path for plotting
            real_space_path_c1.append(curr_c1_pos)
            real_space_path_c2.append(curr_c2_pos)

# convert to numpy arrays
vec_plan_space = np.array(vec_plan_space)
real_space_path_c1 = np.array(real_space_path_c1)
real_space_path_c2 = np.array(real_space_path_c2)
real_space_collided = np.array(real_space_collided)

if len(vec_plan_space) > 0:
    plt.figure(1)
    plt.plot(vec_plan_space[:,0], vec_plan_space[:,1], 'b.')
    plt.axis([0, max_time, 0, 1])
    plt.xlabel('Time (sec)')
    plt.ylabel('Percent Along Path')
    plt.title('Velocity Planning Obstacles for Car 2')
    plt.show()
else:
    print("No collisions along path")

plt.figure(2)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([0, 6, 0, 6])
plt.title('Car 1 and Car 2 Paths')
plt.plot(real_space_path_c1[:, 0], real_space_path_c1[:, 1], 'g.', label="car 1 path")
plt.plot(real_space_path_c2[:, 0], real_space_path_c2[:, 1], 'b.', label="car 2 path")
plt.plot(real_space_collided[:,0], real_space_collided[:,1], 'r.', label='collision')
plt.plot([c1_pos[0, 0], c2_pos[0, 0]], [c1_pos[0, 1], c2_pos[0, 1]], 'ko', label="start pos")
plt.plot([c1_pos[1, 0], c2_pos[1, 0]], [c1_pos[1, 1], c2_pos[1, 1]], 'kx', label="finish pos")
plt.legend(loc="upper right")
plt.show()


