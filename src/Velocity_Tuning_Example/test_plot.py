# Import packages.
import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from PathTrajectory import PathTrajectory
from matplotlib.animation import FuncAnimation

# instantiate car1 and car2 waypoints
c1_pos = np.array([[2, 5],
                   [3.5, 1.5],
                  [5, 5]])
c2_pos = np.array([[1, 3],
                   [3,4],
                   [3.5, 4],
                  [5, 3]])

# set car1 velocity profile waypoints
# [time, percent along path]
c1_vel = np.array([[0, 0],
                   [0.5, 0.5],
                  [1, 1]])
c2_vel = np.array([[0,0],
                  [1,1]])
    
c1_traj = PathTrajectory(c1_pos, c1_vel)


# set max time
max_time = 1.0

# arrays for plotting
vec_plan_space = []
real_space_path_c1 = []
real_space_path_c2 = []
real_space_collided = []
min_obstacle = []
max_obstacle = []

# iterate thru time
for t in np.arange(0.0, max_time, 0.01):
    
    curr_c1_pos = c1_traj.update(t)

    found_collision = []
    
    c2_traj = PathTrajectory(c2_pos, c2_vel)

    for t2 in np.arange(0.0, max_time, 0.01):
        curr_c2_pos = c2_traj.update(t2)

        # calculate if the two cars collide
        if (np.linalg.norm(curr_c1_pos - curr_c2_pos) < 1):

            # add collision points to array for plotting
            real_space_collided.append(curr_c1_pos)
            real_space_collided.append(curr_c2_pos)

            # if they collide, then find the time parameterized value along the path
            # that corresponds to this time
            s = c2_traj.s

            # append to array for plotting
            vec_plan_space.append([t, s])
            found_collision.append(s)
        else:
            # if they don't collide, just add them to the real space path for plotting
            real_space_path_c1.append(curr_c1_pos)
            real_space_path_c2.append(curr_c2_pos)

    if len(found_collision) > 0:
        found_collision = np.array(found_collision)
        min_obstacle.append(found_collision[0])
        max_obstacle.append(found_collision[-1])
    else:
        min_obstacle.append(None)
        max_obstacle.append(None)

# convert to numpy arrays
vec_plan_space = np.array(vec_plan_space)
real_space_path_c1 = np.array(real_space_path_c1)
real_space_path_c2 = np.array(real_space_path_c2)
real_space_collided = np.array(real_space_collided)

n = 100
s = cp.Variable(n, nonneg=True)
y = cp.Variable(100, integer=True)

M = 100.0
constraints = [
    s[0] == 0.0,
    s <= np.ones(n),
    y >= np.zeros(100),
    y <= np.ones(100)
]
for i in range(n - 1):
    constraints += [s[i] - s[i + 1] <= 0, s[i + 1] - s[i] <= 0.01]

for i in range(100):
    if min_obstacle[i] != None:
        constraints += [max_obstacle[i] - s[i] <= M * (1 - y[i]), s[i] - min_obstacle[i] <= M * y[i]]

objective = cp.Maximize(cp.sum(s))
prob = cp.Problem(objective,
                  constraints)
current_time = time.time()
prob.solve()
end_time = time.time()
print(end_time - current_time)

if len(vec_plan_space) > 0:
    plt.figure(1)
    plt.plot(vec_plan_space[:, 0], vec_plan_space[:, 1], 'b.')
    plt.plot(np.arange(0, 1.00, 0.01), s.value)
    plt.axis([0, max_time, 0, 1])
    plt.xlabel('Time (sec)')
    plt.ylabel('Percent Along Path')
    plt.title('Velocity Planning Obstacles for Car 2')
    plt.show()
else:
    print("No collisions along path")

# plt.figure(2)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.axis([0, 6, 0, 6])
# plt.title('Car 1 and Car 2 Paths')
# plt.plot(real_space_path_c1[:, 0], real_space_path_c1[:, 1], 'g.', label="car 1 path")
# plt.plot(real_space_path_c2[:, 0], real_space_path_c2[:, 1], 'b.', label="car 2 path")
# plt.plot(real_space_collided[:, 0], real_space_collided[:, 1], 'r.', label='collision')
# plt.plot([c1_pos[0, 0], c2_pos[0, 0]], [c1_pos[0, 1], c2_pos[0, 1]], 'ko', label="start pos")
# plt.plot([c1_pos[1, 0], c2_pos[1, 0]], [c1_pos[1, 1], c2_pos[1, 1]], 'kx', label="finish pos")
# plt.legend(loc="upper right")
# plt.show()

fig, ax = plt.subplots()
ax.axis([0, 6, 0, 6])
ax.set_aspect("equal")

# set the objects, markersize of 42.28 roughly equals radius = 0.5 w/ [0,6] axis limits
point, = ax.plot(0, 1, marker="o", markersize=42.28)
point2, = ax.plot(0, 1, marker="o", markersize=42.28)

# calc path vectors
c2_fxn = np.array(c2_pos[1] - c2_pos[0])

# set waypoints in time-s space to follow
s_2 = np.array(s.value).reshape((100, 1))
t_2 = np.arange(0, 1.00, 0.01).reshape((100, 1))
time_fxn_2 = np.hstack((t_2, s_2))


c1_traj = PathTrajectory(c1_pos, c1_vel)
c2_traj = PathTrajectory(c2_pos, time_fxn_2)

def line(t):
    global c1_traj
    
    return np.array(c1_traj.update(t))

# animation function for car 2
def line2(t):
    
    global c2_traj
    return np.array(c2_traj.update(t))

# used for running the animation function


def update(t):

    # get x,y positions of each car at time t
    x, y = line(t)
    list1 = [[x], [y]]
    x, y = line2(t)
    list2 = [[x], [y]]

    point.set_data(list1[0], list1[1])
    point2.set_data(list2[0], list2[1])
    point.set_label('Car 1')
    point2.set_label('Car 2')
    plt.legend(loc="upper right")
    return [point, point2]


ani = FuncAnimation(fig, update, interval=50, repeat=False,
                    frames=np.linspace(0, 1, 100, endpoint=False))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cars Avoiding Collision')

plt.show()
