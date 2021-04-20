# Import packages.
import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from PathTrajectory import PathTrajectory
from matplotlib.animation import FuncAnimation
from TrajectorySmoothing import TrajectorySmoothing

# instantiate car1 and car2 waypoints

c1_pos = np.array([[2.22374142e+00, -3.79893641e-03],
                   [1.63769408e+00, 3.30800844e-02],
                   [1.07068519e+00, -2.75612004e-02],
                   [6.65643282e-01, -1.51545069e-01],
                   [7.94652784e-02, -2.07459981e-01],
                   [-2.18877129e-01, -2.38952966e-01],
                   [-8.13397358e-01, -2.63592846e-01],
                   [-1.38573766e+00, -3.71340052e-01],
                   [-1.96368019e+00, -5.18964665e-01],
                   [-2.37448779e+00, -4.63992759e-01],
                   [-2.96652200e+00, -5.43226038e-01],
                   [-3.54476297e+00, -3.99039045e-01],
                   [-3.86796226e+00, -2.06865596e-01],
                   [-4.45580315e+00, -1.74883870e-01],
                   [-4.99744705e+00, -7.54591859e-08]])
c2_pos = np.array([[4.09344206e+00, -1.75239803e-01],
                   [3.79392378e+00, -1.58245731e-01],
                   [3.20005712e+00, -1.61349176e-01],
                   [2.61035169e+00, -2.22532707e-01],
                   [2.07353467e+00, -1.28643557e-01],
                   [1.77660251e+00, -8.58499525e-02],
                   [1.19250716e+00, 2.99515409e-02],
                   [5.94109720e-01, 6.82018614e-02],
                   [-5.42762556e-03, 5.20939999e-02],
                   [-6.03895385e-01, 6.63778575e-02],
                   [-1.20257923e+00, 7.74802401e-02],
                   [-1.80084216e+00, 6.52300186e-02],
                   [-2.39992338e+00, 3.78900126e-02],
                   [-2.99872352e+00, 1.88647965e-08]])

# reverse order of waypoints as they're backwards
c1_pos = np.flip(c1_pos, 0)
c2_pos = np.flip(c2_pos, 0)

time_step = 0.02

c1_speed = 5.0
c2_speed = 2.5
trajSmooth_1 = TrajectorySmoothing(c1_pos, c1_speed)
trajSmooth_2 = TrajectorySmoothing(c2_pos, c2_speed)

# set max time
max_time = int(max(trajSmooth_1.total_time, trajSmooth_2.total_time))
print(max_time)

# set car1 velocity profile waypoints
# [time, percent along path]
c1_vel = np.array([[0, 0],
                  [trajSmooth_1.total_time, 1]])
c2_vel = np.array([[0, 0],
                  [trajSmooth_2.total_time, 1]])

c1_traj = PathTrajectory(trajSmooth_1.get_total_path(time_step), c1_vel, 1.0)

# arrays for plotting
vec_plan_space = []
real_space_path_c1 = []
real_space_path_c2 = []
real_space_collided = []
min_obstacle = []
max_obstacle = []

# iterate thru time
for t in np.arange(0.0, max_time, time_step):

    curr_c1_pos = c1_traj.update(t)

    found_collision = []

    trajSmooth_2 = TrajectorySmoothing(c2_pos, c2_speed)
    c2_traj = PathTrajectory(trajSmooth_2.get_total_path(time_step), c2_vel, 1.0)

    for t2 in np.arange(0.0, max_time, time_step):
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

n = int(max_time * 1 / time_step)
s = cp.Variable(n, nonneg=True)
y = cp.Variable(n, integer=True)

M = float(max_time * 1 / time_step)
constraints = [
    s[0] == 0.0,
    s <= np.ones(n),
    y >= np.zeros(n),
    y <= np.ones(n)
]
for i in range(n - 1):
    constraints += [s[i] - s[i + 1] <= 0, s[i + 1] - s[i] <= time_step / max_time]

for i in range(len(min_obstacle)):
    if min_obstacle[i] != None:
        constraints += [max_obstacle[i] - s[i] <= M * (1 - y[i]), s[i] - min_obstacle[i] <= M * y[i]]

objective = cp.Maximize(cp.sum(s))
prob = cp.Problem(objective,
                  constraints)
current_time = time.time()
prob.solve()
end_time = time.time()
print(end_time - current_time)

# time-s figure
if len(vec_plan_space) > 0:
    plt.figure(1)
    plt.plot(vec_plan_space[:, 0], vec_plan_space[:, 1], 'b.')
    plt.plot(np.arange(0, float(max_time), time_step), s.value)
    plt.axis([0, max_time, 0, 1])
    plt.xlabel('Time (sec)')
    plt.ylabel('Percent Along Path')
    plt.title('Velocity Planning Obstacles for Car 2')
    plt.show()
else:
    print("No collisions along path")

# path figure
plt.figure(2)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([-10, 10, -6, 6])
plt.title('Car 1 and Car 2 Paths')
plt.plot(real_space_path_c1[:, 0], real_space_path_c1[:, 1], 'g.', label="car 1 path")
plt.plot(real_space_path_c2[:, 0], real_space_path_c2[:, 1], 'b.', label="car 2 path")
plt.plot(real_space_collided[:, 0], real_space_collided[:, 1], 'r.', label='collision')
plt.plot(real_space_path_c1[0, 0], real_space_path_c1[0, 1], 'ko', label="start")
plt.plot(real_space_path_c2[0, 0], real_space_path_c2[0, 1], 'ko', label="start")
plt.plot(real_space_path_c1[-1, 0], real_space_path_c1[-1, 1], 'kx', label="end")
plt.plot(real_space_path_c2[-1, 0], real_space_path_c2[-1, 1], 'kx', label="end")
plt.legend(loc="upper right")
plt.show()

# animation figure
fig, ax = plt.subplots()
ax.axis([-10, 10, -6, 6])
ax.set_aspect("equal")

# set the objects, markersize of 42.28 roughly equals radius = 0.5 w/ [0,6] axis limits
point, = ax.plot(0, 1, marker="o", markersize=10.28)
point2, = ax.plot(0, 1, marker="o", markersize=10.28)

# calc path vectors
c2_fxn = np.array(c2_pos[1] - c2_pos[0])

# set waypoints in time-s space to follow
s_2 = np.array(s.value).reshape((n, 1))
t_2 = np.arange(0, max_time, time_step).reshape((n, 1))
time_fxn_2 = np.hstack((t_2, s_2))

trajSmooth_1 = TrajectorySmoothing(c1_pos, c1_speed)
trajSmooth_2 = TrajectorySmoothing(c2_pos, c2_speed)
c1_traj = PathTrajectory(trajSmooth_1.get_total_path(time_step), c1_vel, 1.0)
c2_traj = PathTrajectory(trajSmooth_2.get_total_path(time_step), time_fxn_2, 1.0)


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
                    frames=np.linspace(0, max_time, n, endpoint=False))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cars Avoiding Collision')

plt.show()
