import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
ax.axis([0, 6, 0, 6])
ax.set_aspect("equal")

# set the objects, markersize of 42.28 roughly equals radius = 0.5 w/ [0,6] axis limits
point, = ax.plot(0,1,marker="o", markersize=42.28)
point2, = ax.plot(0,1,marker="o", markersize=42.28)

# set path points
path_points_1 = np.array([[3, 2],
                  [4, 5]])
path_points_2 = np.array([[2, 3],
                  [5, 4]])

# calc path vectors
path_fxn_1 = np.array(path_points_1[1]-path_points_1[0])
path_fxn_2 = np.array(path_points_2[1]-path_points_2[0])

# set waypoints in time-s space to follow
time_fxn_2 = np.array([[0.0,0.0],
                       [0.2, 0.1],
                       [0.4,0.1],
                       [0.6, 0.23],
                       [0.8, 0.42],
                       [0.9, 0.61],
                       [1,1]])

# the animation function for car 1
def line(t):
    return np.array(path_points_1[0] + path_fxn_1*t)

# initialize values for line2 animation function
car2_pos = []
curr_seg = 0
curr_pos = [2,3]

# animation function for car 2
def line2(t):
    global curr_seg
    global curr_pos

    if t > time_fxn_2[curr_seg+1,0]:
        curr_seg += 1

    # calculate speed in real space of car 2
    speed = (time_fxn_2[curr_seg + 1][1]*path_fxn_2 - time_fxn_2[curr_seg][1]*path_fxn_2) / (
                time_fxn_2[curr_seg + 1][0] - time_fxn_2[curr_seg][0])
    curr_pos += speed * 0.01
    out = np.array(curr_pos)
    car2_pos.append([t,out[0]])

    return out

# used for running the animation function
def update(t):

    # get x,y positions of each car at time t
    x,y = line(t)
    list1 = [[x],[y]]
    x,y = line2(t)
    list2 = [[x],[y]]

    point.set_data(list1[0], list1[1])
    point2.set_data(list2[0], list2[1])
    point.set_label('Car 1')
    point2.set_label('Car 2')
    plt.legend(loc="upper right")
    return [point,point2]


ani = FuncAnimation(fig, update, interval=50, repeat=False,
                    frames = np.linspace(0,1,100, endpoint=False))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cars Avoiding Collision')

plt.show()

# car2_pos = np.array(car2_pos)
# plt.figure(2)
# plt.plot(car2_pos[:,0], car2_pos[:,1],'b.')
# plt.xlabel("Time")
# plt.ylabel("X")
# plt.title("Car 2 Velocity Tuned Path")
# plt.show()
#
# plt.figure(3)
# plt.plot(time_fxn_2[:,0], time_fxn_2[:,1],'b.')
# plt.xlabel("Time")
# plt.ylabel("Percent Traveled Along Path")
# plt.title("Car 2 Velocity Tuned Path")
# plt.show()

