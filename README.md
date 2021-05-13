# ISP2021-multi_vehicle_coordination
This is the github project for the Multi-Vehicle Coordination F1Tenth Independent Study Project.  In this project, our development is focused on building a scalable, lightweight framework capable of generating time-parameterized, coordinated overtaking maneuvers which obey vehicular dynamic constraints. Our proposed algorithm utilizes a central node responsible for path planning using RRT* and for calculating velocity-tuned paths using a decoupled configuration space for collision avoidance. After processing the RRT* generated path using minimum jerk trajectory smoothing, the paths are sent to each vehicle and executed using the Pure Pursuit algorithm.

Currently, this package supports multi-vehicle overtaking for 3 vehicles, but future development is focused on scaling to N vehicles. 

## Requirements
- Linux Ubuntu (tested on version 18.04)
- Python 3.8

## Dependencies
- F1/Tenth Gym Simulator (this package currently relies on a fork of the f1tenthgym at: "https://github.com/wesleyyee1996/f1tenth_gym" using the "waypoint_vis" branch for rendering purposes. However, a pull request will be made from this fork into the official f1tenthgym repo in the future)
- ZeroMQ 4.2.2 (https://zeromq.org/languages/python/)
- CVXPY (Convex optimization solver), available at cvxpy.org
- NumPy
- MatPlotLib

## Installation
-  Use the provided `requirements.txt` in the root directory of this repo, in order to install all required modules.\
`pip3 install -r /path/to/requirements.txt`
- The f1tenthgym dependency will need to be cloned manually.

The code is developed with Python 3.8.

## Running the code
* `Step 1:` cd to "src/Map_Levine_velocity_tuning_3_cars"
* `Step 2:` Open the "config_example_map.yaml" file and update the starting positions and orientations for Car 1, 2, and 3. Ensure that vehicle positions are at least 1.5 meters apart. Doing so otherwise will result in collisions!
* `Step 3:` In one terminal window, run "python3 cars.py"
* `Step 4:` In another terminal window, run "python3 master.py"

You should observe the F1Tenth Gym simulation starting with the vehicles displayed on the screen. Upon startup, the cars node will update the master node with their starting positions. The master node will then proceed to plan a coordinated path for all vehicles and when complete, send back paths to all vehicles.

## Folder Structure

All main scripts are co-dependent upon other scripts within the "Map_Levine_velocity_tuning_3_cars" folder.


## Files
| File | Description |
|----|----|
master.py   | The central server through which all coordinated path planning occurs. Receives states from car nodes and returns paths for each car.
cars.py | The actual vehicles, which are treated as client nodes. These execute the F1Tenth Gym simulator.
PathTrajectory.py | Class which stores all information pertaining to a particular path through space (positions and time-parameterizations)
VelocityTuner.py | Class which uses optimizer to plan coordinated paths based on decoupled approach
TrajectorySmoothing.py | Class which smooths RRT* generated path into a continuous minimum jerk path of constant velocity
config_example_map.yaml | Configuration file to modify F1Tenth Gym simulation parameters

