# Comparison of DQN with Q-Learning and SARSA for Robot Local Navigation

Source code for published paper on Comparison of Deep Q-Learning, Q-Learning and SARSA Reinforced Learning for Robot Local Navigation.

## Installation

- Firstly, the following packages ([openai_ros](https://bitbucket.org/theconstructcore/openai_ros.git), [turtlebot3](http://wiki.ros.org/turtlebot3)) and their dependencies should be cloned in your ROS workspace.
- Then, clone this repository and move the contents openai_ros and turtlebot3 to the installed packages.
- Finally, the ROS workspace should be compiled with `catkin_make` and sourced with `source devel/setup.bash`. The compile process should return no error if all the dependencies are met. 

## Repository contents

**my_turtlebot3_training** - This folder contains files for the robot to run Deep Q-Network, Q-Learning, and Sarsa algorithm for training and testing.

**openai_projects** - This folder ontains the RL trained models and logged data during training and testing.

**openai_ros** - This folder contains files for setting up the RL environment.

**turtlebot3** - This folder contains the Gazebo simulation launch files, models, and worlds.

**turtlebot3_rl** - This folder contains the codes for deploying Deep Q-Network, Q-Learning, and Sarsa algorithm in the real environment with the physical Turtlebot3.

## Example usage

**Simulation training**

Run `roslaunch my_turtlebot3_training start_dqlearn_training.launch` in your terminal.

**Simulation testing** 

Run `roslaunch my_turtlebot3_training start_dqlearn_testing.launch` in your terminal.

**Real world testing (deployment)** 

Physical deployment requires the Turtlebot3 itself and a remote PC to run.

On the Turtlebot3:
- Run `roslaunch turtlebot3_bringup turtlebot3_robot.launch`

On the remote PC:
- Run `roscore`
- Run `roslaunch turtlebot3_bringup turtlebot3_remote.launch`
- Run `roslaunch turtlebot3_rl deploy_turtlebot3.launch`

