# Q-Learning, SARSA and DQN (Deep Q-Network) Reinforcement Learning based mobile robot navigation

This repository contains codes to run a Reinforcement Learning based navigation.

There are three algorithms provided which are Q-Learning, SARSA, and DQN. A Turtlebot3 mobile robot platform was used to train and test these algorithms.

[![Watch the video](https://img.youtube.com/vi/D0UsmpTIG3c/maxresdefault.jpg)](https://youtu.be/D0UsmpTIG3c)

If you have found this repository useful or have used this repository in any of your scientific work, please consider citing my work using this [BibTeX Citation](#bibtex-citation). A full demonstration video of the mobile robot navigation has been uploaded on [Youtube](https://www.youtube.com/watch?v=D0UsmpTIG3c).

## Table of contents

* [Installation](#installation)
* [Repository contents](#repository-contents)
* [Getting started](#getting-started)
* [Hardware and software information](#hardware-and-software-information)
* [BibTeX Citation](#bibtex-citation)
* [Acknowledgments](#acknowledgments)

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

## Getting Started

**Simulation training**

1. Run `roslaunch my_turtlebot3_training start_dqlearn_training.launch` in your terminal.

**Simulation testing** 

2. Run `roslaunch my_turtlebot3_training start_dqlearn_testing.launch` in your terminal.

**Real world testing (deployment)** 

3. Physical deployment requires the Turtlebot3 itself and a remote PC to run.

4. On the Turtlebot3:
- Run `roslaunch turtlebot3_bringup turtlebot3_robot.launch`

5. On the remote PC:
- Run `roscore`
- Run `roslaunch turtlebot3_bringup turtlebot3_remote.launch`
- Run `roslaunch turtlebot3_rl deploy_turtlebot3.launch`

## Hardware and Software Information

**Software**

- OS: Ubuntu 18.04
- ROS version: Melodic
- Python version: 2.7
- Gazebo version: 9
- CUDA version: 10.0
- CuDNN version: 7

**Computer Specifications**

- CPU: Intel i7 9700
- GPU: Nvidia RTX 2070

**Mobile Robot Platform**

- [Turtlebot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)

## BibTeX Citation

If you have used this repository in any of your scientific work, please consider citing my work:

```
@inproceedings{anas2022comparison,
  title={Comparison of Deep Q-Learning, Q-Learning and SARSA Reinforced Learning for Robot Local Navigation},
  author={Anas, Hafiq and Ong, Wee Hong and Malik, Owais Ahmed},
  booktitle={International Conference on Robot Intelligence Technology and Applications},
  pages={443--454},
  year={2022},
  organization={Springer}
}
```

## Acknowledgments

* Thank you [Robolab@UBD](https://ailab.space/) for lending the Turtlebot3 robot platform and lab facilities.

