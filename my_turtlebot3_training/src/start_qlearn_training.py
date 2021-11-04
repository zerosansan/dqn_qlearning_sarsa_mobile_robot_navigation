#!/usr/bin/env python

import gym
import numpy as np
import time
import qlearn
import random
import os
from gym import wrappers

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import utils


def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)


if __name__ == '__main__':

    # Set Gazebo and ROS Master to different ports for running multiple instances
    random_number = random.randint(10000, 15000)
    port_gazebo = random_number + 1  # os.environ["ROS_PORT_SIM"]
    os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + str(port_gazebo)

    rospy.init_node('example_turtlebot3_maze_qlearn',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    max_ep_steps = rospy.get_param("/turtlebot3/nsteps")
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name, max_ep_steps)

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtle3_openai_ros_example')
    outdir = pkg_path + '/training_results/qlearn'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Remove log file if exist
    gazebo_world_launch_name = env.get_gazebo_world_launch_name()
    utils.remove_logfile_if_exist(outdir, gazebo_world_launch_name)
    # utils.remove_qfile_if_exist(outdir, "qlearn_qtable")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot3/alpha")
    Epsilon = rospy.get_param("/turtlebot3/epsilon")
    Gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")

    running_step = rospy.get_param("/turtlebot3/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    qtable = qlearn.load_q(outdir + '/qlearn_qtable.txt')
    print("########### SIZE OF Q TABLE: ", len(qtable))
    qlearn.set_q(qtable)

    start_time = time.time()
    highest_reward = 0

    # Discretization bins
    _distance_bins = [i for i in np.arange(0, 3, 0.1)]  # 30 values max
    distance_bins = [round(i, 2) for i in _distance_bins]

    _radian_bins = [i for i in np.arange(-3.14, 3.14, 0.19625)]  # 32 values max
    radian_bins = [round(i, 2) for i in _radian_bins]

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### WALL START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()

        # Discretize observations
        _lsd_obs = [item for item in observation]  # [:-2]  # Laser Scan Distances (m)
        _dtg_obs = [item for item in observation][-2]  # Distance To Goal (m)
        _htg_obs = [item for item in observation][-1]  # Heading To Goal (rad)

        discretized_lsd_obs = np.digitize(_lsd_obs, distance_bins)
        discretized_dtg_obs = np.digitize([_dtg_obs], distance_bins)
        discretized_htg_obs = np.digitize([_htg_obs], radian_bins)

        discretized_obs = np.concatenate([discretized_lsd_obs, discretized_dtg_obs, discretized_htg_obs])

        state = ''.join(map(str, discretized_obs))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            action = qlearn.chooseAction(state)
            observation, reward, done, info = env.step(action)

            # Discretize observations
            _lsd_obs = [item for item in observation]  # [:-2]  # Laser Scan Distances (m)
            _dtg_obs = [item for item in observation][-2]  # Distance To Goal (m)
            _htg_obs = [item for item in observation][-1]  # Heading To Goal (rad)

            discretized_lsd_obs = np.digitize(_lsd_obs, distance_bins)
            discretized_dtg_obs = np.digitize([_dtg_obs], distance_bins)
            discretized_htg_obs = np.digitize([_htg_obs], radian_bins)

            discretized_obs = np.concatenate([discretized_lsd_obs, discretized_dtg_obs, discretized_htg_obs])

            success_episode, failure_episode = env.get_episode_status()

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, discretized_obs))

            # Make the algorithm learn based on the results
            qlearn.learn(state, action, reward, nextState)

            if not done:
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                if (x + 1) % 100 == 0:
                    # save Q-table
                    qtable = qlearn.get_qtable()
                    qlearn.save_q(qtable, outdir, "qlearn_qtable")
                rospy.logwarn("DONE")
                data = [x + 1, success_episode, failure_episode, cumulated_reward, i + 1]
                utils.record_data(data, outdir, gazebo_world_launch_name)
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", i + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)
                last_time_steps = np.append(last_time_steps, [int(i + 1)])
                break

            rospy.logwarn("############### END Step=>" + str(i))
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
