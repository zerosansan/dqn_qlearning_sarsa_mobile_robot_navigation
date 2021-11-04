#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
from gym import wrappers
import time
import os
import json
import liveplot
import deepq
import rospy
import numpy
import rospkg
import random
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import utils
from distutils.dir_util import copy_tree


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

    rospy.init_node('example_turtlebot3_maze_dqlearn',
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
    outdir = pkg_path + '/testing_results/dqlearn'
    path = pkg_path + '/testing_results/dqlearn/turtle_c2_dqn_ep'
    plotter = liveplot.LivePlot(outdir)
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    # Remove log file if exist
    gazebo_world_launch_name = env.get_gazebo_world_launch_name()
    utils.remove_logfile_if_exist(outdir, gazebo_world_launch_name)

    continue_execution = True
    # fill this if continue_execution=True
    resume_epoch = '1500'  # change to epoch to continue from
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = outdir  # resume_path
    params_json = resume_path + '.json'

    # Load weights, monitor info and parameter info.
    with open(params_json) as outfile:
        d = json.load(outfile)
        epochs = d.get('epochs') + 10
        steps = d.get('steps')
        updateTargetNetwork = d.get('updateTargetNetwork')
        explorationRate = -1  # d.get('explorationRate')
        minibatch_size = d.get('minibatch_size')
        learnStart = d.get('learnStart')
        learningRate = d.get('learningRate')
        discountFactor = d.get('discountFactor')
        memorySize = d.get('memorySize')
        network_inputs = d.get('network_inputs')
        network_outputs = d.get('network_outputs')
        network_structure = d.get('network_structure')
        current_epoch = d.get('current_epoch')

        # added
        epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")  # 0.995

    deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
    deepQ.initNetworks(network_structure, training=True)
    deepQ.loadWeights(weights_path)

    clear_monitor_files(outdir)
    copy_tree(monitor_path, outdir)

    env._max_episode_steps = steps  # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir, force=not continue_execution, resume=continue_execution)

    highest_reward = 0
    start_time = time.time()

    # start iterating from 'current epoch'.
    for epoch in xrange(current_epoch + 1, epochs + 1, 1):
        observation = env.reset()
        _observation = numpy.array(observation)
        done = False
        cumulated_reward = 0
        episode_step = 0

        # run until env returns done
        while not done:
            rospy.logwarn("Episode Steps: " + str(episode_step))
            _observation = numpy.array(_observation)
            qValues = deepQ.getQValues(_observation)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)
            success_episode, failure_episode = env.get_episode_status()

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            _observation = newObservation

            if done:
                data = [epoch, success_episode, failure_episode, cumulated_reward, episode_step + 1]
                utils.record_data(data, outdir, gazebo_world_launch_name)
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", episode_step + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)

            episode_step += 1

    env.close()
