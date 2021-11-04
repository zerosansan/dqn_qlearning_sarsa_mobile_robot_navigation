#!/usr/bin/env python

import gym
import numpy
import time
import sarsa
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

    rospy.init_node('example_turtlebot3_maze_sarsa',
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
    outdir = pkg_path + '/testing_results/sarsa'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Remove log file if exist
    gazebo_world_launch_name = env.get_gazebo_world_launch_name()
    utils.remove_logfile_if_exist(outdir, gazebo_world_launch_name)
    # utils.remove_qfile_if_exist(outdir, "sarsa_qtable")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot3/alpha")
    Epsilon = -1  # rospy.get_param("/turtlebot3/epsilon")
    Gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = 10
    nsteps = 200

    running_step = rospy.get_param("/turtlebot3/running_step")

    # Initialises the algorithm that we are going to use for learning
    sarsa = sarsa.Sarsa(actions=range(env.action_space.n),
                        alpha=Alpha, gamma=Gamma, epsilon=Epsilon)

    qtable = sarsa.load_q(outdir + '/sarsa_qtable.txt')
    print("########### SIZE OF Q TABLE: ", len(qtable))
    sarsa.set_q(qtable)

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### WALL START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))
        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            action = sarsa.chooseAction(state)
            observation, reward, done, info = env.step(action)
            success_episode, failure_episode = env.get_episode_status()

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            nextAction = sarsa.chooseAction(nextState)

            if not done:
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                data = [x + 1, success_episode, failure_episode, cumulated_reward, i + 1]
                utils.record_data(data, outdir, gazebo_world_launch_name)
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", i + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)
                break
            rospy.logwarn("############### END Step=>" + str(i))

    env.close()
