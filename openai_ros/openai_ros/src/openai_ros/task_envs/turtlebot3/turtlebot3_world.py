import math

import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
from math import pi

import time
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion

# Set the logging system
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('turtle3_openai_ros_example')
outdir = pkg_path + '/training_results/dqlearn'

GAZEBO_WORLD_LAUNCH_NAME = "turtlebot3_stage_2"


class TurtleBot3WorldEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL " \
                                           "script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\' "
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name=GAZEBO_WORLD_LAUNCH_NAME + ".launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3/config",
                               yaml_file_name="turtlebot3_world.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        num_laser_readings = int(len(laser_scan.ranges) / self.new_ranges)
        high = numpy.full(num_laser_readings, self.max_laser_value)
        low = numpy.full(num_laser_readings, self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")

        self.laser_filtered_pub = rospy.Publisher(
            '/turtlebot3/laser/scan_filtered', LaserScan, queue_size=1)

        self.episode_failure = False
        self.episode_success = False

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=0.05,
                       update_rate=10)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are
        # sluggish and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

        laser_scan = self.get_laser_scan()
        discretized_ranges = laser_scan.ranges
        self.publish_filtered_laser_scan(laser_original_data=laser_scan,
                                         new_filtered_laser_range=discretized_ranges)

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)
        self.previous_angle_from_des_point = self.get_heading_from_desired_point(odometry.pose.pose.position,
                                                                                 odometry.pose.pose.orientation)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>" + str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0:  # FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1:  # LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2:  # RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1 * self.angular_speed
            self.last_action = "TURN_RIGHT"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>" + str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(laser_scan, self.new_ranges)

        # We get the odometry so that SumitXL knows where it is.
        odometry = self.get_odom()
        odometry.pose.pose.position.y

        # We round to only two decimals to avoid very big Observation space
        distance_to_des_point = round(self.previous_distance_from_des_point, 2)
        angle_to_des_point = round(self.previous_angle_from_des_point, 2)

        observations = discretized_observations + [distance_to_des_point, angle_to_des_point]

        rospy.logdebug("Observations==>" + str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    def _get_episode_success_failure_status(self):

        episode_outcome = [self.episode_success, self.episode_failure]

        return episode_outcome

    def _get_gazebo_world_launch_name(self):
        return GAZEBO_WORLD_LAUNCH_NAME

    def _is_done(self):

        if self._episode_done:
            self.episode_failure = True
            self.episode_success = False
            rospy.logerr("TurtleBot2 is Too Close to wall==>")
        else:
            rospy.logwarn("TurtleBot2 is NOT close to a wall ==>")
        odometry = self.get_odom()
        current_position = Point()
        current_position.x = odometry.pose.pose.position.x
        current_position.y = odometry.pose.pose.position.y
        current_position.z = 0.0

        current_orientation = Quaternion()
        current_orientation.x = odometry.pose.pose.orientation.x
        current_orientation.y = odometry.pose.pose.orientation.y
        current_orientation.z = odometry.pose.pose.orientation.z
        current_orientation.w = odometry.pose.pose.orientation.w

        MAX_X = 9.0
        MIN_X = -9.0
        MAX_Y = 9.0
        MIN_Y = -9.0

        # We see if we are outside the Learning Space
        if MAX_X >= current_position.x > MIN_X:
            if MAX_Y >= current_position.y > MIN_Y:
                rospy.logdebug(
                    "TurtleBot Position is OK ==>[" + str(current_position.x) + "," + str(current_position.y) + "]")

                # We see if it got to the desired point
                if self.is_in_desired_position(current_position):
                    self.episode_failure = False
                    self.episode_success = True
                    self._episode_done = True

            else:
                rospy.logerr("TurtleBot to Far in Y Pos ==>" + str(current_position.x))
                self._episode_done = True
        else:
            rospy.logerr("TurtleBot to Far in X Pos ==>" + str(current_position.x))
            self._episode_done = True

        return self._episode_done

    def _compute_reward(self, done):
        odometry = self.get_odom()
        current_position = Point()
        current_position.x = odometry.pose.pose.position.x
        current_position.y = odometry.pose.pose.position.y
        current_position.z = 0.0

        current_orientation = Quaternion()
        current_orientation.x = odometry.pose.pose.orientation.x
        current_orientation.y = odometry.pose.pose.orientation.y
        current_orientation.z = odometry.pose.pose.orientation.z
        current_orientation.w = odometry.pose.pose.orientation.w

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point

        angle_from_des_point = self.get_heading_from_desired_point(current_position, current_orientation)
        angle_difference = angle_from_des_point - self.previous_angle_from_des_point

        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward

            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward += self.forwards_reward
            else:
                rospy.logerr("INCREASE IN DISTANCE BAD")
                reward += 0

            # If there has been a decrease in the angle to the desired point, we reward it
            if angle_difference < 0.0:
                rospy.logwarn("DECREASE IN ANGLE GOOD")
                reward += self.turn_reward
            else:
                rospy.logerr("INCREASE IN ANGLE BAD")
                reward += 0
        else:
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
            else:
                reward = -1 * self.end_episode_points

        self.previous_distance_from_des_point = distance_from_des_point
        self.previous_angle_from_des_point = angle_from_des_point

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods
    def discretize_scan_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges) / new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if i % mod == 0:
                if item == float('Inf') or numpy.isinf(item):
                    discretized_ranges.append(round(self.max_laser_value, 2))
                elif numpy.isnan(item):
                    discretized_ranges.append(round(self.max_laser_value, 2))
                else:
                    discretized_ranges.append(round(item, 2))

                if self.min_range > item > 0:
                    rospy.logerr("done Validation >>> item=" + str(item) + "< " + str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item) + "< " + str(self.min_range))

        self.publish_filtered_laser_scan(laser_original_data=data,
                                         new_filtered_laser_range=discretized_ranges)

        return discretized_ranges

    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):

        rospy.logdebug("new_filtered_laser_range==>" +
                       str(new_filtered_laser_range))

        laser_filtered_object = LaserScan()

        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = laser_original_data.header.frame_id

        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max

        new_angle_incr = abs(laser_original_data.angle_max -
                             laser_original_data.angle_min) / len(new_filtered_laser_range)

        laser_filtered_object.angle_increment = new_angle_incr
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max

        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            if item == 0.0:
                laser_distance = 0.1
            else:
                laser_distance = item
            laser_filtered_object.ranges.append(laser_distance)
            laser_filtered_object.intensities.append(item)

        self.laser_filtered_pub.publish(laser_filtered_object)

    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

    def is_in_desired_position(self, current_position, epsilon=0.20):  # originally 0.05, changed to 0.20
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos

    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param current_position:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :param pstart:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def is_in_desired_angle(self, heading_angle, epsilon=math.radians(10)):
        """
        It return True if the current orientation is similar to the desired position
        """
        is_in_desired_angle = False

        theta_positive = epsilon
        theta_negative = -epsilon

        theta_current = heading_angle

        theta_are_close = (theta_current <= theta_positive) and (theta_current > theta_negative)

        is_in_desired_angle = theta_are_close

        return is_in_desired_angle

    def get_heading_from_desired_point(self, current_position, current_orientation):
        """
        Calculates the angle from the current position to the desired position
        :param current_position:
        :return:
        """
        current_pos_x = current_position.x
        current_pos_y = current_position.y

        yaw = self.get_angle_from_point(current_orientation)

        goal_angle = math.atan2(self.desired_point.y - current_pos_y, self.desired_point.x - current_pos_x)

        _heading = goal_angle - yaw
        if _heading > pi:
            _heading -= 2 * pi

        elif _heading < -pi:
            _heading += 2 * pi

        return _heading

    def get_angle_from_point(self, current_orientation):
        """
        Given a Vector3 Object, get angle from current position
        :param current_orientation:
        :return:
        """
        current_ori_x = current_orientation.x
        current_ori_y = current_orientation.y
        current_ori_z = current_orientation.z
        current_ori_w = current_orientation.w

        orientation_list = [current_ori_x, current_ori_y, current_ori_z, current_ori_w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        return yaw
