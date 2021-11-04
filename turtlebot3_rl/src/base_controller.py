#!/usr/bin/env python
import signal

import rospy
import time
import utils
import os
import rospkg
import numpy as np
import math

from math import pi
from std_msgs.msg import Bool
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from tf.transformations import euler_from_quaternion

ODOM_TOPIC = '/odom'
MODEL_PREDICTION_TOPIC = '/model_predictions'
COLLISION_STATUS_TOPIC = '/collision_status'
IS_IN_GOAL_POSITION_TOPIC = '/is_in_goal_position'
DISTANCE_TO_GOAL_TOPIC = '/distance_to_goal'
HEADING_TO_GOAL_TOPIC = '/heading_to_goal'


class BaseController:
    def __init__(self):
        # ROS node initialization
        rospy.init_node('base_controller', disable_signals=True)
        self.node_name = rospy.get_name()
        rospy.logwarn("%s node started" % self.node_name)

        # Params
        self.linear_forward_speed = rospy.get_param('/turtlebot3/velocity/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/velocity/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/velocity/angular_speed')

        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")

        self.starting_point = Point()
        self.starting_point.x = rospy.get_param("/turtlebot3/starting_pose/x")
        self.starting_point.y = rospy.get_param("/turtlebot3/starting_pose/y")
        self.starting_point.z = rospy.get_param("/turtlebot3/starting_pose/z")

        # Data
        self.model_prediction = None
        self.robot_collision_status = None
        self.robot_reach_goal_status = None

        self.prev_distance_to_goal = self.get_distance_from_goal_position(self.starting_point, self.desired_point)
        self.current_distance_to_goal = None

        self.prev_heading_to_goal = self.get_heading_from_goal_position(self.starting_point, self.desired_point)
        self.current_heading_to_goal = None

        # Logging
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('turtlebot3_rl')
        self.outdir = self.pkg_path + '/src/results'
        self.run_name = rospy.get_param('/turtlebot3/run_name')
        self.cumulated_reward = 0
        self.cumulated_step = 0
        self.success_run = None
        self.failure_run = None

        # Rewards
        self.forward_reward = rospy.get_param('/turtlebot3/reward/move_forward')
        self.turn_reward = rospy.get_param('/turtlebot3/reward/turn')
        self.goal_reward = rospy.get_param('/turtlebot3/reward/reach_goal')

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Subscribers
        rospy.Subscriber(MODEL_PREDICTION_TOPIC, String, self.model_predictor_callback)
        rospy.Subscriber(COLLISION_STATUS_TOPIC, Bool, self.robot_collision_callback)
        rospy.Subscriber(IS_IN_GOAL_POSITION_TOPIC, Bool, self.robot_reach_goal_callback)
        rospy.Subscriber(DISTANCE_TO_GOAL_TOPIC, Float32, self.distance_to_goal_callback)
        rospy.Subscriber(HEADING_TO_GOAL_TOPIC, Float32, self.heading_to_goal_callback)

    def spin(self):
        try:
            r = rospy.Rate(10)

            while not rospy.is_shutdown():
                try:
                    self.main()
                    r.sleep()
                except KeyboardInterrupt:
                    self.stop_robot("Ctrl-C Keyboard Interrupted")
                    break
        except rospy.ROSInterruptException:
            pass

    def main(self):
        """
            Starts giving drive commands to robot after receiving the model's predictions (actions).
        """
        if self.model_prediction is not None:
            self.give_drive_commands(self.model_prediction)

    def model_predictor_callback(self, msg):
        """
            Stores model prediction message data:
            String message:
                "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"
        """
        self.model_prediction = msg.data

    def robot_collision_callback(self, msg):
        """
            Stores robot collision status message data:
            Bool message:
                True if robot has collided.
                False if robot hasn't collided.
        """
        self.robot_collision_status = msg.data

    def robot_reach_goal_callback(self, msg):
        """
            Stores reach goal states message data:
            Bool message:
                True if robot has reached goal position.
                False if robot hasn't reached goal position.
        """
        self.robot_reach_goal_status = msg.data

    def distance_to_goal_callback(self, msg):
        """
            Stores distance to goal position message data.
        """
        _distance_to_goal = round(msg.data, 2)
        self.current_distance_to_goal = _distance_to_goal

    def heading_to_goal_callback(self, msg):
        """
            Stores heading to goal position message data.
        """
        _heading_to_goal = round(msg.data, 2)
        self.current_heading_to_goal = _heading_to_goal

    def give_drive_commands(self, prediction):
        """
            Converts model predictions into drive commands to move the robot.
        """

        rospy.loginfo("Starting action: " + str(prediction))

        linear_speed, angular_speed = 0, 0

        if prediction == "MOVE_FORWARD":
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
        elif prediction == "TURN_LEFT":
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
        elif prediction == "TURN_RIGHT":
            linear_speed = self.linear_turn_speed
            angular_speed = -1 * self.angular_speed

        # Send drive commands to Turtlebot3
        self.move_base(linear_speed, angular_speed, prediction)

        rospy.loginfo("Ending action: " + str(prediction))

    def move_base(self, linear_speed, angular_speed, prediction=None):
        """
            Sends velocity commands to the robot base to move the robot.
            Deployment behavior is also implemented in this function:
                To stop the robot when it has reached the goal position or collided with an obstacle.
                To reward or punish the robot for the actions or circumstances it is in during deployment.
                To log data for performance comparison purposes:
                    Logged data: Cumulated rewards, cumulated steps, success/failure navigation

        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self._check_publishers_connection()
        self.cmd_vel_pub.publish(cmd_vel_value)

        # If distance to goal has decreased
        distance_difference = self.current_distance_to_goal - self.prev_distance_to_goal
        heading_difference = self.current_heading_to_goal - self.prev_heading_to_goal

        if distance_difference < 0.0:
            rospy.loginfo("DISTANCE_DIFFERENCE: " + str(distance_difference))
            self.cumulated_reward += self.forward_reward

        elif heading_difference < 0.0:
            rospy.loginfo("HEADING_DIFFERENCE: " + str(heading_difference))
            self.cumulated_reward += self.turn_reward

        # If reach goal
        elif self.robot_reach_goal_status is True:
            self.cumulated_reward += self.goal_reward
            self.success_run = True
            self.failure_run = False
            self.log_data()
            self.stop_robot("Robot has reached goal location.")
            return

        # If obstacle collision
        elif self.robot_collision_status is True:
            self.cumulated_reward += -self.goal_reward
            self.success_run = False
            self.failure_run = True
            self.log_data()
            self.stop_robot("Robot has collided with an obstacle.")
            return

        # Give reward for successful drive commands
        if prediction == "MOVE_FORWARD":
            self.cumulated_reward += self.forward_reward
        elif prediction == "TURN_LEFT":
            self.cumulated_reward += self.turn_reward
        elif prediction == "TURN_RIGHT":
            self.cumulated_reward += self.turn_reward

        # Add a step
        self.cumulated_step += 1
        self.prev_distance_to_goal = self.current_distance_to_goal
        self.prev_heading_to_goal = self.current_heading_to_goal

        time.sleep(0.2)

    def check_if_subscribed_topics_ready(self):
        """
            Check if all subscribed topics are ready to receive/send messages
        """
        self._check_model_prediction_ready()
        self._check_robot_collision_status_ready()
        self._check_reach_goal_status_ready()
        self._check_distance_to_goal_ready()
        self._check_heading_to_goal_ready()

    def _check_model_prediction_ready(self):
        """
            Check if subscribed topics are ready to receive/send messages
        """
        self.model_prediction = None
        while self.model_prediction is None and not rospy.is_shutdown():
            try:
                self.model_prediction = rospy.wait_for_message(MODEL_PREDICTION_TOPIC, String, timeout=5.0)
                rospy.loginfo("%s: Current /model_prediction is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /model_prediction not ready yet, retrying...", self.node_name)

        return self.model_prediction

    def _check_robot_collision_status_ready(self):
        """
            Check if subscribed topics are ready to receive/send messages
        """
        self.robot_collision_status = None
        while self.robot_collision_status is None and not rospy.is_shutdown():
            try:
                self.robot_collision_status = rospy.wait_for_message(COLLISION_STATUS_TOPIC, String, timeout=5.0)
                rospy.loginfo("%s: Current /collision_status is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /collision_status not ready yet, retrying...", self.node_name)

        return self.robot_collision_status

    def _check_reach_goal_status_ready(self):
        """
            Check if subscribed topics are ready to receive/send messages
        """
        self.robot_reach_goal_status = None
        while self.robot_reach_goal_status is None and not rospy.is_shutdown():
            try:
                self.robot_reach_goal_status = rospy.wait_for_message(IS_IN_GOAL_POSITION_TOPIC, String, timeout=5.0)
                rospy.loginfo("%s: Current /is_in_goal_position is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /is_in_goal_position not ready yet, retrying...", self.node_name)

        return self.robot_reach_goal_status

    def _check_distance_to_goal_ready(self):
        """
            Check if subscribed topics are ready to receive/send messages
        """
        self.distance_to_goal = None
        while self.distance_to_goal is None and not rospy.is_shutdown():
            try:
                self.distance_to_goal = rospy.wait_for_message(DISTANCE_TO_GOAL_TOPIC, Float32, timeout=5.0)
                rospy.loginfo("%s: Current /distance_to_goal is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /distance_to_goal not ready yet, retrying...", self.node_name)

        return self.distance_to_goal

    def _check_heading_to_goal_ready(self):
        """
            Check if subscribed topics are ready to receive/send messages
        """
        self.heading_to_goal = None
        while self.heading_to_goal is None and not rospy.is_shutdown():
            try:
                self.heading_to_goal = rospy.wait_for_message(HEADING_TO_GOAL_TOPIC, Float32, timeout=5.0)
                rospy.loginfo("%s: Current /heading_to_goal is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /heading_to_goal not ready yet, retrying...", self.node_name)

        return self.heading_to_goal

    def _check_publishers_connection(self):
        """
            Checks that all the publishers are working.
        """
        rate = rospy.Rate(10)  # 10hz
        while self.cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.loginfo("No susbribers to cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass

    def stop_robot(self, reason):
        """
            Stops robot from moving after deployment has ended.
        """
        self.move_base(0.0, 0.0)
        rospy.signal_shutdown(reason)

    def log_data(self):
        """
            Logs data for numerical comparison purposes.
        """
        _logged_data = [0, self.success_run, self.failure_run, self.cumulated_reward, self.cumulated_step]
        utils.record_data(_logged_data, self.outdir, self.run_name)

    def get_distance_from_goal_position(self, p_start, p_end):
        """
            Computes distance between robot and goal position.
        """
        a = np.array((p_start.x + self.starting_point.x, p_start.y + self.starting_point.y,
                      p_start.z))  # robot starts from -0.7, -0.7
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

    def get_heading_from_goal_position(self, current_position, goal_position):
        """
            Computes heading angle between robot and goal position.
        """
        current_pos_x = current_position.x + self.starting_point.x
        current_pos_y = current_position.y + self.starting_point.y

        yaw = self.get_angle_from_starting_point()

        goal_angle = math.atan2(goal_position.y - current_pos_y, goal_position.x - current_pos_x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return heading

    def get_angle_from_starting_point(self):
        """
            Gets yaw angle from starting point
        """
        orientation_x = 0.0
        orientation_y = 0.0
        orientation_z = 0.0
        orientation_w = 1.0

        orientation_list = [orientation_x, orientation_y, orientation_z, orientation_w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        return yaw


if __name__ == '__main__':
    try:
        bc = BaseController()
        bc.check_if_subscribed_topics_ready()
        bc.spin()

    except rospy.ROSInterruptException:
        pass
