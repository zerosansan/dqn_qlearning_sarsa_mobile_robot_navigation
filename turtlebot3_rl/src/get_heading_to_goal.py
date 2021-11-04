#!/usr/bin/env python
import rospy
import math

from math import pi
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from turtlebot3_rl.msg import Float

from tf.transformations import euler_from_quaternion

ODOM_TOPIC = '/odom'
HEADING_TO_GOAL_TOPIC = '/heading_to_goal'


class HeadingToGoal:
    def __init__(self):
        # ROS node initialization
        rospy.init_node('get_heading_to_goal', disable_signals=True)
        self.node_name = rospy.get_name()
        rospy.logwarn("%s node started" % self.node_name)

        # Data
        self.heading_to_goal = None

        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")

        self.starting_point = Point()
        self.starting_point.x = rospy.get_param("/turtlebot3/starting_pose/x")
        self.starting_point.y = rospy.get_param("/turtlebot3/starting_pose/y")
        self.starting_point.z = rospy.get_param("/turtlebot3/starting_pose/z")

        # Publishers
        self.heading_to_goal_pub = rospy.Publisher(HEADING_TO_GOAL_TOPIC, Float32, queue_size=1)

        # Subscribers
        rospy.Subscriber(ODOM_TOPIC, Odometry, self.odometry_callback)

    def spin(self):
        try:
            r = rospy.Rate(10)

            while not rospy.is_shutdown():
                try:
                    self.main()
                    r.sleep()
                except KeyboardInterrupt:
                    break
        except rospy.ROSInterruptException:
            pass

    def main(self):
        """
            Starts publishing the heading to goal status message
            when heading to goal has been computed and available.
        """
        if self.heading_to_goal is not None:
            self.heading_to_goal_pub.publish(self.heading_to_goal)

    def odometry_callback(self, msg):
        """
            Stores the odom and heading to goal messages after
            computing the heading to goal position.
        """
        self.odom = msg

        self.get_heading_from_goal_position(self.odom.pose.pose.position, self.odom.pose.pose.orientation)

    def get_heading_from_goal_position(self, current_position, current_orientation):
        """
            Computes the heading angle to goal position.
        """
        current_pos_x = current_position.x + self.starting_point.x
        current_pos_y = current_position.y + self.starting_point.y

        yaw = self.get_angle_from_point(current_orientation)

        goal_angle = math.atan2(self.desired_point.y - current_pos_y, self.desired_point.x - current_pos_x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading_to_goal = heading

    def get_angle_from_point(self, current_orientation):
        """
            Gets the yaw angle from the starting to the goal position.
        """
        current_ori_x = current_orientation.x
        current_ori_y = current_orientation.y
        current_ori_z = current_orientation.z
        current_ori_w = current_orientation.w

        orientation_list = [current_ori_x, current_ori_y, current_ori_z, current_ori_w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        return yaw

    def check_if_subscribed_topics_ready(self):
        """
            Check if all subscribed topics are ready to receive/send messages
        """
        self._check_odom_ready()

    def _check_odom_ready(self):
        """
            Check if subscribed topics are ready to receive/send messages
        """
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=5.0)
                rospy.loginfo("%s: Current /odom is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /odom not ready yet, retrying...", self.node_name)

        return self.odom


if __name__ == '__main__':
    try:
        htg = HeadingToGoal()
        htg.check_if_subscribed_topics_ready()
        htg.spin()

    except rospy.ROSInterruptException:
        pass
