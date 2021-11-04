#!/usr/bin/env python
import rospy
import numpy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from turtlebot3_rl.msg import Float

ODOM_TOPIC = '/odom'
DISTANCE_TO_GOAL_TOPIC = '/distance_to_goal'
IS_IN_GOAL_POSITION_TOPIC = '/is_in_goal_position'


class DistanceToGoal:
    def __init__(self):
        # ROS node initialization
        rospy.init_node('get_distance_to_goal', disable_signals=True)
        self.node_name = rospy.get_name()
        rospy.logwarn("%s node started" % self.node_name)

        # Data
        self.distance_to_goal = None

        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")

        self.starting_point = Point()
        self.starting_point.x = rospy.get_param("/turtlebot3/starting_pose/x")
        self.starting_point.y = rospy.get_param("/turtlebot3/starting_pose/y")
        self.starting_point.z = rospy.get_param("/turtlebot3/starting_pose/z")

        self.desired_point_xy_offset = rospy.get_param("/turtlebot3/desired_pose_xy_offset")

        self.in_goal_position = None

        # Publishers
        self.dis_to_goal_pub = rospy.Publisher(DISTANCE_TO_GOAL_TOPIC, Float32, queue_size=1)
        self.in_goal_position_status_pub = rospy.Publisher(IS_IN_GOAL_POSITION_TOPIC, Bool, queue_size=1)

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
            Starts publishing the distance to goal and "reached goal" status message
            when distance to goal has been computed and available.
        """
        if self.distance_to_goal is not None:
            self.dis_to_goal_pub.publish(self.distance_to_goal)
            self.in_goal_position_status_pub.publish(self.in_goal_position)

    def odometry_callback(self, msg):
        """
            Stores the odom and distance to goal messages after
            computing the distance to goal position.
        """
        self.odom = msg

        self.get_distance_from_goal_position(self.odom.pose.pose.position)

    def is_in_goal_position(self, current_position):
        """
            Check if robot is in goal position and return true if it is and False if it isn't.
        """
        self.in_goal_position = False

        x_pos_plus = self.desired_point.x + self.desired_point_xy_offset
        x_pos_minus = self.desired_point.x - self.desired_point_xy_offset
        y_pos_plus = self.desired_point.y + self.desired_point_xy_offset
        y_pos_minus = self.desired_point.y - self.desired_point_xy_offset

        x_current = current_position.x + self.starting_point.x
        y_current = current_position.y + self.starting_point.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        # True if inside goal position region
        self.in_goal_position = x_pos_are_close and y_pos_are_close

    def get_distance_from_goal_position(self, current_position):
        """
            Get the distance to goal position and also check if the robot has reached the
            goal position or not.
        """
        distance = self.get_distance_from_point(current_position, self.desired_point)

        self.distance_to_goal = distance

        # Check if robot has reached goal position
        self.is_in_goal_position(current_position)

    def get_distance_from_point(self, p_start, p_end):
        """
            Computes the distance to goal position.
        """
        a = numpy.array((p_start.x + self.starting_point.x, p_start.y + self.starting_point.y,
                         p_start.z))  # robot starts from -0.7, -0.7
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

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
                self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
                rospy.loginfo("%s: Current /odom is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /odom not ready yet, retrying...", self.node_name)

        return self.odom


if __name__ == '__main__':
    try:
        dtg = DistanceToGoal()
        dtg.check_if_subscribed_topics_ready()
        dtg.spin()

    except rospy.ROSInterruptException:
        pass
