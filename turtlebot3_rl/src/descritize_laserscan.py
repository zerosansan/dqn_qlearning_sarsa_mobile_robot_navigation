#!/usr/bin/env python
import rospy
import numpy as np

from std_msgs.msg import Bool
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan

LASERSCAN_TOPIC = '/scan'
DESCRITIZED_LASERSCAN_TOPIC = '/descritized_scan'
COLLISION_STATUS_TOPIC = '/collision_status'


class DescritizeLaserScan:
    def __init__(self):
        # ROS node initialization
        rospy.init_node('descritize_laserscan', disable_signals=True)
        self.node_name = rospy.get_name()
        rospy.logwarn("%s node started" % self.node_name)

        # Data
        self.laserscan = None
        self.new_ranges = 50
        self.scan_ranges = []
        self.scan_min_range = None
        self.scan_max_range = None
        self.des_laserscan_object = None

        self.min_range = rospy.get_param('/turtlebot3/laser_scan/min_range')

        self.collision_status = None

        # Publishers
        self.des_laserscan_pub = rospy.Publisher(DESCRITIZED_LASERSCAN_TOPIC, LaserScan, queue_size=1)
        self.col_status_pub = rospy.Publisher(COLLISION_STATUS_TOPIC, Bool, queue_size=1)

        # Subscribers
        rospy.Subscriber(LASERSCAN_TOPIC, LaserScan, self.laserscan_callback)

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
            Starts publishing the discretized laserscan and collision status message
            when discretized object has been computed and available.
        """
        if self.des_laserscan_object is not None:
            self.des_laserscan_pub.publish(self.des_laserscan_object)
            self.col_status_pub.publish(self.collision_status)

    def laserscan_callback(self, msg):
        """
            Stores laserscan related messages and performs discretization.
        """
        self.laserscan = msg
        self.scan_ranges = msg.ranges
        self.scan_max_range = msg.range_max
        self.scan_min_range = msg.range_min

        self.descritize_laserscan(msg, self.new_ranges)

    def descritize_laserscan(self, data, new_ranges):
        """
            Discretizes the laserscan from the default 360 scans to X number of scans.
        """
        self.collision_status = False
        descritized_ranges = []
        mod = len(data.ranges) / new_ranges

        max_laser_value = data.range_max
        min_laser_value = data.range_min

        for i, item in enumerate(data.ranges):
            if i % mod == 0:
                if item == float('Inf') or np.isinf(item):
                    descritized_ranges.append(round(max_laser_value, 1))
                elif np.isnan(item):
                    descritized_ranges.append(round(min_laser_value, 1))
                else:
                    descritized_ranges.append(round(item, 1))

                # Check if robot is too close to obstacle
                if self.min_range > item > 0:
                    self.collision_status = True

        self.create_filtered_laser_scan_object(original_scan=data, new_scan=descritized_ranges)

    def create_filtered_laser_scan_object(self, original_scan, new_scan):
        """
            Creates a laserscan object for discretization.
        """
        laser_descritized_object = LaserScan()

        h = Header()
        # Note you need to call rospy.init_node() before this will work
        h.stamp = rospy.Time.now()
        h.frame_id = original_scan.header.frame_id

        laser_descritized_object.header = h
        laser_descritized_object.angle_min = original_scan.angle_min
        laser_descritized_object.angle_max = original_scan.angle_max

        new_angle_incr = abs(original_scan.angle_max -
                             original_scan.angle_min) / len(new_scan)

        # laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_descritized_object.angle_increment = new_angle_incr
        laser_descritized_object.time_increment = original_scan.time_increment
        laser_descritized_object.scan_time = original_scan.scan_time
        laser_descritized_object.range_min = original_scan.range_min
        laser_descritized_object.range_max = original_scan.range_max

        laser_descritized_object.ranges = []
        laser_descritized_object.intensities = []
        for item in new_scan:
            if item == 0.0:
                laser_distance = 0.1
            else:
                laser_distance = item
            laser_descritized_object.ranges.append(laser_distance)
            laser_descritized_object.intensities.append(item)

        self.des_laserscan_object = laser_descritized_object

    def check_if_subscribed_topics_ready(self):
        """
            Check if all subscribed topics are ready to receive/send messages
        """
        self._check_laserscan_ready()

    def _check_laserscan_ready(self):
        """
            Check if subscribed topics are ready to receive/send messages
        """
        self.laserscan = None
        while self.laserscan is None and not rospy.is_shutdown():
            try:
                self.laserscan = rospy.wait_for_message(LASERSCAN_TOPIC, LaserScan, timeout=5.0)
                rospy.loginfo("%s: Current /scan is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /scan not ready yet, retrying...", self.node_name)

        return self.laserscan


if __name__ == '__main__':
    try:
        dls = DescritizeLaserScan()
        dls.check_if_subscribed_topics_ready()
        dls.spin()

    except rospy.ROSInterruptException:
        pass
