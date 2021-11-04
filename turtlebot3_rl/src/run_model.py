#!/usr/bin/env python
import rospy
import numpy as np
from keras import models
import os
import utils
import rospkg
import random

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from turtlebot3_rl.msg import Float
from std_msgs.msg import Float32

DESCRITIZED_LASERSCAN_TOPIC = '/descritized_scan'
DISTANCE_TO_GOAL_TOPIC = '/distance_to_goal'
HEADING_TO_GOAL_TOPIC = '/heading_to_goal'

MODEL_PREDICTION_TOPIC = '/model_predictions'

MODEL_PARAMS_PATH = os.getcwd() + '/models/turtle_c2_dqn_ep1500.json'
MODEL_WEIGHT_PATH = os.getcwd() + '/models/turtle_c2_dqn_ep1500.h5'


class DeployModel:
    def __init__(self):
        # ROS node initialization
        rospy.init_node('deploy_model', disable_signals=True)
        self.node_name = rospy.get_name()
        rospy.logwarn("%s node started" % self.node_name)

        # Get package directory
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('turtlebot3_rl')
        self.outdir = self.pkg_path + '/src/models'

        # Load .h5 model
        self.dqn_model = models.load_model(MODEL_WEIGHT_PATH)
        self.qlearn_qtable = utils.load_q(self.outdir + "/qlearn_qtable_final.txt")
        self.sarsa_qtable = utils.load_q(self.outdir + "/sarsa_qtable_final.txt")

        # Data
        self.laserscan = None
        self.distance_to_goal = None
        self.heading_to_goal = None

        self.scan_ranges_np = []
        self.distance_to_goal_np = []
        self.heading_to_goal_np = []
        self.action_taken = None

        # Publishers
        self.model_prediction_pub = rospy.Publisher(MODEL_PREDICTION_TOPIC, String, queue_size=1)

        # Subscribers
        rospy.Subscriber(DESCRITIZED_LASERSCAN_TOPIC, LaserScan, self.laserscan_callback)
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
                    break

        except rospy.ROSInterruptException:
            pass

    def main(self):
        """
            Starts the model prediction based on the robot's observation inputs and publishes
            the translated model output (i.e actions)
        """
        model_input = self.scan_ranges_np + self.distance_to_goal_np + self.heading_to_goal_np
        self.run_model(self.dqn_model, model_input)
        self.model_prediction_pub.publish(self.action_taken)

    def laserscan_callback(self, msg):
        """
            Stores laserscan related messages from the discretized laserscan topics.
        """
        self.laserscan = msg
        _scan_ranges = msg.ranges
        scan_ranges = np.around(np.array(_scan_ranges), 2)
        self.scan_ranges_np = list(scan_ranges)

    def distance_to_goal_callback(self, msg):
        """
            Stores the distance to goal messages.
        """
        self.distance_to_goal = msg.data
        self.distance_to_goal_np = [round(self.distance_to_goal, 2)]

    def heading_to_goal_callback(self, msg):
        """
            Stores the heading to goal messages.
        """
        self.heading_to_goal = msg.data
        self.heading_to_goal_np = [round(self.heading_to_goal, 2)]

    def run_model(self, model, model_input):
        """
            :param model: Deep Learning Keras model
            :param model_input: numpy array of scan ranges
            :return: a List of Q values corresponding to different actions
        """
        if model == self.dqn_model:
            state = self._reshape_for_model_input(model_input)
            _output = model.predict(state)
            self.action_taken = self._get_best_action(_output, model)
        if model == self.qlearn_qtable:
            state = ''.join(map(str, model_input))
            _output = [utils.get_q(model, state, a) for a in [0, 1, 2]]
            print(_output)
            self.action_taken = self._get_best_action(_output, model)
        if model == self.sarsa_qtable:
            state = ''.join(map(str, model_input))
            _output = [utils.get_q(model, state, a) for a in [0, 1, 2]]
            print(_output)
            self.action_taken = self._get_best_action(_output, model)

    def _reshape_for_model_input(self, data, shape=1):
        """
            Reshapes the input for model predictions.
        """
        _data = np.array(data)
        data_reshape = _data.reshape(shape, len(_data))

        return data_reshape

    def _get_best_action(self, q_values, model):
        """
            :param qValues: a List of Q values corresponding to different actions
            :return: the index of Q values with the highest Q value a.k.a best action
        """
        if model == self.dqn_model:
            max_index = np.argmax(q_values)
            best_action = self._translate_q(max_index)

            return best_action
        if model == self.qlearn_qtable:
            max_q = max(q_values)
            count = model.count(max_q)
            if count > 1:
                best = [i for i in range(len([0, 1, 2])) if model[i] == max_q]
                i = random.choice(best)
                best_action = self._translate_q(i)

                return best_action
            else:
                i = self.qlearn_qtable.index(max_q)
                best_action = self._translate_q(i)

                return best_action
        if model == self.sarsa_qtable:
            max_q = max(q_values)
            count = model.count(max_q)
            if count > 1:
                best = [i for i in range(len([0, 1, 2])) if model[i] == max_q]
                i = random.choice(best)
                best_action = self._translate_q(i)

                return best_action
            else:
                i = self.qlearn_qtable.index(max_q)
                best_action = self._translate_q(i)

                return best_action

    def _translate_q(self, action_taken):
        """
            Translates the model prediction outputs to readable strings which are then used
            by the base controller node to move the robot.
        """
        if action_taken == 0:
            return "MOVE_FORWARD"
        if action_taken == 1:
            return "TURN_LEFT"
        if action_taken == 2:
            return "TURN_RIGHT"
        else:
            raise ValueError("No action available.")

    def check_if_subscribed_topics_ready(self):
        """
            Check if all subscribed topics are ready to receive/send messages
        """
        self._check_laserscan_ready()
        self._check_distance_to_goal_ready()
        self._check_heading_to_goal_ready()

    def _check_laserscan_ready(self):
        """
            Check if all subscribed topics are ready to receive/send messages
        """
        self.laserscan = None
        while self.laserscan is None and not rospy.is_shutdown():
            try:
                self.laserscan = rospy.wait_for_message(DESCRITIZED_LASERSCAN_TOPIC, LaserScan, timeout=5.0)
                rospy.loginfo("%s: Current /descritized_scan is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /descritized_scan not ready yet, retrying...", self.node_name)

        return self.laserscan

    def _check_distance_to_goal_ready(self):
        """
            Check if all subscribed topics are ready to receive/send messages
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
            Check if all subscribed topics are ready to receive/send messages
        """
        self.heading_to_goal = None
        while self.heading_to_goal is None and not rospy.is_shutdown():
            try:
                self.heading_to_goal = rospy.wait_for_message(HEADING_TO_GOAL_TOPIC, Float32, timeout=5.0)
                rospy.loginfo("%s: Current /heading_to_goal is READY", self.node_name)
            except:
                rospy.loginfo("%s: Current /heading_to_goal not ready yet, retrying...", self.node_name)

        return self.heading_to_goal


if __name__ == '__main__':
    try:
        dm = DeployModel()
        dm.check_if_subscribed_topics_ready()
        dm.spin()

    except rospy.ROSInterruptException:
        pass
