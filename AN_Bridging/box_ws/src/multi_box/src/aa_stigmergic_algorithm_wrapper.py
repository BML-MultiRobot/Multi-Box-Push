#! /usr/bin/env python
"""
Wrapper for entire stigmergic algorithm. Includes:
    - Incorporation of trained policy using hierarchical RL
    - Incorporation of map representation of the environment with stigmergic pheromone communication
    - Interaction and processing through V-Rep simulation
"""
import rospy
import vrep
import numpy as np

from Algs.doubleQ import DoubleQ
from std_msgs.msg import String, Int16, Int8
from geometry_msgs.msg import Vector3
from Tasks.hierarchical_controller import HierarchicalController
from Tasks.task import Task
from Tasks.task import distance as dist
from aa_doubleQ_trained_parameters import doubleQPars

action_map = {0: 'ANGLE_TOWARDS_GOAL', 1: 'PUSH_IN', 2: 'MOVE_BACK', 3: 'ALIGN_Y',
              4: 'PUSH_LEFT', 5: 'PUSH_RIGHT', 6: 'APPROACH', 7: 'ANGLE_TOWARDS'}

class Agent_Manager(object):
    def __init__(self, robot_id):
        self.robot_id = robot_id  # for now...will figure out how to do this later
        self.cmd_vel_topic = "/action#" + str(self.robot_id)
        self.policy = DoubleQ(doubleQPars, name='Dummy', task=Task(),
                              load_path='/home/jimmy/Documents/Research/AN_Bridging/results/policy_comparison_results/all_final/hierarchical_q_policy2.txt')
        self.controller = HierarchicalController()
        rospy.Subscriber("/robotState" + str(self.robot_id), String, self.receive_state_info, queue_size=1)
        rospy.Subscriber("/shutdown" + str(self.robot_id), Int16, self.shutdown, queue_size=1)
        rospy.Subscriber('/finished', Int8, self.receive_restart, queue_size=1)
        self.pub = rospy.Publisher("/robotAction" + str(self.robot_id), Vector3, queue_size=1)
        self.pub_finish = rospy.Publisher("/robot_finished", Int16, queue_size=1)
        self.period = 50
        self.counter = 0
        self.shut_down = False
        self.sleep_duration = 5
        self.box_distance_finished_condition = .5 # These were the parameters used in training
        self.marker_distance_finished_condition = .5
        print(' Robot number:', robot_id, ' is ready!')
        rospy.wait_for_message('/map_is_ready', Int16)  # just wait for the message indicator
        self.publish_finished_to_map()
        rospy.spin()
        return

    def receive_state_info(self, msg):
        if self.counter == 0:
            state = np.array(vrep.simxUnpackFloats(msg.data))
            self.controller.goal = state.ravel()[:2]
            if self.finished(state):
                if not self.shut_down and (not self.is_not_pushing_box(state)) and state[7] < -.25 and dist(state[5:7], np.zeros(2)) < 1: # hole
                    msg = Vector3()
                    msg.x = -2
                    msg.y = -2
                    self.pub.publish(msg)
                else:
                    if not self.shut_down:
                        self.publish_finished_to_map()
                    msg = Vector3()
                    msg.x = 1
                    msg.y = 1
                    self.pub.publish(msg)
            else:
                if self.is_not_pushing_box(state):
                    action_index = 0 if abs(state[6]) > .4 else 1 # align yourself otherwise travel towards node
                else:
                    action_index = self.policy.get_action(state, testing_time=True, probabilistic=True)
                action_name = action_map[action_index]
                adjusted_state_for_controls = self.controller.feature_2_task_state(state)
                left_right_frequencies = self.controller.getPrimitive(adjusted_state_for_controls, action_name)
                msg = Vector3()
                msg.x = left_right_frequencies[0]
                msg.y = left_right_frequencies[1]
                self.pub.publish(msg)
        self.counter = (self.counter + 1) % self.period
        return

    def is_not_pushing_box(self, state):
        # Check V-Rep sim for reference why this is true. For convenience.
        return all([state[i] == -1 for i in range(5)])

    def finished(self, state):
        if self.is_not_pushing_box(state):
            return dist(state[5:8], np.zeros(3)) < self.marker_distance_finished_condition
        else:
            flat = state.flatten()
            if flat[7] < -.25: # hole
                return dist(state[:3], state[5:8]) < self.box_distance_finished_condition and abs(flat[2] - flat[7]) < .2
            return dist(state[:3], state[5:8]) < self.box_distance_finished_condition

    def publish_finished_to_map(self):
        self.pub_finish.publish(Int16(self.robot_id))
        return

    def shutdown(self, msg):
        protocol = msg.data
        if protocol == 1: # shutdown completely
            self.shut_down = True
        else:
            self.shut_down = False

    def receive_restart(self, msg):
        self.shut_down = False
        rospy.wait_for_message('/starting', Int16)
        self.publish_finished_to_map()


if __name__ == '__main__':
    rospy.init_node('Dummy', anonymous=True)
    robot_id = rospy.get_param('~robot_id')
    agent = Agent_Manager(robot_id)


