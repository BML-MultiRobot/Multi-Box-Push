#! /usr/bin/env python

from task import Task, unitVector
from task import ori as orientation
from task import distance as dist
import numpy as np
import rospy
import vrep
import time
from std_msgs.msg import String, Int8, Int16
from hierarchical_controller import HierarchicalController
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt  
import pickle
import sys  

class Hierarchy_MBRL_Task(Task):
    def __init__(self):
        super(Hierarchy_MBRL_Task, self).__init__()
        self.prev = {"S": None, "A": None}
        self.fail = rospy.Publisher("/restart", Int8, queue_size = 1)
        rospy.Subscriber("/simulation", String, self.receive_simulation_description, queue_size=1)
        rospy.Subscriber("/starting", Int16, self.receive_starting_cue, queue_size=1)
        # ordered such that non-box policy simply selects actions 0 to 3
        self.action_map = {0: 'ANGLE_TOWARDS_GOAL', 1: 'PUSH_IN', 2: 'MOVE_BACK', 3: 'ALIGN_Y',
                           4: 'PUSH_LEFT', 5: 'PUSH_RIGHT', 6: 'APPROACH', 7: 'ANGLE_TOWARDS'}

        self.s_n = 10
        self.currReward = 0
        self.rewards = []
        self.testing_rewards = []

        self.has_box_in_simulation = True

        self.curr_rollout = []
        self.data = []
        self.curr_size = 0
        self.curr_episode = (1, False)
        self.local_curr_episode_synchronous = 0
        self.done = True
        self.tracker_for_testing = 0

        self.counter = 0
        self.period = 50

        self.max_steps = 100 # 50
        self.num_steps = 0

        self.reward_stop = np.inf  # once average test return of past 40 tests reaches this value, we stop training

        self.mode = 'GET_METRIC' #'GET_STATE_DATA' # ''
        self.num_gather_data = 10000  # how many data points gather for classification
        self.controller = HierarchicalController()
        self.simulation_name = None
        self.prev_action_was_valid = True
   
    def extractInfo(self):
        self.vTrain = self.agent.vTrain
        self.pubs = self.agent.pubs
        self.trainMode = self.agent.trainMode
        self.name = self.agent.name
        rospy.Subscriber(self.agents[self.name]['sub'], String, self.receiveState, queue_size = 1) 
 
    def sendAction(self, s, changeAction=True, mock_s=None):
        msg = Vector3()
        self.counter -= 1
        if changeAction:
            ret = self.agent.get_action(s.reshape(1,-1))
            print(self.action_map[ret])
        else:
            ret = self.prev['A'][0]
        action = self.action_map[ret]
        action = self.controller.getPrimitive(self.controller.feature_2_task_state(s.ravel()), action)
        msg.x, msg.y = (action[0], action[1])
        self.pubs[self.name].publish(msg)

        adjusted_state_for_controls = self.controller.feature_2_task_state(s.ravel())
        self.prev_action_was_valid = True if self.isValidAction(adjusted_state_for_controls, self.action_map[ret]) else False 
        return ret, action # NOTE: we changed this so we coud get the raw differential drive output
    
    def stop_moving(self):
        msg = Vector3()
        msg.x, msg.y = (0, 0)
        self.pubs[self.name].publish(msg)
        #print(' attemppting stop momving')
    
    def changeAction(self, s, a, complete):
        self.controller.counter = 1
        return self.counter == 0 #or # self.controller.checkConditions(s, a, complete) 
        
    def isValidAction(self, s, a):
        self.controller.counter = 1
        return self.controller.isValidAction(s, a)

    def append_states(self):
        self.curr_size += len(self.curr_rollout)
        self.data.append(self.curr_rollout)

    def reward_function(self, s):
        s = s.ravel()
        succeeded = self.succeeded(s)
        _, done = self.decide_to_restart(s)

        if succeeded:
            if self.simulation_name == 'elevated_scene':
                return 5 - dist(s[:2], s[5:7])
            if self.simulation_name == 'flat_scene':
                return 5 - abs(self.box_ori_global)
            if self.simulation_name == 'slope_scene':
                return 5 - abs(self.box_ori_global)
        if done and not succeeded:
            if self.simulation_name == 'elevated_scene' and (self.box_z_global < .2 and self.bot_z_global > .2):
                return 0
            else:
                return -5
        else:
            if type(self.prev["S"]) != np.ndarray:
                return 0
            previous_local_state = self.prev['S'].ravel()

            dist_state = 2 if self.simulation_name == 'elevated_scene' else 3
            min_dist = .5 if self.simulation_name == 'elevated_scene' else 1
            previous_distance = dist(previous_local_state[0: dist_state], previous_local_state[5: 5 + dist_state])
            curr_distance = dist(s[:dist_state], s[5: 5 + dist_state])
            d_reward = previous_distance - curr_distance

            prev_ori = self.get_goal_angle(previous_local_state)
            curr_ori = self.get_goal_angle(s, display=True)
            ori_reward = prev_ori - curr_ori if abs(s[3]) < .01 and curr_distance > min_dist else 0  # this is to keep certain calculations correct

            """prev_box_from_hole = previous_local_state[:2] - previous_local_state[5:7]
            hole = s[5:7]
            aligned = hole - dot(hole, unitVector(prev_box_from_hole)) * unitVector(prev_box_from_hole)
            prev_align = dist(np.zeros(2), aligned)
            box_from_hole = s[:2] - s[5:7]
            hole = s[5:7]
            aligned = hole - dot(hole, unitVector(box_from_hole)) * unitVector(box_from_hole)
            curr_align = dist(np.zeros(2), aligned)
            align_reward = prev_align - curr_align"""

            """prev_distance_to_box = dist(np.zeros(3), previous_local_state[:3])
            distance_to_box = dist(np.zeros(3), s[:3])
            box_reward = prev_distance_to_box - distance_to_box"""

            """if self.prev_action_was_valid:
                return -.05
            else:
                return -.3"""
            if self.prev_action_was_valid:
                return 3 * np.round(.5 * np.round(d_reward, 2) + .5 * np.round(ori_reward, 2), 3) - .1
            else:
                return -.3

    def get_goal_angle(self, state, display=False):
        box_location = state[:2]
        hole_location = state[5:7]
        box_to_hole = unitVector(hole_location - box_location)
        theta = -state[4]
        rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        box_to_hole = np.dot(rotate, box_to_hole.reshape((-1, 1))).flatten()
        ori = orientation(box_to_hole)  # this returns the inverse tangent of y / x. Maps -pi/2 to pi/2
        # Must map to -pi to pi for v-rep comparison
        relative_y = box_to_hole[1]
        relative_x = box_to_hole[0]
        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0  # since we want to map -pi to pi
        ori = ori + buff

        if abs(relative_y) > .3:
            return ori  # if not pointing in the general correct direction
        else:
            return 0  # otherwise you're fine

    def receive_starting_cue(self, msg):
        self.curr_episode = (msg.data, False)
    
    def receive_simulation_description(self, msg):
        self.simulation_name = msg.data

    def receiveState(self, msg): 
        if self.curr_episode[0] >= self.local_curr_episode_synchronous + 1 and self.curr_episode[1] == False:
            print('ENVIRONMENT: ', self.simulation_name)
            self.done = False
            self.start_time = time.time()
            # Train for 20 episode, Test for 20 episode
            if self.agent.testing_to_record_progress and self.tracker_for_testing % 20 == 0:
                self.agent.testing_to_record_progress = False
                self.tracker_for_testing = 0
            elif not self.agent.testing_to_record_progress and self.tracker_for_testing % 20 == 0 and len(self.agent.policy.exp) >= self.agent.initial_explore:
                self.agent.testing_to_record_progress = True
                self.tracker_for_testing = 0

            self.local_curr_episode_synchronous += 1
            self.tracker_for_testing += 1

            if self.agent.testing_to_record_progress:
                print('  ##### TESTING ##### ')
            elif len(self.agent.policy.exp) < self.agent.initial_explore:
                print('  ##### EXPLORATION ##### ')
            else:
                print('  ##### TRAINING ##### ')

        floats = vrep.simxUnpackFloats(msg.data)
        self.bot_z_global = floats[self.s_n + 1]
        self.box_z_global = floats[self.s_n]
        self.box_y_global = floats[-2]
        self.box_ori_global = floats[-1]
        local_state = np.array(floats[:self.s_n]).ravel()
        
        adjusted_state_for_controls = self.controller.feature_2_task_state(local_state)
        changeAction = self.changeAction(adjusted_state_for_controls, self.action_map[self.prev['A'][0]], complete=False) if type(self.prev['A']) == tuple else True
        s = (np.array(local_state)).reshape(1,-1)

        succeeded = self.succeeded(s.ravel())
        restart, done = self.decide_to_restart(s.ravel())

        self.done = restart or succeeded
        reward = self.reward_function(s)
        if not self.curr_episode[1]:  # Not finished yet with the episode for syncing purposes
            if not self.done:  # If we hasn't been declared done
                if changeAction:
                    self.counter = self.period
                    action_index, action_control = (self.sendAction(s, changeAction))
                    self.num_steps += 1
                    if self.isValidAction(adjusted_state_for_controls, self.action_map[action_index]):
                        if len(self.curr_rollout) > 0:
                            if all([dist(r, s.ravel()) > .3 for r in self.curr_rollout[-5:]]):
                                self.curr_rollout.append(s.ravel())
                        else:
                            self.curr_rollout.append(s.ravel())
                        if self.mode == 'GET_STATE_DATA':
                            print('Length data for collection: ', len(self.curr_rollout) + self.curr_size)
                    if type(self.prev["S"]) == np.ndarray and not self.agent.testing_to_record_progress:
                        print(self.action_map[self.prev['A'][0]], reward)
                        self.agent.store(self.prev['S'], np.array(self.prev["A"][0]), reward, s, 0, done or succeeded, self.prev['A'][0])
                    if self.trainMode:
                        loss = self.agent.train(self.curr_episode[0])

                    self.prev["S"] = s
                    self.prev["A"] = (int(action_index), action_control)
                    if not self.curr_episode[1]:
                        self.currReward += reward
                else:
                    action_index, a = self.sendAction(s, changeAction)
            else:        
                if type(self.prev["S"]) == np.ndarray:
                    prev_s = self.prev['S']
                    if not self.agent.testing_to_record_progress:
                        self.agent.store(prev_s, np.array(self.prev["A"][0]), reward, s, 0, done or succeeded, self.prev['A'][0])
                        print('Last transition recorded with reward: ', reward)

                self.currReward += reward   
                if succeeded:
                    assert reward > 0
                    print(' ##### SUCCESS ')
                    print(' ##### Success reward: ', reward)
        
        self.restartProtocol(self.done , succeeded=succeeded)   
        return 
    
    def succeeded(self, s):
        assert type(self.simulation_name) == str
        if self.has_box_in_simulation:
            if self.simulation_name == 'elevated_scene':
                return dist(s[:2], s[5:7]) < .5 and self.box_z_global < .2 and self.bot_z_global > .3
            if self.simulation_name == 'flat_scene':
                return dist(s[:3], s[5:8]) < .5
            if self.simulation_name == 'slope_scene':
                return dist(s[:3], s[5:8]) < .5
        else:
            return dist(s[5: 8], np.zeros(3)) < .2

    def decide_to_restart(self, s):
        assert type(self.simulation_name) == str
        # if far away from box, far away from goal, box dropped, or bot dropped
        # Returns tuple (restart, done)
        # TODO: Get rid of this
        max_steps = 100 if self.simulation_name == 'slope_scene' else 50
        if self.num_steps > max_steps:
            return True, False
        failed = False
        if self.simulation_name == 'elevated_scene':
            failed = dist(s[:3], np.zeros(3)) > 4 or dist(s[5:8], np.zeros(3)) > 5 or self.box_z_global < .2 or self.bot_z_global < .3
        if self.simulation_name == 'flat_scene':
            failed = dist(s[:3], np.zeros(3)) > 5 or dist(s[5:8], np.zeros(3)) > 5 or abs(self.box_y_global) > 2
        if self.simulation_name == 'slope_scene':
            failed = abs(self.box_ori_global) > 1 or dist(s[:3], np.zeros(3)) > 5 or dist(s[:3], s[5:8]) > 10
        return failed, failed
        
    
    def restartProtocol(self, restart, succeeded = False):
        if restart == 1 and self.curr_episode[0] > (len(self.testing_rewards) + len(self.rewards)): 
            self.curr_episode = (self.curr_episode[0], self.done)
            print(' ######## Episode Reward: ', self.currReward)
            print('')
            if self.trainMode:
                if self.agent.testing_to_record_progress:
                    self.testing_rewards.append(self.currReward)
                else:
                    self.rewards.append(self.currReward)
                    if succeeded:
                        self.agent.successful_append(self.curr_rollout)
            if not self.trainMode:
                self.testing_rewards.append(self.currReward)
            if self.mode == 'GET_STATE_DATA':       
                if len(self.curr_rollout) > 0:
                    time_execute = time.time() - self.start_time 
                    print(' Succeded: ', succeeded, '    Time: ', time_execute)
                    self.curr_rollout.append(int(succeeded))
                    self.curr_rollout.append(time_execute)
                    self.append_states()
                    print(' LENGTH OF DATA: ', self.curr_size)
                    if self.curr_size > self.num_gather_data:
                        self.agent.stop = True
            elif self.mode == 'GET_METRIC':
                self.curr_size += 1
                self.data.append(int(succeeded))
            else:
                avg_test_reward = sum(self.testing_rewards[-40:])/40.0
                print('Avg test reward: ', avg_test_reward)
                if avg_test_reward >= self.reward_stop:  # TEST HERE
                    self.agent.stop = True
            for k in self.prev.keys():
                self.prev[k] = None
            self.currReward = 0
            self.curr_rollout = []
            self.num_steps = 0
            self.agent.reset()
            msg = Int8()
            msg.data = 1
            self.fail.publish(msg)


    def data_to_txt(self, data, path):
        with open(path, "wb") as fp:   #Pickling
            pickle.dump(data, fp)
        return 

    ######### POST TRAINING #########
    def postTraining(self):
        if self.mode == 'GET_STATE_DATA':
            self.data_to_txt(data=self.data, path =  '/home/jimmy/Documents/Research/AN_Bridging/results/policy_training_data/' + self.agent.method + '_state_data.txt')
            self.data_to_txt(data=self.testing_rewards, path = '/home/jimmy/Documents/Research/AN_Bridging/results/policy_training_data/' + self.agent.method + '_post_training_testing_rewards.txt')
            sys.exit(0)
        elif self.mode == 'GET_METRIC':
            self.data_to_txt(data=self.data, path='/home/jimmy/Documents/Research/AN_Bridging/results/policy_training_data/success.txt')
            self.data_to_txt(data=self.testing_rewards, path='/home/jimmy/Documents/Research/AN_Bridging/results/policy_training_data/post_training_testing_rewards.txt')
            sys.exit(0)
        else:
            self.agent.saveModel()
            self.data_to_txt(data = self.rewards, path = '/home/jimmy/Documents/Research/AN_Bridging/results/policy_training_data/rewards.txt')
            self.data_to_txt(data = self.testing_rewards, path = '/home/jimmy/Documents/Research/AN_Bridging/results/policy_training_data/testing_rewards.txt')
    
    def plotLoss(self):
        loss = self.agent.loss
        plt.plot(range(len(loss)), loss, label='Training')
        if len(self.agent.validation_loss) > 0:
            validation_loss = self.agent.validation_loss
            plt.plot(range(len(validation_loss)), validation_loss, 'r', label='Validation')
        plt.legend()
        plt.show() 