#! /usr/bin/env python

import numpy as np 
import torch
from agent import Agent
from Tasks.hierarchical_controller import HierarchicalController
from doubleQ import DoubleQ
import pickle



class Hierarchical_MBRL(Agent):
    def __init__(self, params, name, task, load_path=None):
        super(Hierarchical_MBRL, self).__init__(params, name, task)
        if self.trainMode:
            self.policy = DoubleQ(params['doubleQ_params'], name, task, load_path=1) # IMPORTNAT NOTE: make sure to initialized DoubleQ first. 
            # Otherwise, the task will reference DoubleQ as the agent...not this class (Hierarchical_MBRL)
        else:
            self.policy = DoubleQ(params['doubleQ_params'], name, task, load_path='/home/jimmy/Documents/Research/AN_Bridging/results/policy_comparison_results/all_final/hierarchical_q_policy2.txt')

        self.u_n = self.vPars['u_n']
        self.explore = self.vTrain['explore']
        self.explore_decay = self.vTrain['explore_decay']
        self.min_explore = self.vTrain['min_explore']

        self.num_sequences = 50
        self.length_sequences = 10
        self.controller = HierarchicalController()

        self.policy_action_selection = 'PROB' # 'DETERM' # PROB will trigger probabilistic training but deterministic testing

        self.has_box_in_simulation = True

        self.testing_to_record_progress = True  # Don't change this. Setting this to True makes it START with training

        self.counter = 0
        self.train_every_steps = params['doubleQ_params']['valTrain']['train_every_steps']
        self.initial_explore = params['doubleQ_params']['valTrain']['explore_steps']

        self.success_states = np.zeros((1,10))

        torch.set_num_threads(2)

        task.initAgent(self)
    
        if not load_path:
            while(not self.stop):
                x = 1+1
            task.postTraining()
    
    def successful_append(self, rollout_list):
        end_state = rollout_list[-1]
        self.success_states = np.vstack((self.success_states, end_state))

    def saveModel(self):
        self.policy.saveModel()
        pass
    
    def save_data(self, data):
        path = '/home/jimmy/Documents/Research/AN_Bridging/results/policy_training_data/model_data_elevated_push_hole_differential_2.txt'
        with open(path, "wb") as fp:  
            pickle.dump(data, fp)
        return
    
    def store(self, s, a, r, sprime, aprime, done, action_index):
        if not self.testing_to_record_progress:
            self.counter += 1
            s = self.concatenate_identifier(s)
            a = np.array([a]) if type(a) == int else a
            sprime = self.concatenate_identifier(sprime)
            self.policy.store(s, action_index, np.array([r]).reshape(1, -1), sprime, aprime, done)
            print(' Experience length: ', len(self.policy.exp))

    def concatenate_identifier(self, s):
        """identifier = 1 if self.has_box_in_simulation else 0
        return np.hstack((s, np.repeat(identifier, s.shape[0]).reshape(-1, 1)))"""
        # we get rid of this because train two different policies is easier
        return s
    
    def get_action(self, s):
        self.task.stop_moving()
        if not self.testing_to_record_progress:
            self.explore = max(self.min_explore, self.explore * self.explore_decay) # NOTE: was .9996 before
        return self.return_q_policy_action_index(s, testing_time=self.testing_to_record_progress)

    def return_q_policy_action_index(self, s, testing_time):
        i = np.random.random() 
        print('')
        if i < self.explore and not testing_time and self.trainMode:
            return np.random.randint(self.u_n)
        if self.policy_action_selection == 'PROB':
            return self.policy.get_action(self.concatenate_identifier(s), testing_time, probabilistic=True)
        else:
            return self.policy.get_action(self.concatenate_identifier(s), testing_time, probabilistic=False)

    def train(self, episode=0):
        if len(self.policy.exp) >= self.initial_explore and not self.testing_to_record_progress and self.trainMode:
            if self.counter % self.train_every_steps == 0:
                for i in range(self.train_every_steps):
                    loss = self.policy.train(override=True)
