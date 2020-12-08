#! /usr/bin/env python

import numpy as np 
import torch
import torch.nn as nn
import math 
import rospy
from std_msgs.msg import String, Int8
from geometry_msgs.msg import Vector3
import vrep
import matplotlib.pyplot as plt
import torch.optim as optim
from Buffers.CounterFactualBuffer import Memory

from Networks.network import Network
from Networks.softNetwork import SoftNetwork
from agent import Agent
from Buffers.CounterFactualBuffer import Memory 

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


""" Soft actor critic implementation"""

class SAC(Agent):
    def __init__(self, params, name, task):
        super(SAC, self).__init__(params, name, task)
        self.aPars      = params['actPars']
        self.aTrain     = params['actTrain']
        self.qPars      = params['qPars']
        self.qTrain     = params['qTrain']
        if self.trainMode:
            self.QNets = [Network(self.qPars, self.qTrain).to(device) for i in range(2)]
            self.QTargets = [Network(self.qPars, self.qTrain).to(device) for i in range(2)]
            self.policyNet = SoftNetwork(self.aPars, self.aTrain).to(device)
        else:
            print('Not implemented')

        for target_param, param in zip(self.VTar.parameters(), self.VNet.parameters()):
            target_param.data.copy_(param)

        self.expSize    = self.vTrain['buffer']
        self.actions    = self.aPars['neurons'][-1]
        self.state      = self.aPars['neurons'][0]
        self.exp        = Memory(size = self.expSize)

        task.initAgent(self)
    
        while(not self.stop):
            x = 1+1
        task.postTraining()

    def store(self, s, a, r, sprime, aprime, done):
        self.exp.push(s, a, r, 1-done, aprime, sprime)

    def load_nets(self):
        pass

    def saveModel(self):
        torch.save(self.policyNet.state_dict(),
                   '/home/jimmy/Documents/Research/AN_Bridging/model_training_data/hierarchical_sac_policy.txt')
        pass
    
    def get_action(self, s):
        distribution = self.policyNet(torch.FloatTensor(s))
        return distribution.sample().detach().numpy()

    def send_to_device(self, s, a, r, next_s, d):
        s = torch.FloatTensor(s).to(device)
        a = torch.FloatTensor(a).to(device)
        r = torch.FloatTensor(r).unsqueeze(1).to(device)
        next_s = torch.FloatTensor(next_s).to(device)
        d = torch.FloatTensor(np.float32(d)).unsqueeze(1).to(device)
        return s, a, r, next_s, d
        
    def train(self):
        if len(self.exp) > 1000:
            # NOTE: Tailored to discrete action settings
            s, a, r, masks, _, next_s, _, _, _ = self.exp.sample(batch=self.batch_size)
            d = 1 - masks
            s, a, r, next_s, d = self.send_to_device(s, a, r, next_s, d)

            q_over_actions = [net(s) for net in self.QNets]
            next_action_distribution = self.policyNet(next_s)
            next_actions = self.get_action(next_s)
            q_targets = [torch.gather(net(next_s), 1, torch.LongTensor(next_actions).unsqueeze(1)) for net in self.QTargets]
            q_targets = torch.min(q_targets[0], q_targets[1])

            next_distribution = self.policyNet(next_s)
            next_v = next_distribution @ (q_targets - self.alpha * torch.log(next_action_distribution))

            next_q = r + (1 - d) * self.discount * next_v
            q_loss = [net.get_loss(torch.gather(q_over_actions[i], 1, torch.LongTensor(a).unsqueeze(1)), next_q.detach()) for i, net in enumerate(self.QNets)]
            q_loss = sum(q_loss)

            action_distribution = self.policyNet(s)
            new_a = action_distribution.sample()
            log_prob = action_distribution.log_prob(new_a)

            minimized_q = torch.min(q_over_actions[0], q_over_actions[1])
            target = minimized_q - (action_distribution @ minimized_q)
            actor_loss = log_prob * self.alpha * torch.log(action_distribution) - (action_distribution @ target)

            self.QNet.optimizer.zero_grad()
            q_loss.backward()
            self.QNet.optimizer.step()

            self.policyNet.optimizer.zero_grad()
            actor_loss.backward()
            self.policyNet.optimizer.step()

            for target_param, param in zip(self.VTar.parameters(), self.VNet.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - 5*1e-3) + param.data * 5*1e-3)

            self.totalSteps += 1
