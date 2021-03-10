#! /usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import Int8, String
from geometry_msgs.msg import Vector3
from collections import namedtuple
from vrep_util.vrep import *
import matplotlib.pyplot as plt
import torch

from buffers.policy_buffer import PolicyMemory
from buffers.reward_buffer import RewardMemory
from cla import CLA
from reward_network import RewardNetwork
from env_util import to_state_id, save_data


Agent = namedtuple('Agent', ['x', 'y', 'ori', 'id'])
Target = namedtuple('Target', ['x', 'y'])
Sample = namedtuple('Sample', ('state', 'action'))


class MultiAgent:
    def __init__(self, num_agents, params):
        """ Misc """
        self.num_agents = num_agents
        self.locations = [(0, 0)] * num_agents  # maps robot id to current location tuple
        self.orientations = [0] * num_agents
        self.ball_location, self.prev_ball, self.ball_start = (0, 0), None, (0, 0)
        self.action_map = {0, 1}

        """ ROS Subscriptions and Publishers """
        self.target_publishers = {i: rospy.Publisher("/goalPosition" + str(i), Vector3, queue_size=1) for i in range(num_agents)}
        for i in range(num_agents):
            rospy.Subscriber("/state" + str(i), String, self.receive_vrep, queue_size=1)
        self.pub_restart = rospy.Publisher("/restart", Int8, queue_size=1)
        self.pub_start = rospy.Publisher("/start", Int8, queue_size=1)
        rospy.Subscriber("/finished", Int8, self.finished, queue_size=1)

        """ Tracking Steps """
        self.period = 50
        self.episode_length = params['max_ep_len']
        self.total_steps = 0
        self.steps = np.array([0] * num_agents)  # maps robot id to number steps received

        """ Previous state trackers """
        self.prev_local_sample = {}
        self.prev_reordering = []
        self.prev_global_sample = None

        """ Network Architecture """
        self.u_n = len(self.action_map)
        neurons = [3*self.num_agents + 1] + [params['reward_width']] * params['reward_depth'] + [1]

        """ Buffer Replays """
        self.reward_buffer = RewardMemory(size=1e6)
        self.policy_buffer = PolicyMemory()

        """ Policy CLA Parameters"""
        # state_indicators = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1, 2, 3]]
        state_indicators = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1], [0, 1, 2]]
        # state_indicators = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2]]
        self.policy = CLA(state_indicators, self.u_n, params['a'], params['b'], params['q_learn'], params['gamma'], params['td_lambda'],
                          params['explore'], params['explore_decay'], params['min_explore'])
        self.policy_epochs = params['policy_epochs']
        self.rim_size = params['rim_size']
        self.near_ball = .225
        self.near_agent = .1

        """ Reward Network Parameters """
        self.reward_network = RewardNetwork(neurons, params['reward_lr'], self.u_n, self.num_agents, params['boltzmann'], params['noise_std'])
        self.rotation_invariant = params['rotation_invariance']
        self.train_every = params['train_every']
        self.reward_batch = params['reward_batch']
        self.train_now = params['explore_steps']
        self.epochs = params['epochs']

        """ Trackers for data """
        self.reward_network_loss, self.entropy, self.episode_rewards, self.x_travel, self.y_travel = [], [], [], [], []
        self.entropy.append(self.policy.average_entropy())
        self.curr_reward = 0

        rospy.sleep(3)
        self.pub_start.publish(Int8(1))
        self.test = {0: 0, 1: 0}
        rospy.spin()
        return

    def publish_targets(self, targets):
        """ targets: dictionary mapping robot id to global, goal location """
        for id, target in targets.items():
            pub = self.target_publishers[id]
            msg = Vector3(target.x, target.y * self.rim_size, 0)
            pub.publish(msg)
        return

    def receive_vrep(self, msg):
        """ Message from V-REP. Enforces only send new action to all agents if
            all agents have gone through self.period steps """
        agent = self.unpack_msg(msg)
        self.locations[agent.id] = (agent.x, agent.y)
        self.orientations[agent.id] = agent.ori
        self.steps[agent.id] += 1
        self.ball_start = self.ball_location if self.total_steps == 0 else self.ball_start
        if not any(self.steps <= self.period):
            sys.exit(0) # make sure each time step is accounted for
        if all(self.steps >= self.period):
            states, targets, actions = self.action_step()
            failed = self.failed()
            if self.total_steps > self.episode_length or failed:
                self.record(states, actions, done=int(failed), episode_end=int(True))
                self.restart_protocol()
            else:
                self.steps = np.array([0] * num_agents)
                self.total_steps += 1
                self.record(states, actions)
        return

    def failed(self):
        return False

    def restart_protocol(self):
        """ Restart the episode protocol """
        self.pub_restart.publish(Int8(1))
        self.total_steps = 0
        self.steps = np.array([0] * num_agents)
        self.episode_rewards.append(self.curr_reward)

        delta_x = self.ball_location[0] - self.ball_start[0]
        dev_y = abs(self.ball_location[1] - self.ball_start[1])
        self.x_travel.append(delta_x)
        self.y_travel.append(dev_y)

        print('Reward Buffer: ', len(self.reward_buffer), 'Policy Buffer: ', len(self.policy_buffer))
        print('Rewards: ', self.curr_reward, ' X Travel: ', delta_x, '  Y Abs Travel: ', dev_y)
        print('')

        self.train()

        self.prev_local_sample = {}
        self.prev_reordering = []
        self.prev_global_sample = None
        self.prev_ball = None

        self.curr_reward = 0
        self.pub_start.publish(Int8(1))

    def record(self, states, actions, done=0, episode_end=0):
        """ Handle buffer updates and storage of replay """
        curr_global_state, reordering = self.generate_global_state(actions)

        if self.prev_local_sample != {}:
            # Global Information
            prev_global_state, prev_global_action = self.prev_global_sample.state, self.prev_global_sample.action
            id_to_state_place = self.prev_reordering

            # Reward buffer
            r = self.reward_function(prev_global_state, prev_global_action, curr_global_state.state)
            self.reward_buffer.push(prev_global_state, prev_global_action, r, self.num_agents)
            self.curr_reward += r

            # Policy Buffer
            for i in self.prev_local_sample.keys():
                prev_s, prev_a = self.prev_local_sample[i]
                state_index = id_to_state_place[i]
                if prev_a == prev_global_action[state_index]:
                    self.policy_buffer.push(prev_s, prev_a, states[i], actions[i], state_index, done, prev_global_state, prev_global_action, episode_end, i)
                else:
                    print('### DANGER. Possible data recording bug in function "record" under multiagent_coordinator? ')

        # Reset prev
        self.prev_local_sample = {i: Sample(states[i], actions[i])for i in range(self.num_agents)}
        self.prev_global_sample = curr_global_state
        self.prev_reordering = reordering
        self.prev_ball = self.ball_location
        return

    def train(self, override=False):
        """ Handles training the CLA agent """
        # Update the reward network
        if override or (len(self.reward_buffer) >= self.train_now):
            # Train Reward
            print(' ######### TRAINING #########')
            batches = self.reward_buffer.batch_all_memory(self.reward_batch)
            for i in range(self.epochs):
                sys.stdout.write("\r Reward Training Progress: %d%%" % (int(100*(i+1)/self.epochs)))
                sys.stdout.flush()
                epoch_loss = self.reward_network.train_multiple_transitions(batches)
                self.reward_network_loss.append(epoch_loss)
            print('Reward Average Losses: ', self.reward_network_loss[-1])
            print('')

            # Train Policy
            rollouts = self.policy_buffer.batch_all_memory(shuffle=(not self.policy.q_learn))
            for i in range(self.policy_epochs):
                sys.stdout.write("\r Policy Training Progress: %d%%" % (int(100 * (i + 1) / self.policy_epochs)))
                sys.stdout.flush()
                for r in rollouts:
                    self.update_policy(r)
            self.policy.update_explore()
            print('Entropy: ', self.entropy[-1])
            print('')

            self.policy_buffer.clear()
            if not override:
                self.train_now = len(self.reward_buffer) + self.train_every

        return

    def update_reward(self, r_trans):
        reward_network_loss = self.reward_network.train_single_transition(r_trans)
        self.reward_network_loss.append(reward_network_loss)

    def update_policy(self, p_trans):
        average_entropy = self.policy.update_policy(p_trans.local_state, p_trans.local_action,
                                                    p_trans.next_state, p_trans.next_action, p_trans.global_state,
                                                    p_trans.global_action, p_trans.robot_id, p_trans.done, self.reward_network)
        self.entropy.append(average_entropy)

    def reward_function(self, s, a, s_prime):
        """ Reward Function using global states and actions """
        s, s_prime = s.flatten(), s_prime.flatten()

        locations = s[:-1]
        squared = np.square(locations)

        squared_distances = squared[::2] + squared[1::2]
        prev_distance = np.sum(np.sqrt(squared_distances))

        locations = s_prime[:-1]
        squared = np.square(locations)
        squared_distances = squared[::2] + squared[1::2]
        new_distance = np.sum(np.sqrt(squared_distances))

        r_agent_distance = prev_distance - new_distance
        # TODO: This is a test
        r_ball_forward = self.ball_location[0] - self.prev_ball[0]
        r_ball_side = abs(self.prev_ball[1]) - abs(self.ball_location[1])
        return r_ball_forward * 5 + r_ball_side * 5  # + r_agent_distance

    def action_step(self):
        """ Creates new targets for all agents and then sends to all agents """
        targets, local_states, actions = {}, {}, {}
        for i in range(self.num_agents):
            local_states[i] = to_state_id(i, self.locations, self.orientations, self.ball_location, self.near_ball)
            actions[i] = self.policy.get_action(local_states[i])
            targets[i] = Target(local_states[i] % 10, actions[i])
        self.publish_targets(targets)
        return local_states, targets, actions

    def unpack_msg(self, msg):
        """ Handles unpacking message from v-rep"""
        state = simxUnpackFloats(msg.data)
        state[3] = int(state[3])
        self.ball_location = (state[4], state[5])
        agent = Agent(*state[:4])
        return agent

    def finished(self, msg):
        """ Handle finishing all training """
        save_data('/home/jimmy/Documents/Research/CLA/results/policy.pickle', self.policy)
        save_data('/home/jimmy/Documents/Research/CLA/results/reward_net_loss.pickle', self.reward_network_loss)
        save_data('/home/jimmy/Documents/Research/CLA/results/entropy.pickle', self.entropy)
        save_data('/home/jimmy/Documents/Research/CLA/results/episode_rewards.pickle', self.episode_rewards)
        save_data('/home/jimmy/Documents/Research/CLA/results/x_travel.pickle', self.x_travel)
        save_data('/home/jimmy/Documents/Research/CLA/results/y_travel.pickle', self.y_travel)

        torch.save(self.reward_network.state_dict(), '/home/jimmy/Documents/Research/CLA/results/reward_network.txt')

        plt.plot(range(len(self.reward_network_loss)), self.reward_network_loss)
        plt.title('Reward Network Loss over Training Steps')
        plt.show()

        plt.plot(range(len(self.entropy)), self.entropy)
        plt.title('Average Policy Entropy over Training Steps')
        plt.show()

        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.title('Episode Accumulated Rewards')
        plt.show()

        sys.exit(0)

    def generate_global_state(self, actions_dict):
        """ Generates global state to pass into reward network.
            Transforms all locations to relative to BALL """
        locations_relative_to_ball, all_actions = [], []
        for i in range(self.num_agents):
            locations_relative_to_ball.append(np.subtract(self.locations[i], self.ball_location))
            all_actions.append(actions_dict[i])

        if self.rotation_invariant:
            angles = [np.arctan2(tup[1], tup[0]) for tup in locations_relative_to_ball]
            angles = [a + 2*np.pi if a < 0 else a for a in angles]
            state_place_to_id = np.argsort(angles)
            locations_relative_to_ball = np.take(locations_relative_to_ball, state_place_to_id, axis=0).flatten()
            all_actions = np.take(all_actions, state_place_to_id).flatten()
            id_to_state_place = {robot_id: state_place for state_place, robot_id in enumerate(state_place_to_id)}
        else:
            id_to_state_place = {i: i for i in range(self.num_agents)}
            all_actions = np.array(all_actions).flatten()

        state = np.append(locations_relative_to_ball, self.ball_location[1])
        return Sample(state, all_actions), id_to_state_place


if __name__ == '__main__':
    rospy.init_node('Dum', anonymous=True)
    num_agents = rospy.get_param('~num_bots')
    params = {
              # general parameters
              'train_every': 1000, 'max_ep_len': 100, 'explore_steps': 2000,

              # reward network
              'reward_width': 300, 'reward_depth': 3, 'reward_lr': 3e-4,
              'reward_batch': 250, 'rotation_invariance': False, 'epochs': 75,
              'noise_std': 0,

              # General Policy parameters
              'policy_epochs': 1, 'a': .1,   # a = lr for q learn

              # cla-specific parameters
              'b': 0, 'boltzmann': 50,

              # diff-q policy gradient parameters
              'q_learn': False, 'gamma': .8, 'td_lambda': .9,

              # control parameters
              # 'rim_size': .02,
              'rim_size': .05,

              # exploration
              'explore': 1, 'explore_decay': .5, 'min_explore': .05,
              }
    agent = MultiAgent(num_agents, params)
