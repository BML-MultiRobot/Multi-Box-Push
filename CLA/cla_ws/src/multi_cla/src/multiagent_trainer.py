#! /usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import Int8, String
from geometry_msgs.msg import Vector3
from collections import namedtuple
from vrep_util.vrep import *
import matplotlib.pyplot as plt
import torch

from buffers.off_policy_buffer import PolicyMemory
from buffers.on_policy_buffer import OnPolicyMemory
from buffers.reward_buffer import RewardMemory
from cla import CLA
from reward_network import RewardNetwork
from env_util import to_state_id, save_data, load_data, get_rotation_map, convert_to_rotation_invariant, agent_moved


Agent = namedtuple('Agent', ['x', 'y', 'ori', 'id'])
Target = namedtuple('Target', ['radians', 'dist'])
Sample = namedtuple('Sample', ('state', 'action'))


class MultiAgent:
    def __init__(self, num_agents, params):
        """ Misc """
        self.num_agents = num_agents
        self.locations = [(0, 0)] * num_agents  # maps robot id to current location tuple
        self.orientations = [0] * num_agents
        self.ball_location, self.prev_ball, self.ball_start = (0, 0), None, (0, 0)
        self.action_map = {0, 1, 2, 3}  # {0, 1}
        self.movement_constraint = params['movement_constraint']

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
        self.policy_buffer = PolicyMemory(size=params['policy_buffer_size'])
        self.on_policy_buffer = OnPolicyMemory(size=params['on_policy_buffer_size'])

        """ Policy CLA Parameters"""
        state_indicators = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [0, 1], [0, 1, 2], [0, 1], [0, 1]]

        if params['load_policy']:
            self.policy = load_data(params['load_policy'])
        else:
            self.policy = CLA(state_indicators, self.u_n, params)
        self.rim_size = params['rim_size']
        self.near_ball = params['near_ball']
        self.near_agent = .1
        self.policy_batch = params['policy_batch']
        self.policy_steps = params['policy_steps']
        self.distribution_hold = params['distribution_hold']
        self.distribution_explore = params['distribution_explore']

        """ Reward Network Parameters """
        self.reward_network = RewardNetwork(neurons, self.u_n, self.num_agents, params)
        self.rotation_invariant = params['rotation_invariance']
        self.reward_batch = params['reward_batch']
        self.explore_steps = params['explore_steps']
        self.reward_epochs = params['reward_epochs']
        if params['reward_path']:
            self.reward_network.load_state_dict(torch.load(params['reward_path']))

        """ Trackers for data """
        self.reward_network_loss, self.entropy, self.episode_rewards, self.x_travel, self.y_travel, self.q_loss, self.coma_loss = [], [], [], [], [], [], []
        self.entropy.append(self.policy.average_entropy(False))
        self.curr_reward = 0

        """ Testing """
        self.test_mode = params['test_mode']

        rospy.sleep(3)
        self.pub_start.publish(Int8(1))
        self.time = rospy.get_time()
        rospy.spin()
        return

    def publish_targets(self, targets):
        """ targets: dictionary mapping robot id to global, goal location """
        for id, target in targets.items():
            pub = self.target_publishers[id]
            msg = Vector3(target.radians, target.dist, 0)
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
            sys.exit(0)  # make sure each time step is accounted for
        if all(self.steps >= self.period):
            states, targets, actions = self.action_step()
            failed = self.failed()

            if self.total_steps >= self.episode_length or failed:
                print('Time per iteration: ', rospy.get_time() - self.time)
                self.record(states, actions, done=int(failed), episode_end=int(True))
                self.restart_protocol()
            else:
                self.steps = np.array([0] * num_agents)
                self.total_steps += 1
                self.record(states, actions)
            self.time = rospy.get_time()
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

        self.train()

        print('Reward Buffer: ', len(self.reward_buffer), 'Policy Buffer: ', len(self.policy_buffer))
        print('Rewards: ', self.curr_reward, ' X Travel: ', delta_x, '  Y Abs Travel: ', dev_y)
        if len(self.entropy) > 0 and len(self.reward_network_loss) > 0 and len(self.q_loss) > 0 and len(self.coma_loss) > 0:
            print('Entropy: ', self.entropy[-1], ' Reward Average Losses: ', self.reward_network_loss[-1],
                  ' Q Loss: ', self.q_loss[-1], ' Coma LOSS', self.coma_loss[-1])
        print('')

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

            # Joint Buffer
            if self.distribution_hold - self.distribution_explore <= 0:
                for i in self.prev_local_sample.keys():
                    prev_s, prev_a = self.prev_local_sample[i]
                    state_index = id_to_state_place[i]
                    if prev_a == prev_global_action[state_index]:
                        self.on_policy_buffer.push(prev_s, prev_a, states[i], actions[i], state_index, done, r,
                                                prev_global_state,
                                                prev_global_action, curr_global_state.state, curr_global_state.action,
                                                episode_end, i)
                    else:
                        print(
                            '### DANGER. Possible data recording bug in function "record" under multiagent_coordinator? ')

            # Policy Buffer
            for i in self.prev_local_sample.keys():
                prev_s, prev_a = self.prev_local_sample[i]
                state_index = id_to_state_place[i]
                if prev_a == prev_global_action[state_index]:

                    self.policy_buffer.push(prev_s, prev_a, states[i], actions[i], state_index, done, r, prev_global_state,
                                            prev_global_action)

                else:
                    print('### DANGER. Possible data recording bug in function "record" under multiagent_coordinator? ')
            # if self.distribution_hold == 0:
            #     self.policy_buffer.clear(leave=self.distribution_explore * (self.num_agents * self.episode_length))
        else:
            print(' Empty. Waiting for next one. ')

        # Reset prev
        self.prev_local_sample = {i: Sample(states[i], actions[i]) for i in range(self.num_agents)}
        self.prev_global_sample = curr_global_state
        self.prev_reordering = reordering
        self.prev_ball = self.ball_location
        return

    def train(self, override=False):
        """ Handles training the CLA agent """
        # Update the reward network
        if (override or (len(self.reward_buffer) >= self.explore_steps)) and not self.test_mode:
            # Train Reward
            for i in range(self.reward_epochs):
                batch = self.reward_buffer.batch_all_memory(self.reward_batch)
                epoch_loss = self.reward_network.train_multiple_transitions(batch)
                self.reward_network_loss.append(epoch_loss)

            if len(self.policy_buffer) >= self.explore_steps * self.num_agents:
                if self.distribution_hold <= 0:
                    rollouts = self.on_policy_buffer.batch_all_memory(shuffle=False)
                    self.policy.update_policy(rollouts, None, self.distribution_hold <= 0)
                    # TODO: Clear on_policy_buffer
                else:
                    for i in range(self.policy_steps):
                        # Train Policy
                        samples = self.policy_buffer.sample(batch=self.policy_batch)
                        average_entropy, q_loss = self.policy.update_policy(samples, self.reward_network, self.distribution_hold <= 0)
                        self.entropy.append(average_entropy)
                        self.q_loss.append(q_loss)

            print('distribution hold: ', self.distribution_hold, '  explore: ', self.policy.explore)
            p = self.policy.policy_final if self.distribution_hold <= 0 else self.policy.policy
            print(' Bottom left: ', p[100020110], '  Center: ', p[20008], '   Top left: ', p[1020108])
            self.policy.update_explore(self.distribution_hold <= 0)
            self.distribution_hold -= 1

        return

    def reward_function(self, s, a, s_prime):
        """ Reward Function using global states and actions """
        r_ball_forward = self.ball_location[0] - self.prev_ball[0]
        r_ball_side = abs(self.prev_ball[1]) - abs(self.ball_location[1])
        return r_ball_forward * 5 + r_ball_side * 5

    def action_step(self):
        """ Creates new targets for all agents and then sends to all agents """
        targets, local_states, actions = {}, {}, {}
        local_states, location_indicators, allowed = to_state_id(self.num_agents, self.locations,
                                                                  self.orientations, self.ball_location, self.near_ball)
        for i in range(self.num_agents):
            actions[i] = self.policy.get_action(local_states[i], self.distribution_hold <= 0, probabilistic=True)
            if actions[i] == 0 or actions[i] == 1 or (self.movement_constraint and (actions[i] == 2 and not allowed[i]['c']) or (actions[i] == 3 and not allowed[i]['cc'])):
                targets[i] = Target(location_indicators[i] * (np.pi/8), min(actions[i], 1) * self.rim_size)
            elif actions[i] == 2:
                targets[i] = Target(((location_indicators[i] - 1) % 16) * (np.pi/8), self.rim_size)
            else:  # actions[i] == 3
                targets[i] = Target(((location_indicators[i] + 1) % 16) * (np.pi / 8), self.rim_size)
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
        save_data('/home/jimmy/Documents/Research/CLA/results/q_loss.pickle', self.q_loss)

        torch.save(self.reward_network.state_dict(), '/home/jimmy/Documents/Research/CLA/results/reward_network.txt')

        plt.plot(range(len(self.reward_network_loss)), self.reward_network_loss)
        plt.title('Reward Network Loss over Training Steps')
        plt.show()

        plt.plot(range(len(self.q_loss)), self.q_loss)
        plt.title('Q Value Average Loss over Training Steps')
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
            state_place_to_id = get_rotation_map(locations_relative_to_ball)
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
              'max_ep_len': 100, 'explore_steps': 1000, 'test_mode': False,

              # reward network
              'reward_width': 300, 'reward_depth': 3, 'reward_lr': 3e-4,
              'reward_batch': 200, 'rotation_invariance': True,
              'noise_std': 0, 'reward_weight': 16, 'reward_epochs': 5,
              'reward_path': None, # '/home/jimmy/Documents/Research/CLA/results/off_policy_cf_ddac_8_weight_2_state/reward_network.txt', # None,

              # General Policy parameters
              'policy_batch': 250, 'a': 5e-2,  # a = lr for q learn policy gradient
              'load_policy': None, #'/home/jimmy/Documents/Research/CLA/results/off_policy_cf_ddac_8_weight_2_state/policy.pickle',  # '/home/jimmy/Documents/Research/CLA/results/off_policy_cf_ddac/policy.pickle',
              'movement_constraint': True,  # Toggles whether we constrain action space with if statements
              'policy_steps': 10, 'policy_buffer_size': 10000,

              # diff-q policy gradient parameters
              'q_lr': .05, 'gamma': .98,
              'counterfactual': True, 'distribution_hold': 0, 'distribution_explore': 0,

              # control parameters
              # 'rim_size': .02,
              'rim_size': .075, 'near_ball': .325,

              # exploration
              'explore': .1, 'final_explore': 0, 'final_explore_decrement': .01,
              }
    agent = MultiAgent(num_agents, params)
