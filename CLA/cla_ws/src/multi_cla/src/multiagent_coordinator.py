#! /usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import Int8, String
from geometry_msgs.msg import Vector3
from collections import namedtuple
from vrep_util.vrep import *

from buffers.policy_buffer import PolicyMemory
from buffers.reward_buffer import RewardMemory
from cla import CLA
from reward_network import RewardNetwork


Agent = namedtuple('Agent', ['x', 'y', 'ori', 'id'])
Target = namedtuple('Target', ['x', 'y'])
Sample = namedtuple('Sample', ('state', 'action'))

episodes = 100


class MultiAgent:
    def __init__(self, num_agents, params):
        self.num_agents = num_agents
        self.locations = [(0, 0)] * num_agents # maps robot id to current location tuple
        self.orientations = [0] * num_agents
        self.ball_location, self.prev_ball = (0, 0), None
        self.action_map = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

        self.target_publishers = {i: rospy.Publisher("/goalPosition" + str(i), Vector3, queue_size=1) for i in range(num_agents)}
        for i in range(num_agents):
            rospy.Subscriber("/state" + str(i), String, self.receive_vrep, queue_size=1)
        self.pub_restart = rospy.Publisher("/restart", Int8, queue_size=1)

        self.period = 50
        self.episode_length = params['max_ep_len']
        self.total_steps = 0
        self.steps = np.array([0] * num_agents)  # maps robot id to number steps received

        self.prev_local_sample = {}
        self.prev_global_sample = None

        self.u_n = len(self.action_map)
        num_input_nodes = 3 * self.num_agents  # Each agent: 2-d location, action index of all but 1. Then 1 robot id.
        neurons = [num_input_nodes] + [params['reward_width']] * params['reward_depth'] + [self.u_n]

        # state_indicators = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1, 2, 3]]
        state_indicators = [[0, 1], [0, 1, 2, 3], [0, 1]]
        self.policy = CLA(state_indicators, self.u_n, params['a'], params['b'])
        self.reward_network = RewardNetwork(neurons, params['reward_lr'])

        self.near_ball = .5
        self.near_agent = .1

        self.reward_buffer = RewardMemory()
        self.policy_buffer = PolicyMemory()

        self.policy_batch = params['policy_batch']
        self.reward_batch = params['reward_batch']
        self.explore_steps = params['explore_steps']

        rospy.spin()
        return

    def get_new_position(self, s, robot_id):
        a = self.policy.get_action(s)
        direction = self.action_map[a]
        position = self.locations[robot_id]
        return position + direction

    def publish_targets(self, targets):
        """ targets: dictionary mapping robot id to global, goal location """
        for id, target in targets.items():
            pub = self.target_publishers[id]
            msg = Vector3(target.x, target.y, 0)
            pub.publish(msg)
        return

    def receive_vrep(self, msg):
        """ Message from V-REP. Enforces only send new action to all agents if
            all agents have gone through self.period steps """
        agent = self.unpack_msg(msg)
        self.locations[agent.id] = (agent.x, agent.y)
        self.orientations[agent.id] = agent.ori
        self.steps[agent.id] += 1
        assert any(self.steps <= self.period)  # make sure each time step is accounted for
        if all(self.steps >= self.period):
            if self.total_steps > self.episode_length:
                self.restart_protocol()
            else:
                states, targets, actions = self.action_step()
                self.steps = np.array([0] * num_agents)
                self.total_steps += 1
                self.record(states, actions)
                self.train()
        return

    def restart_protocol(self):
        self.pub_restart.publish(Int8(1))
        self.total_steps = 0
        self.steps = np.array([0] * num_agents)

    def record(self, states, actions):
        """ Handle buffer updates and storage of replay """
        curr_global_state = self.generate_global_state(actions)

        if self.prev_local_sample != {}:
            # Global Information
            prev_global_state, prev_global_action = self.prev_global_sample.state, self.prev_global_sample.action

            # Reward buffer
            r = self.reward_function(prev_global_state, prev_global_action, curr_global_state.state)
            self.reward_buffer.push(prev_global_state, prev_global_action, r, self.num_agents)

            # Policy Buffer
            for i in self.prev_local_sample.keys():
                prev_s, prev_a = self.prev_local_sample[i]
                self.policy_buffer.push(prev_s, prev_a, i, prev_global_state, prev_global_action)

        # Reset prev
        self.prev_local_sample = {i: Sample(states[i], actions[i])for i in range(self.num_agents)}
        self.prev_global_sample = curr_global_state
        self.prev_ball = self.ball_location
        return

    def train(self):
        """ Handles training the CLA agent """
        # Update the reward network
        if len(self.reward_buffer) > self.explore_steps:
            r_trans = self.reward_buffer.sample(batch=self.reward_batch)
            self.reward_network.update(r_trans.global_state, r_trans.global_action,
                                       r_trans.robot_id, r_trans.reward)

        if len(self.policy_buffer) > self.policy_batch:
            p_trans = self.policy_buffer.sample()
            self.policy.update_policy(p_trans.local_state, p_trans.local_action,
                                      p_trans.global_state, p_trans.global_action,
                                      p_trans.robot_id, self.reward_network)

        return

    def reward_function(self, s, a, s_prime):
        s, s_prime = s.flatten(), s_prime.flatten()
        squared = np.square(s)
        squared_distances = squared[::2] + squared[1::2]
        prev_distance = np.sum(np.sqrt(squared_distances))

        squared = np.square(s_prime)
        squared_distances = squared[::2] + squared[1::2]
        new_distance = np.sum(np.sqrt(squared_distances))

        r_agent_distance = prev_distance - new_distance
        r_ball_forward = self.ball_location[0] - self.prev_ball[0]
        r_ball_side = abs(self.prev_ball[1]) - abs(self.ball_location[1])

        return r_agent_distance / (self.num_agents) + r_ball_forward * 10 + r_ball_side * 10

    def action_step(self):
        """ Creates new targets for all agents and then sends to all agents """
        targets, local_states, actions = {}, {}, {}
        for i in range(self.num_agents):
            local_states[i] = self.to_state_id(i)
            actions[i] = self.policy.get_action(local_states[i])
            print(actions[i], len(self.reward_buffer), len(self.policy_buffer))
            direction = self.action_map[actions[i]]
            position = self.locations[i]
            target = tuple(np.add(position, direction))
            targets[i] = Target(*target)
        self.publish_targets(targets)
        return local_states, targets, actions

    def unpack_msg(self, msg):
        """ Handles unpacking message from v-rep"""
        state = simxUnpackFloats(msg.data)
        state[3] = int(state[3])
        agent = Agent(*state)
        return agent

    def receive_ball(self, msg):
        """ Callback function for ball location """
        x, y = msg.x, msg.y
        self.ball_location = (x, y)
        return

    def generate_global_state(self, actions_dict):
        """ Generates global state to pass into reward network.
            Transforms all locations to relative to BALL """
        locations_relative_to_ball, all_actions = [], []
        for i in range(self.num_agents):
            locations_relative_to_ball.append(np.subtract(self.locations[i], self.ball_location))
            all_actions.append(actions_dict[i])
        state, all_actions = np.array(locations_relative_to_ball).flatten(), np.array(all_actions).flatten()
        return Sample(state, all_actions)

    def to_state_id(self, robot_id):
        """ Generates local, discrete state for robot number {robot_id}

            Indicators: near ball (1), ball location (4), y sign (1), on left (1), on right(1), front(1), back (1)
            Total: 256 states. Lots of them. But, we have lots of robots.

            Simpler: near ball (1), ball location (4), y sign (1)
            Total: 16
            """
        robot_location, robot_ori = self.locations[robot_id], self.orientations[robot_id]
        near_ball = int(self.distance(robot_location, self.ball_location) < self.near_ball)
        ball_location = self.determine_direction(robot_location, robot_ori, self.ball_location)
        ball_y = 0 if self.ball_location[1] > 0 else 1
        near_agent = [0, 0, 0, 0]
        for i in range(self.num_agents):
            if i != robot_id:
                dist = self.distance(robot_location, self.locations[i])
                if dist < self.near_agent:
                    direction = self.determine_direction(robot_location, robot_ori, self.locations[i])
                    near_agent[direction] = 1
        features = [near_ball, ball_location, ball_y]# near_agent + [near_ball, ball_location]
        id = 0
        for i, f in enumerate(features):
            id += f * (10 ** i)
        return id

    def distance(self, point1, point2):
        assert len(point1) == len(point2)
        return np.sum(np.square(np.subtract(point1, point2)))

    def determine_direction(self, location, orientation, target):
        """ Return 0, 1, 2, 3 for front, left, back, right """
        vector = np.subtract(target, location)
        target_vector = vector / np.linalg.norm(vector)
        ori_vector = np.array([np.cos(orientation), np.sin(orientation)])
        ori_y_vector = np.array([np.cos(orientation + np.pi/2), np.sin(orientation + np.pi/2)])
        along_x = np.dot(ori_vector, target_vector)
        along_y = np.dot(ori_y_vector, target_vector)
        if abs(along_y) > along_x:
            return 1 if along_y > 0 else 3
        else:
            return 0 if along_x > 0 else 2


if __name__ == '__main__':
    rospy.init_node('Dum', anonymous=True)
    num_agents = rospy.get_param('~num_bots')
    params = {'policy_batch': 256, 'a': 1,  'b': 1,
              'reward_width': 200, 'reward_depth': 3, 'reward_lr': 3e-4,
              'reward_batch': 256, 'explore_steps': 10,
              'max_ep_len': 10, }
    agent = MultiAgent(num_agents, params)
