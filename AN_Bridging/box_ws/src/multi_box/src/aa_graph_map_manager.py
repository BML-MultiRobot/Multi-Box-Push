#! /usr/bin/env python
"""
Uses the graphMap and runs the algorithm in conjunction with V-Rep
"""
import matplotlib.pyplot as plt
from Tasks.data_analysis import Analysis
import rospy, sys, vrep
from std_msgs.msg import String, Int16, Float32MultiArray, Int8
from geometry_msgs.msg import Vector3
from aa_graphMap import StigmergicGraphVREP
import aa_graphMap_node_simulation
from copy import deepcopy
import networkx as nx
import numpy as np
import pickle
import time

path = '/home/jimmy/Documents/Research/AN_Bridging/results/policy_comparison_results/all_final/state_data.txt'


class Graph_Map_Manager(object):
    def __init__(self, num_agents):
        """ Data Analysis for Classification """
        self.data_analyzer = Analysis(path)
        self.prob_threshold = .7
        self.episode = None

        """ Topics for V-REP episode management """
        rospy.Subscriber('/finished', Int8,  self.receive_time_is_up, queue_size=1)
        rospy.Subscriber('/starting', Int16, self.receive_episode_number, queue_size=1)

        """ Training Classifiers """
        self.rf = self.data_analyzer.get_classifiers(estimators=200, depth=5)
        self.classifier = self.rf

        """ Initializing Map and Relevant Variables"""
        self.num_agents = num_agents
        self.most_recently_assigned_positions = {}
        self.map = StigmergicGraphVREP()

        """ Topics for Initial Map Building"""
        self.map_generator_request = rospy.Publisher('/map_information_request', Int16, queue_size=1)
        rospy.Subscriber('/map_information_response', Float32MultiArray, self.receive_initial_map_response)
        self.map_finished = rospy.Publisher('/map_is_ready', Int16, queue_size=1)

        """ Topics for agent management, node assignments and stigmergic algorithm"""
        rospy.Subscriber("/robot_finished", Int16, self.receive_finish_indicator_from_robot, queue_size=10)
        self.agent_to_target_pubs, self.agent_to_neighbor_request, self.agent_to_neighbor_response_topic = {}, {}, {}
        self.agent_to_shut_down = {}
        for i in range(self.num_agents):
            self.agent_to_target_pubs[i] = rospy.Publisher("/target" + str(i), Vector3, queue_size=10)
            self.agent_to_neighbor_request[i] = rospy.Publisher('/neighbor_states' + str(i), String, queue_size=10)
            self.agent_to_shut_down[i] = rospy.Publisher("/shutdown" + str(i), Int16, queue_size=1)
            self.agent_to_neighbor_response_topic[i] = response_topic = '/neighbor_response' + str(i)
            rospy.Subscriber(response_topic, Float32MultiArray, self.receive_neighbor_box_states)

        """ V-REP Map Retrieval and Conversion to Nodes for Stigmergic Algorithm """
        rospy.sleep(2)
        request = Int16(1)
        self.map_generator_request.publish(request)

        map_information = rospy.wait_for_message('/map_information_response', Float32MultiArray)
        map_information = map_information.data
        map_information = self.parse_split_list_by_delimiter(map_information, -float('inf'))

        nodes = self.parse_split_list_by_delimiter(map_information[0], float('inf'))
        robots = self.convert_list_of_lists_to_int(self.parse_split_list_by_delimiter(map_information[1], float('inf')))
        inclusions = self.convert_list_of_lists_to_int(self.parse_split_list_by_delimiter(map_information[2], float('inf')))
        exclusions = self.convert_list_of_lists_to_int((self.parse_split_list_by_delimiter(map_information[3], float('inf'))))
        boxes = list(map(lambda lst: list(map(int, lst[:2])) + [lst[2]], self.parse_split_list_by_delimiter(map_information[4], float('inf'))))
        goal = int(self.parse_split_list_by_delimiter(map_information[5], float('inf'))[0][0])
        nodes = list(map(lambda lst: lst[:3] + [int(lst[3])], nodes))
        self.create_map_using_vrep_markers(nodes, robots, inclusions, exclusions, boxes, goal)

        """ Initializing Networkx Graph Visualizer for Easier Debugging """
        self.visualizer = aa_graphMap_node_simulation.Trainer([], [], [], [], [], None)
        self.visualizer.start_state = deepcopy(self.map)
        self.visualizer.current_environment = self.map
        self.visualizer.goal_index = goal
        self.display_graph = nx.grid_2d_graph(2, 2)
        self.visualizer.update_display_graph_using_current_environment(self.display_graph, self.episode)
        plt.ion()
        plt.show()
        self.map_finished.publish(Int16(1))

        """ For keeping track of episode termination """
        self.restart_publisher = rospy.Publisher("/restart", Int8, queue_size=1)
        self.robot_steps = [0 for i in range(self.num_agents)]
        self.curr_episode = 1
        self.max_steps = 50
        self.max_episodes = 30
        self.max_time_in_seconds = 800  # just in case the policy doesn't do very well or something goes wrong...
        self.total_steps = []
        self.success = []

        """ While Loop to Replace rospy.spin() because visualizer needs to be called in main loop """
        start = time.time()
        while True:
            rospy.wait_for_message('/robot_finished', Int16)
            self.visualizer.update_display_graph_using_current_environment(self.display_graph, self.episode)
            reached_goal = [self.has_reached_goal(robot_id) for robot_id in range(self.num_agents)]
            curr_time = time.time() - start
            if not all(reached_goal):
                steps = min([self.robot_steps[robot_id] for robot_id in range(self.num_agents) if not reached_goal[robot_id]])
                print('Until episode termination: ', max(float(steps / float(self.max_steps)), curr_time / float(self.max_time_in_seconds)))
                timed_out = steps > self.max_steps or curr_time > self.max_time_in_seconds
            else:
                steps = max(self.robot_steps)
                timed_out = False
            if all(reached_goal) or timed_out:
                self.restart_publisher.publish(Int8(1))
                self.receive_time_is_up(1)
                self.curr_episode += 1
                self.total_steps.append(steps if curr_time <= self.max_time_in_seconds else self.max_steps)
                self.success.append(all(reached_goal))
                start = time.time()
                print('')
                print('Running total steps: ', self.total_steps)
                print('')
                print('Running success: ', self.success)
                print('')
                self.robot_steps = [0 for i in range(self.num_agents)]
                rospy.wait_for_message('/starting', Int16)
                time.sleep(1)
                for i in range(self.num_agents):
                    self.receive_finish_indicator_from_robot(Int16(i))
            if self.curr_episode > self.max_episodes:
                break

        self.data_to_txt(data=self.total_steps,
                         path='/home/jimmy/Documents/Research/AN_Bridging/results/stigmergic_node_data/total_steps.txt')
        self.data_to_txt(data=self.success,
                         path='/home/jimmy/Documents/Research/AN_Bridging/results/stigmergic_node_data/success.txt')

        sys.exit(0)

    def data_to_txt(self, data, path):
        with open(path, "wb") as fp:  # Pickling
            pickle.dump(data, fp)
        return

    def receive_time_is_up(self, msg):
        """ Receive msg from manager.py that time is up and reset the environment"""
        self.map.post_episode_pheromone_update()
        self.visualizer.reset_environment_but_preserve_trained_data()
        self.map = self.visualizer.current_environment
        self.most_recently_assigned_positions = deepcopy(self.initial_positions)
        for i, shut_pub in self.agent_to_shut_down.items():
            shut_pub.publish(Int16(0))
        return

    def receive_episode_number(self, msg):
        """ Receives the episode number from manager.py to properly sync with V-REP"""
        self.episode = msg.data

    def convert_list_of_lists_to_int(self, lst_of_lsts):
        """ Converts each element in list of lists to integer. Used when converting V-REP information to nodes"""
        return list(map(lambda lst: list(map(int, lst)), lst_of_lsts))

    def parse_split_list_by_delimiter(self, lst, delimiter):
        """ Split list into list of lists by looking for delimiter in the list."""
        if len(lst) == 0:
            return lst
        lst = lst[1:] if lst[0] == delimiter else lst
        result, curr = [], []
        for elem in lst:
            if elem == delimiter:
                result.append(curr)
                curr = []
            else:
                curr.append(elem)
        result.append(curr) if len(curr) > 0 else None
        return result

    def receive_initial_map_response(self, msg):
        """ Dummy callback method for retrieving map information """
        return

    def receive_neighbor_box_states(self, msg):
        """ Dummy callback method for retrieving 'states' corresponding to nodes neighboring node with box"""
        return

    def create_map_using_vrep_markers(self, nodes, robots, inclusions, exclusions, boxes, goal):
        """ Creates a map using VREP markers, robots, and boxes """
        # save = {'nodes': nodes, 'robots': robots, 'inclusions': inclusions, 'exclusions': exclusions, 'boxes': boxes, 'goal': goal}
        # path = '/home/jimmy/Documents/Research/AN_Bridging/results/vrep_env_info.json'
        # import json
        # with open(path, 'w') as fp:
        #     json.dump(save, fp)
        # print(yes)
        inclusions = [tuple(tup) for tup in inclusions]
        exclusions = [tuple(tup) for tup in exclusions]
        self.map.convert_to_nodes(nodes, inclusions, exclusions, boxes, robots, goal, vrep_simulation=True)
        for node_index, robot_id in robots:
            self.most_recently_assigned_positions[robot_id] = node_index
        self.initial_positions = deepcopy(self.most_recently_assigned_positions)
        return

    def receive_finish_indicator_from_robot(self, msg):
        """ Receive indicator from robot that it has reached its assigned location. Assign it a new one. """
        robot_id = msg.data
        self.robot_steps[robot_id] += 1
        if self.map.robots[robot_id].current_node:
            self.map.update_agent_location(robot_id, self.most_recently_assigned_positions[robot_id])
        if self.has_reached_goal(robot_id):
            self.agent_to_shut_down[robot_id].publish(Int16(1))
        else:
            self.send_target_to_robot_in_vrep(robot_id)


    def has_reached_goal(self, robot_id):
        """ A robot has reached the goal if its current_node is null...it's been 'removed' from the map """
        return not self.map.robots[robot_id].current_node

    def send_target_to_robot_in_vrep(self, robot_id):
        """ Send new target node index to agent. Sent to VREP client. """
        target_node_index, reference_node_index, box_index, pheromone = self.get_new_target_from_map(robot_id)
        if type(target_node_index) != int and not target_node_index:
            return
        print(' Sending robot ', robot_id, ' to ', target_node_index, ' using pheromone: ', pheromone, '   Explore: ', self.map.robots[robot_id].explore)
        self.most_recently_assigned_positions[robot_id] = target_node_index
        msg = Vector3(x=box_index, y=target_node_index, z=reference_node_index)
        self.agent_to_target_pubs[robot_id].publish(msg)

    def get_new_target_from_map(self, robot_id):
        """ Get new target for specified robot using stigmergic map algorithm """
        traversable_box_nodes = self.get_dict_box_id_to_pushable_nodes(robot_id)  # dictionary mapping box id to node ids
        target_index, box_index, pheromone = self.map.get_agent_target(robot_id, traversable_box_nodes)
        return target_index, self.most_recently_assigned_positions[robot_id], box_index, pheromone

    def get_dict_box_id_to_pushable_nodes(self, robot_id):
        """ Return dictionary mapping box id to set of node ids that are the agent is likely able to push box to"""
        box_request, curr_node_index = self.map.get_box_nodes(robot_id)  # Returns a dictionary mapping box index to list of nodes to refer to
        eligible_neighbor_states = {}  # maps box id to node id it can travel to
        for box_id, lst_of_nodes in box_request.items():
            msg = String(data=vrep.simxPackFloats([box_id] + lst_of_nodes + [curr_node_index]))
            self.agent_to_neighbor_request[robot_id].publish(msg)
            lst_of_states = rospy.wait_for_message(self.agent_to_neighbor_response_topic[robot_id], String).data
            lst_of_states = self.parse_split_list_by_delimiter(lst_of_states, float('inf'))
            lst_of_states = map(lambda x: np.array(x).reshape((1, -1)), lst_of_states)
            node_id_to_states = dict(zip(lst_of_nodes, lst_of_states))
            eligible_neighbor_states[box_id] = {node_id for node_id, s in node_id_to_states.items() if self.prob_succeed(s) >= self.prob_threshold}
        return eligible_neighbor_states

    def prob_succeed(self, state):
        predicted_to_succeed = self.classifier.predict_proba(np.array(state)).flatten()[1]
        print('Probability push succeed: ', predicted_to_succeed)
        return predicted_to_succeed


if __name__ == "__main__":
    rospy.init_node('Dummy', anonymous=True)
    num_agents = rospy.get_param('~num_agents')
    graph = Graph_Map_Manager(num_agents)
