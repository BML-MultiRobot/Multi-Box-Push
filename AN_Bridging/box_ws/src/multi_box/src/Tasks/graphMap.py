import numpy as np 
from task import distance as dist
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import heapq

# Logistical parameters
PHEROMONE_ID_STAGGER = 100
INITIAL_BOX_DISTANCE_SET = .1
RADIUS = 1.9  # radius for node creation (not actually for algorithm)
SPEED = 1  # time in seconds per step for visualization
MAX_DISTANCE_SET = 6  # 10 for large # 6 for small environments

# Algorithm hyper-parameters
EPISODES = 100
MAX_STEPS = 15

DETECTION_RADIUS = 2
FRACTION_OF_AGENTS = .5
EXPLORE_DECAY = .95
START_EXPLORE = .3

B_preference_decay = .9  # Decays every time we attempt to place a box
E_spatial_decay = 2
E_temporal_decay = .7


""" Convention: 
        -2 pheromone: exploration. More density -> more explored
        -1 pheromone: classic distance pheromone placed anywhere 
        0 -> indices < PHEROMONE_ID_STAGGER: node indices of self.nodes that correspond to candidate holes 
        PHEROMONE_ID_STAGGER >= : len(self.boxes) indices that correspond to various boxes on the map
"""


def stigmergic_main(nodes, inclusions, exclusions, box_data, robot_data, goal_index):
    trainer = Trainer(nodes, inclusions, exclusions, box_data, robot_data, goal_index)
    trainer.main_algorithm()


class Trainer(object):
    def __init__(self, nodes, inclusions, exclusions, box_data, robot_data, goal_index):
        self.start_state = StigmergicGraph()
        self.start_state.convert_to_nodes(nodes, inclusions, exclusions, box_data, robot_data, goal_index)
        self.current_environment = None

    def main_algorithm(self):
        self.current_environment = deepcopy(self.start_state)
        indicator_success_each_episode = []
        display_graph = nx.grid_2d_graph(3, 2)
        self.update_display_graph_using_current_environment(display_graph)
        plt.ion()
        plt.show()
        for i in range(EPISODES):
            print('')
            print(' ##### EPISODE: ', i + 1)

            achieved_goal = False
            restart = False
            self.current_environment.pre_episode_pheromone_update()
            step = 1
            while not achieved_goal and not restart and step <= MAX_STEPS:
                print('')
                print(' Step number: ', step)

                self.update_display_graph_using_current_environment(display_graph)
                plt.draw()
                plt.pause(.2)

                achieved_goal, restart = self.current_environment.one_step()
                step += 1
            print('Achieved goal: ', achieved_goal)

            self.update_display_graph_using_current_environment(display_graph)
            plt.draw()
            plt.pause(SPEED)

            self.current_environment.post_episode_pheromone_update()
            indicator_success_each_episode.append(int(achieved_goal))
            self.reset_environment_but_preserve_trained_data()
        # x = range(len(indicator_success_each_episode))
        # plt.title("Moving Average Success")
        # plt.legend()
        # line_success = Analysis.get_moving_average(indicator_success_each_episode, 10)
        # plt.plot(x, line_success, 'r')
        # plt.show()

    def update_display_graph_using_current_environment(self, networkx_graph):
        networkx_graph.clear()
        plt.clf()
        plt.xlim(-1, 6)
        plt.ylim(-1, 2)
        coordinates = {i: (node.coords[0], node.coords[1]) for i, node in enumerate(self.current_environment.nodes)}
        networkx_graph.add_nodes_from(coordinates.keys())
        all_edges = set()
        for i, node in enumerate(self.current_environment.nodes):
            curr_edges = [(self.current_environment.nodes.index(n), i) if self.current_environment.nodes.index(n) < i
                          else (i, self.current_environment.nodes.index(n)) for n in node.traversable_neighbors]
            all_edges = all_edges.union(set(curr_edges))
        networkx_graph.add_edges_from(list(all_edges))
        colors_for_pheromones = [n.pheromones[-1] for n in self.current_environment.nodes]

        # Draw the pheromones
        plt.subplot(321)
        plt.title('Distance Pheromone')
        nx.draw(networkx_graph, node_color=colors_for_pheromones, pos=coordinates, node_size=500, with_labels=True, cmap=plt.cm.Greens)

        # Draw the current location of the agent
        plt.subplot(322)
        plt.title('Agent Locations')
        index_current_agents = [self.current_environment.nodes.index(agent.current_node) for agent in self.current_environment.robots]
        colors_of_agents = [50 if index in index_current_agents else 0 for index in range(len(self.current_environment.nodes))]
        nx.draw(networkx_graph, node_color=colors_of_agents, pos=coordinates, node_size=500, with_labels=True, cmap=plt.cm.Reds)

        # Draw each of the box node's current_pheromone_value if a node has a box. Otherwise, leave it alone
        plt.subplot(324)
        plt.title('Box Value')
        color_boxes = [sum([box.current_pheromone_value for box in node.box])/len(node.box) if len(node.box) > 0 else 0 for node in self.current_environment.nodes]
        nx.draw(networkx_graph, node_color=color_boxes, pos=coordinates, node_size=500, with_labels=True, cmap=plt.cm.Blues)

        # Draw indications of where the boxes currently are
        plt.subplot(325)
        plt.title('Indicator Box Location')
        color_box_locations = [50 if node.box else 0 for node in self.current_environment.nodes]
        nx.draw(networkx_graph, node_color=color_box_locations, pos=coordinates, node_size=500, with_labels=True, cmap=plt.cm.Reds)

        # Draw indications of where the holes are
        plt.subplot(326)
        plt.title('Location of Hole Locations')
        color_hole_locations = [50 if node.is_source else 0 for node in self.current_environment.nodes]
        nx.draw(networkx_graph, node_color=color_hole_locations, pos=coordinates, node_size=500, with_labels=True, cmap=plt.cm.Reds)

    def reset_environment_but_preserve_trained_data(self):
        new_environment = deepcopy(self.start_state)
        new_environment.exclusions = self.current_environment.exclusions
        new_environment.inclusions = self.current_environment.inclusions
        # Iterate in list order
        for k, node in enumerate(new_environment.nodes):
            node.pheromones = self.current_environment.nodes[k].pheromones
            node.distance_to_goal = self.current_environment.nodes[k].distance_to_goal
            node.official = self.current_environment.nodes[k].official
        for k, box in enumerate(new_environment.boxes):
            box.placement_preferences = self.current_environment.boxes[k].placement_preferences
        for k, bot in enumerate(new_environment.robots):
            bot.explore = self.current_environment.robots[k].explore
        # copy the pheromones, box successful/attempted placements.
        self.current_environment = new_environment


class StigmergicGraph(object):
    def __init__(self):
        self.nodes = []
        self.robots = []
        self.boxes = []
        self.inclusions = []
        self.exclusions = []
        self.pheromone_set = {}  # Indicates which indices in self.nodes correspond to hole with pheromone.
        self.placed_box_indices = set()
        self.goal = None

    def one_step(self):
        # After initialization, in order of decreasing value of agent (or something similar) decide who is moving

        agent_values = []
        for agent in self.robots:
            # list of tuples (amount of pheromone, node neighbor) in order of pheromone index
            agent_value = agent.get_value_of_agent(self)
            agent_values.append((agent_value, agent))
        if all([v < 0 for v, agent in agent_values]):
            return False, True

        sorted_by_value = sorted(agent_values, key=lambda x: x[0], reverse=True)
        num_agents_execute = max(int(FRACTION_OF_AGENTS * len(self.robots)), 1)
        i = 0
        agents_with_changed_targets = []
        while i < num_agents_execute:
            _, agent = sorted_by_value.pop(0)
            # TODO: Implement previous target choice! Then, move one step per depending on current task
            agent.choose_target(self)
            agents_with_changed_targets.append(agent)
            i += 1

        all_touched_nodes = []
        for agent in agents_with_changed_targets:
            """ 
            If the pheromone is -1, then we are moving probabilistically towards higher D pheromones. 
            If between 0 and PHEROMONE_ID_STAGGER, we are moving a box directly to a hole candidate. 
            If greater than PHEROMONE_ID_STAGGER, then we are moving directly to a box."""
            if PHEROMONE_ID_STAGGER > agent.target_pheromone >= 0:
                path = self.get_path_to_hole(agent.current_node, agent.target_node)
                last_node_in_path = path[-1]  # Last node in path to target. If unreachable, this is just current_node
                agent_new_location = path[-2]
                path = path[:-1]  # This represents path up to and including agent_new_location

                box = agent.current_node.remove_box()
                agent.target_node.place_box(box)
                box.placement_preferences[self.nodes.index(agent.target_node)] *= B_preference_decay

                # Assert that we only remove and place once and assert that there is a path to the box candidate place
                assert not self.box_has_been_moved(box)
                assert agent.target_node == last_node_in_path

                self.placed_box_indices.add(self.boxes.index(box))
                self.update_neighbors_to_reflect_box_change(agent.current_node)
                self.update_neighbors_to_reflect_box_change(agent.target_node)
                box.current_node = last_node_in_path
                print('Robot number: ', self.robots.index(agent), ' followed pheromone: ', agent.target_pheromone,
                      '. Moved BOX to node : ', self.nodes.index(agent.target_node), ' And robot ended at node: ',
                      self.nodes.index(agent_new_location))
            else:
                path = [agent.current_node, agent.target_node]  # TODO: This assumes we only travel in steps of ONE
                last_node_in_path = agent.target_node
                agent_new_location = last_node_in_path
                print('Robot number: ', self.robots.index(agent), ' followed pheromone: ', agent.target_pheromone,
                      '. Moved ITSELF to node : ', self.nodes.index(agent_new_location))
            self.add_d_pheromones_to_path(path)
            self.add_e_pheromones_to_path(path)
            all_touched_nodes.extend(path)
            agent.current_node = agent_new_location
        self.update_pheromones(all_touched_nodes)

        if any([agent.current_node == self.goal for agent in self.robots]):
            return True, True
        return False, False

    def print_all_pheromones_in_order(self):
        for i, node in enumerate(self.nodes):
            print(' Node index: ', i, ' Distances: ', node.distance_to_goal, ' Pheromone: ', node.pheromones[-1])
            # print(' Node index: ', i, ' Pheromones: ', node.pheromones)
        """for i, box in enumerate(self.boxes):
            print(' Box index: ', i, ' Attached pheromone: ', box.current_pheromone_value)"""

    def update_neighbors_to_reflect_box_change(self, current_node):
        all_neighbors = current_node.neighbors
        all_neighbors = all_neighbors + [current_node]
        for node in all_neighbors:
            neighbors = self.get_neighbors(node, RADIUS, include_intraversable=True)
            traversable_neighbors = self.get_neighbors(node, RADIUS, include_intraversable=False)
            node.neighbors = neighbors
            node.traversable_neighbors = traversable_neighbors
        return

    def add_d_pheromones_to_path(self, path):
        official_node_set = {n for n in self.nodes if n.official}
        for node in path[1:]:  # Update D pheromone along the path. Pheromone index is -1. Don't drop on the first
            if node == self.goal:
                node.official = True
                node.distance_to_goal = 0
            else:
                traversable_neighbors = node.traversable_neighbors
                official_neighbors = [neighbor for neighbor in traversable_neighbors if neighbor in official_node_set]
                is_now_official = len(official_neighbors) > 0
                has_neighbors_right_now = len(traversable_neighbors) > 0
                already_is_official = node.official
                if is_now_official and has_neighbors_right_now:
                    closest_to_goal = max(official_neighbors, key=lambda neighbor: neighbor.distance_to_goal + self.get_distance_between_nodes(node, neighbor))
                    official_distance = closest_to_goal.distance_to_goal + self.get_distance_between_nodes(node, closest_to_goal)
                    if already_is_official:
                        node.distance_to_goal = min(node.distance_to_goal, official_distance)
                    else:
                        node.distance_to_goal = official_distance
                    node.official = True
                else:
                    if not already_is_official:
                        node.distance_to_goal = self.get_distance_between_nodes(node, self.goal)

            pheromone_set_to = -node.distance_to_goal + MAX_DISTANCE_SET
            node.handle_adding_d_pheromone(pheromone_set_to, self)

    def get_distance_between_nodes(self, first, second):
        return dist(first.pos[:2], second.pos[:2])

    def get_detection_distance_between_nodes(self, first, second):
        first_position = deepcopy(first.pos)
        second_position = deepcopy(second.pos)
        first_position[2] = first_position[2] - sum([b.height for b in first.box])
        second_position[2] = second_position[2] - sum([b.height for b in second.box])
        return dist(first_position, second_position)


    def add_e_pheromones_to_path(self, path):
        for node in path:  # Update E pheromone but include the first node as well
            node.pheromones[-2] += 1

    def exists_path_box_to_hole(self, start, finish):
        path = self.get_path_to_hole(start, finish, from_box_to_hole=True)
        return path[-1] == finish

    def get_path_to_hole(self, start, finish, from_box_to_hole=True):
        priority_queue = []
        node_to_prev_node = {}  # maps a node to the previous node that had called dijkstra on it
        visited = set()

        heapq.heappush(priority_queue, (0, start))
        visited.add(start)
        iteration = 0

        while True:
            if len(priority_queue) == 0:
                break
            distance_to_curr, curr_node = heapq.heappop(priority_queue)
            # assert curr_node not in visited
            visited.add(curr_node)
            if curr_node == finish:
                break

            neighbors = curr_node.neighbors if (from_box_to_hole and curr_node == start) else curr_node.traversable_neighbors

            if finish in curr_node.neighbors and finish not in neighbors:  # If the hole is nearby, then append to neighbors
                neighbors = neighbors + [finish]
            for n in neighbors:
                if n not in visited:  # if it hasn't been visited, there's a possibility of getting better path
                    nodes_in_pq = (list(map(lambda x: x[1], priority_queue)))
                    if n in nodes_in_pq:  # if it's currently in the pq, update the priority
                        index = nodes_in_pq.index(n)
                        curr_dist = distance_to_curr + dist(n.pos, curr_node.pos)
                        prev_dist = priority_queue[index][0]
                        if curr_dist < prev_dist:
                            priority_queue[index] = (curr_dist, n)
                            node_to_prev_node[n] = curr_node
                    else:  # otherwise add it to the pq
                        heapq.heappush(priority_queue, (distance_to_curr + dist(n.pos, curr_node.pos), n))
                        node_to_prev_node[n] = curr_node
            prev_node = curr_node
            iteration += 1
        # Getting the path:
        if finish not in node_to_prev_node.keys():
            return [start]  # Returns only the first node if there is no path

        curr_node = finish
        path = [finish]
        while curr_node != start:
            curr_node = node_to_prev_node[curr_node]
            path.insert(0, curr_node)
        return path

    def boxes_calculate_values(self):
        for i, box in enumerate(self.boxes):
            subset_of_candidates_with_path = [value for index, value in box.placement_preferences.items()
                                              if self.exists_path_box_to_hole(box.current_node, self.nodes[index])]
            box_value = max(subset_of_candidates_with_path)
            # TODO: Nullify the box value if it has already been claimed
            box.current_pheromone_value = box_value

    def update_pheromones(self, all_touched_nodes):
        self.boxes_calculate_values()
        self.spread_non_box_pheromones_along_path(all_touched_nodes)
        self.decay(all_touched_nodes)
        self.add_temp_pheromones()

    def spread_non_box_pheromones_along_path(self, path):
        for node in path:
            neighbors = node.traversable_neighbors
            for n in neighbors:
                if n.pheromones[-2] < node.pheromones[-2]:
                    magnitude = node.pheromones[-2]
                    n.temp_pheromones[-2] += magnitude * np.exp(-E_spatial_decay * dist(n.pos, node.pos))

    def box_has_been_moved(self, box):
        return self.boxes.index(box) in self.placed_box_indices

    def decay(self, all_touched_nodes):
        for node in all_touched_nodes:
            pheromones = node.pheromones
            pheromones[-2] *= E_temporal_decay

    def add_temp_pheromones(self):
        for node in self.nodes:
            pheromones = {p: node.pheromones[p] + node.temp_pheromones[p] for p in node.pheromones.keys()}
            node.pheromones = pheromones
            node.temp_pheromones = {p: 0 for p in self.pheromone_set}

    def pre_episode_pheromone_update(self):
        self.boxes_calculate_values()
        return

    def post_episode_pheromone_update(self):
        # Get rid of all non-negative pheromones that are not from sources. Keep D and E pheromones as well.
        for pheromone_id, n in enumerate(self.nodes):
            new_pheromones = {p: 0 for p in self.pheromone_set}
            new_pheromones[-1] = n.pheromones[-1]
            new_pheromones[-2] = n.pheromones[-2]
            n.pheromones = new_pheromones
        return

    """ ############################## Insertion and Initialization of the Graph ################################ """

    def get_vrep_input_and_convert_to_nodes(self, node_data_msg, inclusions_msg, exclusions_msg, box_msg, robot_msg, goal_msg):
        nodes = node_data_msg.data  # List of tuples. Tuples correspond to node coordinates and box_id (0 if no box)
        inclusions = inclusions_msg.data 
        inclusions = [(nodes[index1][0], nodes[index2][0]) for index1, index2 in inclusions]
        exclusions = exclusions_msg.data
        exclusions = [(nodes[index1][0], nodes[index2][0]) for index1, index2 in exclusions]
        box_data = box_msg.data 
        robot_data = robot_msg.data
        goal_index = goal_msg.data
        self.convert_to_nodes(nodes, inclusions, exclusions, box_data, robot_data, goal_index)

    def convert_to_nodes(self, nodes, inclusions, exclusions, box_data, robot_data, goal_index):
        # Insert all the nodes into the graph and initialize goal
        self.insert_nodes(nodes, RADIUS, inclusions, exclusions)
        self.goal = self.nodes[goal_index]
        pheromone_set = {-1, -2}

        # Insert all boxes in correct places and change heights of positions. Get relevant pheromone set
        for index_of_current_node, box_id, height in box_data:
            current_node = self.nodes[index_of_current_node]
            candidate_nodes = [node for node in self.nodes if node.box_id == box_id]
            candidate_node_indices = [i for i, node in enumerate(self.nodes) if node.box_id == box_id]
            new_box = Box(height, candidate_nodes, box_id, candidate_node_indices, current_node)
            current_node.place_box(new_box)
            pheromone_set.update([PHEROMONE_ID_STAGGER + len(self.boxes)])
            self.boxes.append(new_box)
        self.pheromone_set = pheromone_set

        for node in self.nodes:
            self.update_neighbors_to_reflect_box_change(node)

        # Initialize the pheromone set of all nodes
        for node in self.nodes:
            # Set 1 for all source nodes' corresponding pheromone. Set 1 for all D pheromone. Set 0 otherwise.
            node.pheromones = {p: MAX_DISTANCE_SET if p == -1 else 0 for p in self.pheromone_set}
            node.temp_pheromones = {p: 0 for p in self.pheromone_set}
        # Initialize the robots into the graph
        for robot_curr_node, robot_id in robot_data:
            self.robots.append(StigmergicAgent(self.nodes[robot_curr_node], robot_id))

    def insert_nodes(self, coords, radius, inclusions, exclusions):
        """
        Takes a list of nodes (4-d tuples) to add into the graph. Radius represents closeness to add an edge. Exclusions
        is list of tuples representing exclusions from edges
        """
        self.inclusions = inclusions
        self.exclusions = exclusions
        for coord in coords:
            curr = Node(coord)
            self.nodes.append(curr)
        for node in self.nodes:
            neighbors = self.get_neighbors(node, radius, include_intraversable=True)
            traversable_neighbors = self.get_neighbors(node, radius, include_intraversable=False)
            node.neighbors = neighbors
            node.traversable_neighbors = traversable_neighbors

    def get_neighbors(self, node, radius, include_intraversable=True):
        """
        node: Node class instance
        radius: desired radius to be considered neighbors
        inclusions: node index to node index to include in edges
        exclusions: node index to node index to exclude in edges
        """
        neighbors = []
        curr_index = self.nodes.index(node)
        for other_index, node_other in enumerate(self.nodes):
            if (curr_index, other_index) in self.exclusions or (other_index, curr_index) in self.exclusions:
                continue
            elif self.get_distance_between_nodes(node, node_other) <= radius and node_other != node and \
                    (include_intraversable or abs(node.z - node_other.z) < .1):
                neighbors.append(node_other)
            elif (curr_index, other_index) in self.inclusions or (other_index, curr_index) in self.inclusions:
                neighbors.append(node_other)
        return neighbors

    """ #################################### Planning Utils ############################################## """

    def get_next_relative_coords(self, relative_pos, global_pos, node1, node2):
        """
        Given the robot perceives node1 to be at position "relative" to it and robot is at global_pos, find the position of node2 relative to the robot
        """
        one_to_two = node2.pos - node1.pos
        rob_to_one = node1.pos - global_pos

        rob_to_two_global = rob_to_one + one_to_two

        rotation = np.linalg.solve(rob_to_one, relative_pos).T
        relative_coords = np.matmul(rotation, one_to_two)
        return relative_coords


class Box(object):
    def __init__(self, height, candidate_nodes, box_id, candidate_indices, current_node):
        # Height is height of box
        # candidate_nodes is list of nodes representing possible places it can be placed
        self.box_id = box_id
        self.height = height
        self.current_node = current_node
        self.candidates = candidate_nodes
        self.candidate_indices = candidate_indices   # Hole indices this box can be placed in
        self.placement_preferences = {c: MAX_DISTANCE_SET for c in self.candidate_indices}
        self.current_pheromone_value = MAX_DISTANCE_SET
        self.claimed = False


class StigmergicAgent(object):
    def __init__(self, current_node, robot_id):
        self.value = 0
        self.robot_id = robot_id
        self.current_node = current_node
        self.target_pheromone = None
        self.target_node = None
        self.explore = START_EXPLORE

    def get_value_of_agent(self, graph):
        box = None if all([graph.box_has_been_moved(b) for b in self.current_node.box]) else self.current_node.box[-1]
        if box:
            return max(box.placement_preferences.values())
        else:
            maximums = [p[0] for p in self.get_max_pheromones_assuming_not_near_box(graph)]
            return max(maximums) if len(maximums) > 0 else -1

    def get_max_pheromones_assuming_not_near_box(self, graph):
        """
        Assumes that they agent is not at a moveable box
        Returns list of tuples (max_pheromone_concentration, pheromone_index, max_node_associated_with_pheromone)
        """
        possible_pheromones = [PHEROMONE_ID_STAGGER + i for i in range(len(graph.boxes)) if i not in graph.placed_box_indices]
        if len(self.current_node.traversable_neighbors) > 0:
            possible_pheromones.append(-1)

        neighbors = self.current_node.neighbors
        max_pheromones_to_associated_node = []
        for pheromone_index in possible_pheromones:
            # TODO: Insert collision detection weights here
            if pheromone_index >= PHEROMONE_ID_STAGGER:  # if there is a box nearby
                box_index = pheromone_index - PHEROMONE_ID_STAGGER
                distance_to_box = graph.get_detection_distance_between_nodes(self.current_node, graph.boxes[box_index].current_node)
                max_pheromone = graph.boxes[box_index].current_pheromone_value if distance_to_box < DETECTION_RADIUS else 0
                max_node_associated_with_pheromone = graph.boxes[box_index].current_node
            else:
                pheromones_neighbors = [(n.pheromones[pheromone_index], n) for n in neighbors if
                                        pheromone_index in n.pheromones.keys()]
                max_pheromone, max_node_associated_with_pheromone = max(pheromones_neighbors, key=lambda x: x[0])
            max_pheromones_to_associated_node.append((max_pheromone, pheromone_index, max_node_associated_with_pheromone))
        return max_pheromones_to_associated_node

    def calculate_collision_detection_weights(self):
        # TODO: Implement this. Make sure that weights are proportional to value. The higher the value, the less you want
        # TODO: to get in their way. If your value is far higher, then just go for it.
        return

    def softmax_skewed_down(self, prob):
        prob = np.array(prob)
        prob = np.exp(prob) - 1   # Zero value pheromones get zero probability
        prob = prob + (.01 if np.sum(prob) == 0 else 0)
        prob = prob / np.sum(prob)  # normalize probabilities
        return prob

    def choose_target(self, graph):
        randomize = np.random.random()
        box = None if all([graph.box_has_been_moved(b) for b in self.current_node.box]) else self.current_node.box[-1]
        # TODO: For all of these target_node selections, incorporate collision detection weights
        if box:
            pheromone_options = [i for i in box.candidate_indices if graph.exists_path_box_to_hole(self.current_node, graph.nodes[i])]
            if randomize < self.explore:  # Move it to a place with less E pheromone
                candidate_indices = self.current_node.box[-1].candidate_indices
                explored_pheromone_concentrations = [graph.nodes[i].pheromones[-2] for i in pheromone_options]
                prob = np.exp(-np.array(explored_pheromone_concentrations))  # Higher E pheromone gets lower probability
                prob = prob / np.sum(prob)
                choice_index = np.random.choice(len(prob), p=prob)
                self.target_pheromone = candidate_indices[choice_index]
                self.target_node = graph.nodes[self.target_pheromone]
                print(' Random probability push to location: ', prob[choice_index], ' Explore value: ', self.explore)
            else:
                prob = [box.placement_preferences[index] for index in pheromone_options]
                prob = self.softmax_skewed_down(prob)
                choice_index = np.random.choice(len(prob), p=prob)
                self.target_pheromone = pheromone_options[choice_index]
                self.target_node = graph.nodes[self.target_pheromone]
                print(' Probability pushed to location', pheromone_options[choice_index], ': ', prob[choice_index])
        else:
            max_pheromones_to_associated_node = self.get_max_pheromones_assuming_not_near_box(graph)
            # concentration, pheromone index, nodes
            if randomize < self.explore:
                print('Random actions chosen')
                has_box_move_able = [any([not graph.box_has_been_moved(b) for b in n.box])
                                          for n in self.current_node.neighbors]
                neighbor_subset = [(n, has_box_move_able[i]) for i, n in enumerate(self.current_node.neighbors)
                                   if n in self.current_node.traversable_neighbors or has_box_move_able[i]]
                explored_pheromone_concentrations = [min([graph.nodes[c].pheromones[-2] for c in n.box[-1].candidate_indices])
                                                     if has_box else n.pheromones[-2] for n, has_box in neighbor_subset]
                prob = np.exp(-np.array(explored_pheromone_concentrations))  # Higher E pheromone gets lower probability
                prob = prob / np.sum(prob)
                choice_index = np.random.choice(len(prob), p=prob)
                self.target_pheromone = -1
                self.target_node = neighbor_subset[choice_index][0]
                if self.target_node not in self.current_node.traversable_neighbors:
                    self.target_pheromone = PHEROMONE_ID_STAGGER + graph.boxes.index(self.target_node.box[-1])
                print('  Random probability travel to location: ', prob[choice_index], ' Explore value: ', self.explore)
            else:
                prob = np.array([p[0] for p in max_pheromones_to_associated_node])
                prob = self.softmax_skewed_down(prob)
                choice_index = np.random.choice(len(prob), p=prob)
                pheromone_id = max_pheromones_to_associated_node[choice_index][1]
                self.target_pheromone = pheromone_id
                self.target_node = self.get_target_node_from_pheromone_id(pheromone_id, graph)
                print('  Pheromone chosen: ', pheromone_id, '   Probability chose pheromone: ', prob[choice_index])
        self.explore *= EXPLORE_DECAY

    def get_target_node_from_pheromone_id(self, pheromone_id, graph):
        if pheromone_id >= PHEROMONE_ID_STAGGER:  # this signals we are traveling to a box
            idx_of_the_box = pheromone_id - PHEROMONE_ID_STAGGER
            target_node = graph.boxes[idx_of_the_box].current_node
            target_node = graph.get_path_to_hole(self.current_node, target_node, from_box_to_hole=False)[1]
        elif pheromone_id >= 0:  # this signals we are taking a box to a hole
            idx_of_the_hole = pheromone_id
            target_node = graph.nodes[idx_of_the_hole]
        else:  # this signals we are traveling to somewhere random according to distance (-1) pheromone
            target_node = self.hill_climb_to_source(self.current_node, pheromone_id, max_num_moves=1)
        return target_node

    def hill_climb_to_source(self, curr_node, pheromone_id, max_num_moves=1):
        """ Given a node to travel to, returns the node corresponding to source concentration of the pheromone
            This does not characterize a path for the robot, just a target. Does not take into account feasibility of
            path given robot locomotion.
        """
        if max_num_moves == 0:  # NOTE: we pass in the next choice node already!
            return curr_node
        neighbors = curr_node.traversable_neighbors
        curr_pheromone = [n.pheromones[pheromone_id] for n in neighbors]
        prob = np.exp(curr_pheromone) - 1  # Zero value pheromone gets zero probability
        prob = prob / np.sum(prob)
        choice_node_index = np.random.choice(len(prob), p=prob)
        print(' Probability chose to travel to specific node: ', prob[choice_node_index])
        return self.hill_climb_to_source(neighbors[choice_node_index], pheromone_id, max_num_moves-1)


class Node(object):
    def __init__(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        self.box_id = coordinates[3]
        self.coords = tuple([coordinates[0], coordinates[1], coordinates[2]])
        self.pos = np.array([self.x, self.y, self.z])
        self.box = []

        self.neighbors = []  # list of Nodes (can be an intraversable hole)
        self.traversable_neighbors = []  # list of Nodes that this Node can physically reach
        self.pheromones = {}  # maps pheromone id to concentration
        self.temp_pheromones = {}
        self.is_source = self.box_id >= 0
        self.official = False
        self.distance_to_goal = INITIAL_BOX_DISTANCE_SET

    def place_box(self, box):
        self.box.append(box)
        self.z += box.height
        self.coords = tuple([self.x, self.y, self.z])
        self.pos = np.array([self.x, self.y, self.z])

    def remove_box(self):
        # Assert that when removing a box to move, there is only one box, signaling that this is definitely not
        # a candidate node. We check for more precautions in the method this is called
        assert len(self.box) == 1
        box = self.box.pop(-1)
        self.z -= box.height
        self.coords = tuple([self.x, self.y, self.z])
        self.pos = np.array([self.x, self.y, self.z])
        return box

    def handle_adding_d_pheromone(self, concentration, graph):
        if self.is_source and self.box:
            for box in self.box:
                if box.box_id == self.box_id:
                    box.placement_preferences[graph.nodes.index(self)] += concentration
                    box.placement_preferences[graph.nodes.index(self)] = min(box.placement_preferences[graph.nodes.index(self)], 50)
        # CHANGED: Change here. Do not add the concentration but assign it
        self.pheromones[-1] = concentration
