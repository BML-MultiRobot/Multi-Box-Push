import numpy as np
from Tasks.task import distance as dist
from copy import deepcopy
from aa_graph_classes import Box, StigmergicAgent, StigmergicAgentVREP, Node
import matplotlib.pyplot as plt
import networkx as nx
import heapq

# Logistical parameters
PHEROMONE_ID_STAGGER = 100  # just for distinguishing pheromones
INITIAL_BOX_DISTANCE_SET = .1  # just for presetting node distances (set low to encourage initial exploration)
RADIUS = 2  # 1.9 for node simulation...radius for node creation (not actually for algorithm)
SPEED = 1  # time in seconds per step for visualization
MAX_DISTANCE_SET = 20  # 20 for large # 10 for medium # 6 for small environments
USING_HOLE_PHEROMONES = True

# Training hyper-parameters
EPISODES = 100
MAX_STEPS = 20

# Performance hyper-parameters
DETECTION_RADIUS = 10 # 3 for simulation...10 for V-REP
EXPLORE_DECAY = .95
START_EXPLORE = .7


B_preference_decay = .8  # Decays every time we attempt to place a box
E_spatial_decay = 2  # Exploration spatial spreading when agent lands
E_temporal_decay = .7  # Temporal decay when agent lands

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
        self.start_state.convert_to_nodes(nodes, inclusions, exclusions, box_data, robot_data, goal_index) if goal_index else None
        self.current_environment = None
        self.goal_index = goal_index

    def main_algorithm(self):
        self.current_environment = deepcopy(self.start_state)
        indicator_success_each_episode = []
        display_graph = nx.grid_2d_graph(2, 2)
        self.update_display_graph_using_current_environment(display_graph, 1)
        plt.ion()
        plt.show()
        for i in range(EPISODES):
            print('')
            print(' ##### EPISODE: ', i + 1)

            achieved_goal = False
            restart = False
            step = 1
            while not achieved_goal and not restart and step <= MAX_STEPS:
                print('')
                print(' Step number: ', step)

                self.update_display_graph_using_current_environment(display_graph, i + 1)
                plt.pause(SPEED)

                achieved_goal, restart = self.current_environment.one_step()
                step += 1
            print('Achieved goal: ', achieved_goal)

            self.update_display_graph_using_current_environment(display_graph, i + 1)
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

    def update_display_graph_using_current_environment(self, networkx_graph, episode):
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
        plt.subplot(221)
        plt.title('Episode ' + str(episode) + ' Distance Pheromone. Goal: ' + str(self.goal_index))
        nx.draw(networkx_graph, node_color=colors_for_pheromones, pos=coordinates, node_size=500, with_labels=True,
                cmap=plt.cm.Greens)

        # Draw the current location of the agent
        plt.subplot(222)
        plt.title('Episode ' + str(episode) + ' Agent Locations')
        index_current_agents = [self.current_environment.nodes.index(agent.current_node) for agent in
                                self.current_environment.robots if agent.current_node != None]
        colors_of_agents = [50 if index in index_current_agents else 0 for index in
                            range(len(self.current_environment.nodes))]
        nx.draw(networkx_graph, node_color=colors_of_agents, pos=coordinates, node_size=500, with_labels=True,
                cmap=plt.cm.Blues)

        # Draw indications of where the boxes currently are
        plt.subplot(223)
        plt.title('Episode ' + str(episode) + ' Box Location')
        color_box_locations = [50 if node.box else 0 for node in self.current_environment.nodes]
        nx.draw(networkx_graph, node_color=color_box_locations, pos=coordinates, node_size=500, with_labels=True,
                cmap=plt.cm.Reds)

        # Draw indications of where the holes are
        plt.subplot(224)
        plt.title('Episode ' + str(episode) + ' Hole Locations')
        color_hole_locations = [50 if node.is_source else 0 for node in self.current_environment.nodes]
        nx.draw(networkx_graph, node_color=color_hole_locations, pos=coordinates, node_size=500, with_labels=True,
                cmap=plt.cm.Reds)

        plt.draw()
        plt.pause(SPEED)

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
        self.finished_robots = set()
        self.placed_box_indices = set()
        self.goal = None

        self.prev_paths = {}
        self.bot_to_current_path = {}

    ################################## STIGMERGIC NODE SIMULATION BEGIN ##############################################
    def one_step(self):
        # After initialization, in order of decreasing value of agent (or something similar) decide who is moving
        self.boxes_calculate_values()
        agent_values = []
        unfinished_robots = [agent for agent in self.robots if agent not in self.finished_robots]
        for agent in unfinished_robots:
            # list of tuples (amount of pheromone, node neighbor) in order of pheromone index
            agent_value = agent.get_value_of_agent(self)
            agent.value = agent_value
            agent_values.append((agent_value, agent))
        if all([v < 0 for v, agent in agent_values]):
            return False, True

        agents_with_changed_targets = [agent[1] for agent in agent_values if agent[0] > 0]
        for agent in agents_with_changed_targets:
            agent.choose_target(self)

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
            else:
                path = [agent.current_node, agent.target_node]
                last_node_in_path = agent.target_node
                agent_new_location = last_node_in_path
            self.add_d_pheromones_to_path(path)
            self.add_e_pheromones_to_path(path)
            all_touched_nodes.extend(path)
            agent.current_node = agent_new_location
        self.update_pheromones(all_touched_nodes)

        for agent in self.robots:
            if agent.current_node == self.goal:
                agent.current_node = None
                self.finished_robots.add(agent)
        if len(self.finished_robots) == len(self.robots):
            return True, True
        return False, False

    def update_neighbors_to_reflect_box_change(self, current_node):
        all_neighbors = current_node.neighbors
        all_neighbors = all_neighbors
        for node in all_neighbors:
            neighbors = self.get_neighbors(node, RADIUS, include_intraversable=True)
            traversable_neighbors = self.get_neighbors(node, RADIUS, include_intraversable=False)
            node.neighbors = neighbors
            node.traversable_neighbors = traversable_neighbors
        # NOTE: Current node gets special treatment. If we want to move to a different node given at a node with box,
        # need to know if the other places can be traveled to
        current_node.neighbors = self.get_neighbors(current_node, RADIUS, include_intraversable=True)
        if len(current_node.box) > 0 and not self.box_has_been_moved(current_node.box[-1]):
            box = current_node.box[-1]
            current_node.traversable_neighbors = [n for n in current_node.neighbors if abs(n.z - (current_node.z - box.height)) < .1]
        else:
            current_node.traversable_neighbors = self.get_neighbors(current_node, RADIUS, include_intraversable=False)
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
                    closest_to_goal = max(official_neighbors, key=lambda
                        neighbor: neighbor.distance_to_goal + self.get_distance_between_nodes(node, neighbor))
                    official_distance = closest_to_goal.distance_to_goal + self.get_distance_between_nodes(node,
                                                                                                           closest_to_goal)
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

    def exists_path_box_to_hole(self, start, finish):
        path = self.get_path_to_hole(start, finish, from_box_to_hole=True)
        return path[-1] == finish

    def exists_path_to_box(self, start, finish):
        path = self.get_path_to_hole(start, finish, from_box_to_hole=False)
        return path[-1] == finish, path

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

            neighbors = curr_node.neighbors if (
                        from_box_to_hole and curr_node == start) else curr_node.traversable_neighbors

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
            if USING_HOLE_PHEROMONES:
                subset_of_candidates_with_path = [value for index, value in box.placement_preferences.items()
                                                  if self.check_if_box_has_hole_pheromone_indicator(box.current_node,
                                                                                                    index)]
            else:
                subset_of_candidates_with_path = [value for index, value in box.placement_preferences.items()
                                                  if self.exists_path_box_to_hole(box.current_node, self.nodes[index])]
            box_value = max(subset_of_candidates_with_path) if len(subset_of_candidates_with_path) > 0 else 0
            box.current_pheromone_value = box_value if not box.claimed else 0

    def check_if_box_has_hole_pheromone_indicator(self, current_node, candidate_index):
        neighbors = current_node.neighbors
        has_path = any([n.pheromones[candidate_index] > 0 for n in neighbors])
        return has_path

    def update_pheromones(self, all_touched_nodes):
        self.boxes_calculate_values()
        self.drop_hole_pheromones(all_touched_nodes)
        self.update_hole_pheromones(all_touched_nodes)
        self.spread_E_pheromones_along_path(all_touched_nodes)
        self.decay(all_touched_nodes)
        self.add_temp_pheromones()

    def drop_hole_pheromones(self, path):
        for node in path:
            hole_nodes = [self.nodes.index(n) for n in node.neighbors if n.box_id >= 0]
            pheromones = node.pheromones
            for hole in hole_nodes:
                pheromones[hole] = min(pheromones[hole], self.get_distance_between_nodes(self.nodes[hole], node))
        return

    def update_hole_pheromones(self, path):
        hole_pheromones = {p for p in self.pheromone_set if 0 <= p < PHEROMONE_ID_STAGGER}
        for node in path:
            neighbors = node.traversable_neighbors
            if len(neighbors) > 0:
                for p in hole_pheromones:
                    max_p = max([n.pheromones[p] * np.exp(-E_spatial_decay * dist(node.pos, n.pos)) for n in neighbors])
                    if node.pheromones[p] < max_p:
                        node.pheromones[p] = max_p
        return

    def add_e_pheromones_to_path(self, path):
        for node in path:  # Update E pheromone but include the first node as well
            node.pheromones[-2] += 1

    def spread_E_pheromones_along_path(self, path):
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

    def post_episode_pheromone_update(self):
        # Get rid of all non-negative pheromones that are not from sources. Keep D and E pheromones as well.
        for pheromone_id, n in enumerate(self.nodes):
            new_pheromones = {p: 0 for p in self.pheromone_set}
            new_pheromones[-1] = n.pheromones[-1]
            new_pheromones[-2] = n.pheromones[-2]
            n.pheromones = new_pheromones
        return

    """ ############################## Insertion and Initialization of the Graph ################################ """

    def convert_to_nodes(self, nodes, inclusions, exclusions, box_data, robot_data, goal_index, vrep_simulation=False):
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
            pheromone_set.update(candidate_node_indices)
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
            agent = StigmergicAgent(self.nodes[robot_curr_node], robot_id) if not vrep_simulation else StigmergicAgentVREP(self.nodes[robot_curr_node], robot_id)
            self.robots.append(agent)

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
                    (include_intraversable or abs(node.z - node_other.z) < .3):
                neighbors.append(node_other)
            elif (curr_index, other_index) in self.inclusions or (other_index, curr_index) in self.inclusions:
                neighbors.append(node_other)
        return neighbors
