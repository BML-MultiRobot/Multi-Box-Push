import aa_graphMap_node_simulation
import numpy as np

class Box(object):
    def __init__(self, height, candidate_nodes, box_id, candidate_indices, current_node):
        # Height is height of box
        # candidate_nodes is list of nodes representing possible places it can be placed
        self.box_id = box_id
        self.height = height
        self.current_node = current_node
        self.candidate_indices = candidate_indices   # Hole indices this box can be placed in
        self.placement_preferences = {c: 2 * aa_graphMap_node_simulation.MAX_DISTANCE_SET for c in self.candidate_indices}
        self.current_pheromone_value = 2 * aa_graphMap_node_simulation.MAX_DISTANCE_SET
        self.claimed = False

class StigmergicAgent(object):
    def __init__(self, current_node, robot_id):
        self.value = 0
        self.robot_id = robot_id
        self.current_node = current_node
        self.target_pheromone = None
        self.target_node = None
        self.explore = aa_graphMap_node_simulation.START_EXPLORE

    def get_value_of_agent(self, graph):
        box = self.get_box(graph)
        if box:
            return max(box.placement_preferences.values())
        else:
            pheromones, _ = self.get_max_pheromones_assuming_not_near_box(graph)
            return max(map(lambda x: x[0], pheromones)) if len(pheromones) > 0 else 0

    def get_max_pheromones_assuming_not_near_box(self, graph):
        """
        Assumes that they agent is not at a moveable box
        Returns list of tuples (max_pheromone_concentration, pheromone_index, max_node_associated_with_pheromone)
        """
        possible_pheromones = [aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER + i for i in range(len(graph.boxes)) if i not in graph.placed_box_indices and not graph.boxes[i].claimed]
        if len(self.current_node.traversable_neighbors) > 0:
            possible_pheromones.append(-1)

        traversable_neighbors = self.current_node.traversable_neighbors
        max_pheromones_to_associated_node = []
        collision_weights = self.calculate_collision_detection_weights(graph)
        for pheromone_index in possible_pheromones:
            if pheromone_index >= aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER:  # if there is a box nearby
                box_index = pheromone_index - aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER
                distance_to_box = graph.get_detection_distance_between_nodes(self.current_node, graph.boxes[box_index].current_node)
                exists_path_to_box, path = graph.exists_path_to_box(self.current_node, graph.boxes[box_index].current_node)
                node_where_box_is = graph.boxes[box_index].current_node
                if len(path) <= 1:  # This means that the agent is right next to the box :(
                    continue
                if distance_to_box <= aa_graphMap_node_simulation.DETECTION_RADIUS and exists_path_to_box:
                    node_with_hole_to_values = graph.boxes[box_index].placement_preferences
                    vals = {val for node_index, val in node_with_hole_to_values.items() if self.can_push_directly_from_box_to_hole(graph, node_where_box_is, graph.nodes[node_index])}
                    value = max(graph.boxes[box_index].current_pheromone_value, max(vals) if len(vals) > 0 else -np.inf)
                    max_pheromone = value * collision_weights[path[1]]
                    collision_weights[node_where_box_is] = collision_weights[path[1]]
                    max_node_associated_with_pheromone = node_where_box_is
                else:
                    max_pheromone = 0
                    max_node_associated_with_pheromone = None
            elif pheromone_index == -1:
                pheromones_neighbors = [(n.pheromones[pheromone_index] * collision_weights[n], n) for n in traversable_neighbors if
                                        pheromone_index in n.pheromones.keys()]
                max_pheromone, max_node_associated_with_pheromone = max(pheromones_neighbors, key=lambda x: x[0])
            else:
                assert False
            if max_pheromone > 0:
                assert max_node_associated_with_pheromone  # make sure not None
                max_pheromones_to_associated_node.append((max_pheromone, pheromone_index, max_node_associated_with_pheromone))
        return max_pheromones_to_associated_node, collision_weights

    def calculate_collision_detection_weights(self, graph):
        # Calculates collision indicators for each neighboring node. Independent to traversability.
        neighbors, neighbor_to_weight = self.current_node.neighbors, {}
        for n in neighbors:
            if not self.robot_in_node(n, graph):
                second_neighbors = n.neighbors[:]  # make a copy because mutable
                second_neighbors.remove(self.current_node)
                num_not_filled = len([s_n for s_n in second_neighbors if not any([bot.current_node == s_n for bot in graph.robots])])
                neighbor_to_weight[n] = 1 if num_not_filled == len(second_neighbors) else 0
            else:
                neighbor_to_weight[n] = 0
        return neighbor_to_weight

    def robot_in_node(self, node, graph):
        return any([bot.current_node == node for bot in graph.robots])

    def softmax_skewed_down(self, prob):
        prob = (np.exp(np.array(prob)) - 1) + .01
        return prob / np.sum(prob)  # normalize probabilities

    def get_box(self, graph):  # No need take into account claimed because agents cannot be in same node anyway.
        return None if all([graph.box_has_been_moved(b) for b in self.current_node.box]) else self.current_node.box[-1]

    def choose_target(self, graph, reachable_box_nodes=None):
        randomize = np.random.random()
        box = self.get_box(graph)
        if box:
            box.claimed = True
            pheromone_options = [i for i in box.candidate_indices if graph.check_if_box_has_hole_pheromone_indicator(box, i) or self.can_push_directly_from_box_to_hole(graph, box.current_node, graph.nodes[i])]
            if randomize < self.explore:  # Move to random candidate index
                # candidate_indices = self.current_node.box[-1].candidate_indices
                num_options = len(pheromone_options)
                choice_index = np.random.randint(num_options)
                self.target_pheromone = pheromone_options[choice_index]
                self.target_node = graph.nodes[self.target_pheromone]
                # print(' ### EXPLORE ### Uniform probability chosen travel to hole', ' Explore value: ', self.explore)
            else:
                prob = [box.placement_preferences[index] for index in pheromone_options]
                # print("### OPTIONS", pheromone_options, prob)
                prob = self.softmax_skewed_down(prob)
                choice_index = np.random.choice(len(prob), p=prob)
                self.target_pheromone = pheromone_options[choice_index]
                self.target_node = graph.nodes[self.target_pheromone]
                # print(' Probability pushed to location', pheromone_options[choice_index], ': ', prob[choice_index])
            box.placement_preferences[graph.nodes.index(self.target_node)] *= aa_graphMap_node_simulation.B_preference_decay
        else:
            max_pheromones_to_associated_node, collision_weights = self.get_max_pheromones_assuming_not_near_box(graph)
            assert self.value > 0
            assert len(max_pheromones_to_associated_node) > 0
            # concentration, pheromone index, nodes
            if randomize < self.explore:
                # print('Random actions chosen')
                # Get boolean array stating if neighbors have move-able box
                has_box_move_able = self.lst_of_neighbors_with_moveable_boxes(graph)
                # Get neighbor subset that can be traveled to or has moveable box
                neighbor_subset = [n for n in self.current_node.neighbors
                                   if n in self.current_node.traversable_neighbors or (n in has_box_move_able)]
                # Get explore pheromone concentrations of each neighbor
                explored_pheromone_concentrations = [n.pheromones[-2] for n in neighbor_subset]
                mask_for_collisions = np.array([collision_weights[n] for n in neighbor_subset])
                prob = np.exp(aa_graphMap_node_simulation.Boltzmann * -np.array(explored_pheromone_concentrations))  # Higher E pheromone gets lower probability
                prob = prob * mask_for_collisions
                prob = prob / np.sum(prob)
                choice_index = np.random.choice(len(prob), p=prob)
                self.target_node = neighbor_subset[choice_index]
                if False: # self.target_node not in self.current_node.traversable_neighbors:
                    self.target_pheromone = aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER + graph.boxes.index(self.target_node.box[-1])
                else:
                    self.target_pheromone = -2
                # print(' ### EXPLORE ### Random probability traveled to location', graph.nodes.index(self.target_node), ': ', prob[choice_index])
            else:
                prob = np.array([p[0] * collision_weights[p[2]] for p in max_pheromones_to_associated_node])
                prob = self.softmax_skewed_down(prob)
                assert np.all(prob > 0)
                assert len(prob) > 0
                choice_index = np.random.choice(len(prob), p=prob)
                pheromone_id = max_pheromones_to_associated_node[choice_index][1]
                self.target_pheromone = pheromone_id
                self.target_node = self.get_target_node_from_pheromone_id(pheromone_id, graph, collision_weights)
                # if pheromone_id != -1:
                #    print(' Probability traveled to location', graph.nodes.index(self.target_node), ': ', prob[choice_index])
        if randomize < self.explore:
            self.explore *= aa_graphMap_node_simulation.EXPLORE_DECAY
            self.explore = max(aa_graphMap_node_simulation.MIN_EXPLORE, self.explore)
            print('### EXPLORE probability: ', self.explore)
        if aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER > self.target_pheromone >= 0:
            assert len(self.current_node.box) >= 1

    def lst_of_neighbors_with_moveable_boxes(self, graph):
        neighbors_plus_current_node = self.current_node.neighbors + [self.current_node]
        return [n for n in neighbors_plus_current_node if any([not graph.box_has_been_moved(b) for b in n.box])]

    def get_target_node_from_pheromone_id(self, pheromone_id, graph, collision_weights):
        if pheromone_id >= aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER:  # this signals we are traveling to a box
            idx_of_the_box = pheromone_id - aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER
            target_node = graph.boxes[idx_of_the_box].current_node
            target_node = graph.get_path_to_hole(self.current_node, target_node, from_box_to_hole=False)[1]
        elif pheromone_id >= 0:  # this signals we are taking a box to a hole
            idx_of_the_hole = pheromone_id
            target_node = graph.nodes[idx_of_the_hole]
        else:  # this signals we are traveling to somewhere random according to distance (-1) pheromone
            target_node = self.hill_climb_to_source(graph, self.current_node, pheromone_id, collision_weights)
        return target_node

    def hill_climb_to_source(self, graph, curr_node, pheromone_id, collision_weights):
        """ Given a node to travel to, returns the node corresponding to source concentration of the pheromone
            This does not characterize a path for the robot, just a target. Does not take into account feasibility of
            path given robot locomotion.
        """
        neighbors = curr_node.traversable_neighbors
        curr_pheromone = np.array([n.pheromones[pheromone_id] * collision_weights[n] for n in neighbors])
        prob = np.exp(aa_graphMap_node_simulation.Boltzmann * curr_pheromone) - 1  # Zero value pheromone gets zero probability
        prob = prob / np.sum(prob)
        choice_node_index = np.random.choice(len(prob), p=prob)
        # print(' Probability traveled to location', graph.nodes.index(neighbors[choice_node_index]), ': ', prob[choice_node_index])
        # print(' Probability chose to travel to specific node: ', prob[choice_node_index])
        return neighbors[choice_node_index]

    def can_push_directly_from_box_to_hole(self, graph, node_with_box, node_with_hole):
        has_path = graph.exists_path_box_to_hole(node_with_box, node_with_hole)
        return graph.get_distance_between_nodes(node_with_box, node_with_hole) <= aa_graphMap_node_simulation.DETECTION_RADIUS and has_path


class StigmergicAgentVREP(StigmergicAgent):

    def choose_target(self, graph, reachable_box_nodes=None, unallowed_nodes=[]):
        """ Choose new target for the agent given the graph's local information.

        reachable_box_nodes: dict box_id to list of node ids box can be pushed to given agent's curr location.

        Method is called in VREP sim assuming the agent requires a brand new target (not already pushing box to hole)"""
        randomize = np.random.random()
        node_with_box_to_hole_index_options = self.map_node_with_box_to_candidate_hole_nodes(graph, reachable_box_nodes)
        max_pheromones_to_associated_node, collision_weights = self.get_max_pheromones_assuming_not_near_box(graph)

        if randomize < self.explore: # Choose a hole target for box in neighbor for each of the boxes. Decide at uniform
            traversable_neighbors = self.current_node.traversable_neighbors
            candidate_nodes = traversable_neighbors + list([k for k, v in node_with_box_to_hole_index_options.items() if len(v) > 0])
            candidate_nodes = [n for n in candidate_nodes if n in collision_weights.keys() and collision_weights[n] > 0]  # TODO: This might cause bugs!

            explored_pheromone_concentrations = [n.pheromones[-2] for n in candidate_nodes]
            prob = np.exp(-np.array(explored_pheromone_concentrations))  # Higher E pheromone gets lower probability
            if prob.size == 0:
                self.target_pheromone, self.target_node = None, None
                return
            prob = prob / np.sum(prob)
            random_choice = candidate_nodes[np.random.choice(len(prob), p=prob)]
            if len(node_with_box_to_hole_index_options.keys()) > 0:  # NEW RULE: If next to a box, you have to push it.
                random_choice = np.random.choice(list(node_with_box_to_hole_index_options.keys()))

            if random_choice in node_with_box_to_hole_index_options.keys():  # Chose the box
                self.target_pheromone = graph.boxes.index(random_choice.box[-1]) + aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER
                self.target_node = graph.nodes[np.random.choice(node_with_box_to_hole_index_options[random_choice])]
            else:
                self.target_pheromone = -2
                self.target_node = random_choice
            self.explore *= aa_graphMap_node_simulation.EXPLORE_DECAY
            self.explore = max(aa_graphMap_node_simulation.MIN_EXPLORE, self.explore)
        else:
            box_pheromone_to_node_associated_to_hole = {}
            max_pheromones_to_associated_node = [tup for tup in max_pheromones_to_associated_node if
                                                 tup[1] < aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER]
            if len(node_with_box_to_hole_index_options) > 0:
                closest_node_with_box = min(list(node_with_box_to_hole_index_options.keys()),
                                            key=lambda node: graph.get_detection_distance_between_nodes(node, self.current_node))
                box = closest_node_with_box.box[-1]
                box_index_pheromone = graph.boxes.index(box) + aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER
                old_box_pheromone = next((x for x in max_pheromones_to_associated_node if x[1] == box_index_pheromone), None)
                max_pheromones_to_associated_node.remove(old_box_pheromone) if old_box_pheromone else None
                lst_of_hole_indices = node_with_box_to_hole_index_options[closest_node_with_box]
                if len(lst_of_hole_indices) > 0:  # If this box has options, modify the pheromone decision
                    hole_preferences = [box.placement_preferences[i] for i in lst_of_hole_indices]
                    p = self.softmax_skewed_down(hole_preferences)
                    hole_index_choice = lst_of_hole_indices[np.random.choice(len(p), p=p)]
                    box_pheromone_to_node_associated_to_hole[box_index_pheromone] = graph.nodes[hole_index_choice]
                    new_pheromone_concentration = box.placement_preferences[hole_index_choice]  # TODO: This was changed!
                    max_pheromones_to_associated_node.append((new_pheromone_concentration, box_index_pheromone, closest_node_with_box))
            if len(max_pheromones_to_associated_node) == 0:
                self.target_node = None
                self.target_pheromone = None
                return

            prob = np.array([p[0] * collision_weights[p[2]] if p[2] in collision_weights.keys() else p[0] for p in max_pheromones_to_associated_node])  # TODO: This might cause bugs
            # print(self.robot_id, prob)
            prob = self.softmax_skewed_down(prob)

            assert np.all(prob > 0)
            assert len(prob) > 0
            choice_index = np.random.choice(len(prob), p=prob)
            _, pheromone_id, _ = max_pheromones_to_associated_node[choice_index]
            self.target_pheromone = pheromone_id
            if pheromone_id in list(map(lambda x: x + aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER, reachable_box_nodes.keys())):
                node_associated_to_hole = box_pheromone_to_node_associated_to_hole[pheromone_id]
                assert node_associated_to_hole
                self.target_node = node_associated_to_hole
            else:
                self.target_node = self.get_target_node_from_pheromone_id(pheromone_id, graph, collision_weights)

    def map_node_with_box_to_candidate_hole_nodes(self, graph, reachable_box_nodes):
        """ Given each neighboring box, determine the holes it can be pushed to using sensor and pheromone information.
        Then, see if the first target in the determined path is achievable by comparing with reachable_box_nodes.
        Returns dictionary mapping instance of node with box to list of node indices associated with candidate holes.

        reachable_box_nodes: dict box_id to list of node ids box can be pushed to given agent's curr location.  """
        has_box_move_able = self.lst_of_neighbors_with_moveable_boxes(graph)
        box_instance_to_hole_index_options = {}
        # For every nearby node with a box on it, figure out
        for node_with_box in has_box_move_able:
            box = node_with_box.box[-1]
            box_index = graph.boxes.index(box)
            hole_options = [i for i in box.candidate_indices if graph.check_if_box_has_hole_pheromone_indicator(box, i) or
                            self.can_push_directly_from_box_to_hole(graph, box.current_node, graph.nodes[i])]

            nodes_to_push_box_to = []
            for node_index_associated_to_hole in hole_options:
                path = graph.get_path_to_hole(start=box.current_node, finish=graph.nodes[node_index_associated_to_hole])
                if len(path) < 2:
                    print(node_index_associated_to_hole)
                    print(yes)
                next_node_index = graph.nodes.index(path[1])
                achievable = next_node_index in reachable_box_nodes[box_index]
                if achievable:
                    nodes_to_push_box_to.append(node_index_associated_to_hole)
            # nodes_to_push_box_to contains node indices of possible possible achievable places we can push the box to
            if len(nodes_to_push_box_to) > 0:
                box_instance_to_hole_index_options[node_with_box] = nodes_to_push_box_to
        return box_instance_to_hole_index_options

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
        self.distance_to_goal = aa_graphMap_node_simulation.INITIAL_BOX_DISTANCE_SET

    def place_box(self, box):
        self.box.append(box)
        self.z += box.height
        self.coords = tuple([self.x, self.y, self.z])
        self.pos = np.array([self.x, self.y, self.z])

    def remove_box(self):
        # Assert that when removing a box to move, there is only one box, signaling that this is definitely not
        # a candidate node. We check for more precautions in the method this is called
        assert len(self.box) >= 1
        box = self.box.pop(-1)
        self.z -= box.height
        self.coords = tuple([self.x, self.y, self.z])
        self.pos = np.array([self.x, self.y, self.z])
        return box

    def handle_adding_d_pheromone(self, concentration, graph):
        if self.is_source and self.box:
            for box in self.box:
                if box.box_id == self.box_id and graph.box_has_been_moved(box):
                    box.placement_preferences[graph.nodes.index(self)] *= (1.0 / aa_graphMap_node_simulation.B_preference_decay)
                    box.placement_preferences[graph.nodes.index(self)] = min(box.placement_preferences[graph.nodes.index(self)], 10 * aa_graphMap_node_simulation.MAX_DISTANCE_SET)
                    print('New placement preference', graph.boxes.index(box), box.placement_preferences[graph.nodes.index(self)])
        self.pheromones[-1] = concentration
