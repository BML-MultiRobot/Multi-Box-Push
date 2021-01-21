from aa_graphMap_node_simulation import StigmergicGraph
import aa_graphMap_node_simulation

""" Convention: 
        -2 pheromone: exploration. More density -> more explored
        -1 pheromone: classic distance pheromone placed anywhere 
        0 -> indices < PHEROMONE_ID_STAGGER: node indices of self.nodes that correspond to candidate holes 
        PHEROMONE_ID_STAGGER >= : len(self.boxes) indices that correspond to various boxes on the map
"""


class StigmergicGraphVREP(StigmergicGraph):

    def get_robot_information(self, robot_id):
        agent = self.robots[robot_id]
        curr_node = agent.current_node
        curr_node_id = self.nodes.index(curr_node)
        return agent, curr_node, curr_node_id

    def get_box_information(self, box_pheromone_or_id):
        if box_pheromone_or_id >= aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER:
            box_id = box_pheromone_or_id - aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER
        else:
            box_id = box_pheromone_or_id
        box = self.boxes[box_id]
        curr_node = box.current_node
        curr_node_id = self.nodes.index(curr_node)
        return box_id, box, curr_node, curr_node_id

    def get_box_nodes(self, robot_id):
        """ Returns dict mapping box id to list of nodes to check able to push box to. Returns current node id robot."""
        agent, curr_node, curr_node_id = self.get_robot_information(robot_id)
        box_dict = {}
        has_box = agent.lst_of_neighbors_with_moveable_boxes(self)  # Filtered list of neighbors (including curr node)
        for n in has_box:
            box_dict[self.boxes.index(n.box[-1])] = [self.nodes.index(node) for node in n.neighbors if node != curr_node]
        return box_dict, curr_node_id

    def get_agent_target(self, robot_id, traversable_box_nodes):
        """ Determines new target node and pheromone for robot.
        Traversable_box_nodes is dict box_id to list of node ids.
        Return target node index, target box index and target pheromone

        An agent can be doing 1 of 5 things:

            1. Traveling along a path set previously
            2. Not moving at all when there are no possible targets
            3. Attempting to push a box directly to a hole
            4. Traveling towards a box
            5. Traveling according to (E) pheromone or (D) pheromone (both of which are simple)
        """
        agent, curr_node, curr_node_id = self.get_robot_information(robot_id)
        """ Option 1: """
        preset_path = self.bot_to_current_path.get(robot_id, [])
        if len(preset_path) > 0:
            can_push_box_to_next_node = self.nodes.index(preset_path[0]) in traversable_box_nodes[agent.target_pheromone]
            if can_push_box_to_next_node:
                agent_new_location_index = self.nodes.index(preset_path[0])
                box_index = agent.target_pheromone
                self.bot_to_current_path[robot_id] = preset_path
                return agent_new_location_index, box_index, agent.target_pheromone

        agent.choose_target(self, traversable_box_nodes)

        """ Option 2: """
        if not (agent.target_node or agent.target_pheromone):  # If there are no available targets
            return None, None, None

        if agent.target_pheromone >= aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER:
            box_index, box, box_node, box_node_id = self.get_box_information(agent.target_pheromone)
            """ Option 3: """
            if box_node in curr_node.neighbors:  # Pushing directly to hole
                path = self.get_path_to_hole(box_node, agent.target_node)[1:]
                agent_new_location = path[0]
                # TODO: Only decay when you first claim it? Might already handle because only goes through this once. Becomes Option 1 after claim.
                self.boxes[box_index].placement_preferences[self.nodes.index(agent.target_node)] *= aa_graphMap_node_simulation.B_preference_decay
                print('New placement preference: ', self.boxes[box_index].placement_preferences[self.nodes.index(agent.target_node)])
            else:
                """ Option 4: """

                path = self.get_path_to_hole(curr_node, box_node, from_box_to_hole=False)[1:2]
                agent_new_location = path[0]
                box_index = -1
        else:
            """ Option 5: """
            assert agent.target_pheromone < 0
            box_index = -1
            path = [agent.target_node]
            agent_new_location = agent.target_node

        assert len(path) > 0
        self.bot_to_current_path[robot_id] = path
        agent_new_location_index = self.nodes.index(agent_new_location)
        return agent_new_location_index, box_index, agent.target_pheromone

    def update_agent_location(self, robot_id, new_node_id):
        """ Updates agent location on the map. Used for infrastructure purposes. Allows agent to update pheromones."""
        agent, curr_node, curr_node_id = self.get_robot_information(robot_id)
        new_node = self.nodes[new_node_id]
        traveled_path = self.bot_to_current_path[robot_id] if robot_id in self.bot_to_current_path.keys() else [new_node]
        if len(traveled_path) == 0:  # didn't go anywhere
            return
        traveled_path.pop(0)

        """ Handles updating moving a box on the map """
        if agent.target_pheromone >= aa_graphMap_node_simulation.PHEROMONE_ID_STAGGER:
            box_index, box, box_node, box_node_id = self.get_box_information(agent.target_pheromone)
            if (box_node in curr_node.neighbors) or (box_node == agent.current_node):  # if we moved it
                box = box_node.remove_box()
                new_node.place_box(box)
                assert not self.box_has_been_moved(box)
                self.update_neighbors_to_reflect_box_change(curr_node)
                self.update_neighbors_to_reflect_box_change(box_node)
                self.update_neighbors_to_reflect_box_change(new_node)
                box_prev_node = box.current_node
                box.current_node = new_node
                if len(self.bot_to_current_path[robot_id]) == 0:
                    self.placed_box_indices.add(box_index)
                    self.handle_reaching_goal(robot_id)
                    new_node = box_prev_node

        """ Handle adding pheromones"""
        path = [curr_node, new_node]
        self.add_d_pheromones_to_path(path)
        self.add_e_pheromones_to_path(path)
        self.update_pheromones(path)
        agent.current_node = new_node

        """ Handle reaching goal node/location """
        self.handle_reaching_goal(robot_id)
        return

    def handle_reaching_goal(self, robot_id):
        agent, _, _ = self.get_robot_information(robot_id)
        if self.nodes.index(agent.current_node) == self.nodes.index(self.goal):
            agent.current_node = None
            self.finished_robots.add(agent)
        return
