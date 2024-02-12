import search
import random
import math
import numpy as np
import copy

ids = ["314779166", "322620873"]

class State:
    """ state structure:
        pirate_locations = list(), for example: self.pirate_location_idx = list()
                                                for start_point in initial['pirate_ships'].values():
                                                    self.pirate_location_idx.append(start_point)  # Indices of the initial location of the pirate ships.
        num_treasures_held_per_pirate = list()
        treasures_locations = list(). """
    def __init__(self, pirates, treasures, num_marines):
        self.pirate_locations = list()
        for p in pirates.values():
            self.pirate_locations.append(p)

        self.num_treasures_held_per_pirate = [0] * len(self.pirate_locations)
        self.treasures_locations = treasures
        
        self.marines_locations_indices = np.zeros(shape=num_marines, dtype=int)  # Index of track list, which indicates the location of the marine. The marine starts in the first entry of the track list.
        self.marines_directions = np.ones(shape=num_marines)  # Direction of the marine w.r.t its track: 1 = next item in list, -1 = previous item in list

    def __lt__(self, other):
        return id(self) < id(other)  # TODO: Changed tie braker if necessary.

class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial_map = np.array(initial['map'])
        self.len_rows = initial_map.shape[0]
        self.len_cols = initial_map.shape[1]
        self.location_matrix = np.zeros(shape=(self.len_rows, self.len_cols, 6), dtype=int)
        # For every location in the map, we will have a list of possibilities that can be done in this location.
        # The possibilities are: base, up, down, left, right, treasure collecting.
        # This list is represented by the following indices: 0, 1, 2, 3, 4, 5.
        # In the 0 to 4 indices will be indicators that represent whether the corresponding action is possible,
        # and in the 5th index will be the number of the treasure that can be collected in this location.

        self.islands_with_treasures = list()
        for t in initial['treasures'].values():
            self.islands_with_treasures.append(t)
        self.num_of_treasures = len(initial['treasures'])
        # self.treasures_collecting_locations = np.full((self.num_of_treasures, 4, 2), -1)  # A matrix that represents the indices of the treasures in the map. The first index is the treasure number, the second index is the location in which we can collect the treasure in the map.

        def is_valid_location(location):
            x, y = location
            return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (initial_map[x][y] != 'I')
    
        for i in range(self.len_rows):
            for j in range(self.len_cols):
                if initial_map[i][j] == 'B':
                    self.base = np.array([i, j])
                    self.location_matrix[i][j][0] = 1  # base; can deposit treasure here

                if is_valid_location((i-1, j)):
                    self.location_matrix[i][j][1] = 1  # up
                if is_valid_location((i+1, j)):
                    self.location_matrix[i][j][2] = 1  # down
                if is_valid_location((i, j-1)):
                    self.location_matrix[i][j][3] = 1  # left
                if is_valid_location((i, j+1)):
                    self.location_matrix[i][j][4] = 1  # right

        # def update_treasure_collecting_locations(t_num, count, i, j):
        #     self.location_matrix[i][j][5] = t_num
        #     self.treasures_collecting_locations[t_num-1][count][0] = i
        #     self.treasures_collecting_locations[t_num-1][count][1] = j

        for treasure, location in initial['treasures'].items():
            treasure_num = int(treasure.split('_')[1])  # The number of the treasure.
            i = location[0]
            j = location[1]
            # count = 0
            for b in [-1, 1]:
                if is_valid_location((i + b, j)):
                    self.location_matrix[i + b][j][5] = self.location_matrix[i + b][j][5] * 10 + treasure_num
                    # update_treasure_collecting_locations(treasure_num, count, i + b, j)
                    # count += 1
                if is_valid_location((i, j + b)):
                    self.location_matrix[i + b][j][5] = self.location_matrix[i][j + b][5] * 10 + treasure_num
                    # update_treasure_collecting_locations(treasure_num, count, i, j + b)
                    # count += 1
        
        self.marines_tracks = initial['marine_ships'].values()
        self.num_marines = len(initial['marine_ships'])

        initial_state = State(initial['pirate_ships'], copy.deepcopy(self.islands_with_treasures), self.num_marines)
        search.Problem.__init__(self, initial_state)


    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        all_possible_actions = list()
        pirate_locations = state.pirate_locations
        
        for pirate in range(len(pirate_locations)):
            pirate_name = "pirate_ship_" + str(pirate+1)
            row_location = pirate_locations[pirate][0]
            col_location = pirate_locations[pirate][1]
            action_options = self.location_matrix[row_location][col_location]
            
            # Reminder to the indices and their meanings, according to order:
            # 0=base, 1=up, 2=down, 3=left, 4=right, 5=treasure collecting.
            if action_options[0]: all_possible_actions.append(("deposit_treasures", pirate_name))

            if action_options[1]: all_possible_actions.append(('sail', pirate_name, (row_location - 1, col_location)))
            if action_options[2]: all_possible_actions.append(('sail', pirate_name, (row_location + 1, col_location)))
            if action_options[3]: all_possible_actions.append(('sail', pirate_name, (row_location, col_location - 1)))
            if action_options[4]: all_possible_actions.append(('sail', pirate_name, (row_location, col_location + 1)))

            if action_options[5] and (state.num_treasures_held_per_pirate[pirate] < 2):
                t_num = action_options[5] % 10
                while t_num:
                    all_possible_actions.append(('collect_treasure', pirate_name, 'treasure_' + str(t_num)))
                    t_num = action_options[5] // 10

            all_possible_actions.append(('wait', pirate_name))

        return all_possible_actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = copy.deepcopy(state)
        pirate_num = int(action[1].split("_")[2])

        for i, track in enumerate(self.marines_tracks):
            if (state.marines_locations_indices[i] < len(track) - 1) and ((state.marines_directions[i] == 1) or (state.marines_locations_indices[i] == 0)):
                new_state.marines_locations_indices[i] += 1
                new_state.marines_directions[i] = 1
            elif (state.marines_locations_indices[i] > 0) and ((state.marines_directions[i] == -1) or (state.marines_locations_indices[i] == len(track) - 1)):
                new_state.marines_locations_indices[i] -= 1
                new_state.marines_directions[i] = -1

        marines_locations = [marine_track[new_state.marines_locations_indices[i]] for i, marine_track in enumerate(self.marines_tracks)]

        if action[0] == 'deposit_treasures':
            new_state.num_treasures_held_per_pirate[pirate_num - 1] = 0  # Updating the number of treasures held by the pirate ship that did this action.
            for loc_idx in range(len(new_state.treasures_locations)):
                if new_state.treasures_locations[loc_idx] == pirate_num:
                    new_state.treasures_locations[loc_idx] = 'b'
                    new_state.num_treasures_held_per_pirate[pirate_num - 1] = 0

        if action[0] == 'sail':
            new_state.pirate_locations[pirate_num - 1] = action[2]
                    
        if action[0] == 'collect_treasure' and new_state.num_treasures_held_per_pirate[pirate_num - 1] < 2:
            new_state.num_treasures_held_per_pirate[pirate_num - 1] += 1  # Updating the number of treasures held by the pirate ship that did this action.
            new_state.treasures_locations[int(action[2].split("_")[1]) - 1] = pirate_num
        
        # if action[0] == 'wait', nothing changes (except the marines - handled below).

        for m in range(len(self.marines_tracks)):
            if marines_locations[m] == new_state.pirate_locations[pirate_num - 1]:
                for loc_idx in range(len(new_state.treasures_locations)):
                    if new_state.treasures_locations[loc_idx] == pirate_num:
                        new_state.treasures_locations[loc_idx] = self.islands_with_treasures[loc_idx]
                        new_state.num_treasures_held_per_pirate[pirate_num - 1] = 0

        return new_state


    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        for t in state.treasures_locations:
            if t != 'b':
                return False
        return True

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return self.h_2(node)
    
    def h_1(self, node):
        return sum(1 for t in node.state.treasures_locations if t == 'I') / len(node.state.pirate_locations)            

    def h_2(self, node):
        num_pirates = len(node.state.pirate_locations)
        min_distances_to_base = np.full((self.num_of_treasures,), np.inf)
        for t_idx, location in enumerate(node.state.treasures_locations):
            if location == 'b':
                min_distances_to_base[t_idx] = 0
                continue
            elif type(location) == int:
                location = node.state.pirate_locations[location - 1]
            for i, direction in enumerate([(1,0), (-1,0), (0,-1), (0,1)]):
                x = location[0]+direction[0]
                y = location[1]+direction[1]
                if (self.location_matrix[location[0]][location[1]][i] == 1) and (x != self.base[0] or y != self.base[1]):
                    temp_dist = abs(self.base[0]-x) + abs(self.base[1]-y)
                    if temp_dist < min_distances_to_base[t_idx]:
                        min_distances_to_base[t_idx] = temp_dist
        return np.sum(min_distances_to_base) / num_pirates



    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""

    def calculate_distances_from_treasure(self, num_pirates):
        ''' We use BFS to calculate the distance from the base to each treasure. We normalize the distances by the number of pirates.
            This function is currently redundent, but we had no heart deleting it :( '''
        queue = [(self.base, 0)]  # A list of tuples, each tuple represents the index of a node and its distance from the base node. queue[1] is the distance that we passed so far.
        opened = []
        normalized_distances = {}  # Dictionary representing the treasure number and its distance from the base.
        while queue:
            current = queue.pop(0)
            opened.append(current[0])
            for t_num in range(len(self.num_of_treasures)):
                if ('t', t_num+1) in self.location_matrix[current[0][0]][current[0][1]]:  # Changed the location matrix since then
                    normalized_distances[t_num+1] = current[1]/num_pirates
            if len(normalized_distances.keys()) == len(self.num_of_treasures):
                return normalized_distances
            for i in [-1,1]:
                potential_child_location = (current[0][0] + i, current[0][1])
                if self.is_valid_location(potential_child_location) and (potential_child_location not in opened):
                    queue.append((potential_child_location, current[1] + 1))
                potential_child_location = (current[0][0], current[0][1] + i)
                if self.is_valid_location(potential_child_location) and (potential_child_location not in opened):
                    queue.append((potential_child_location, current[1] + 1))
        
        for t_num in range(len(self.num_of_treasures)):
            if t_num+1 not in normalized_distances.keys():
                normalized_distances[t_num+1] = float('inf')
        return normalized_distances


def create_onepiece_problem(game):
    return OnePieceProblem(game)

