import search
import random
import math

ids = ["314779166", "322620873"]

class State:
    """ state structure:
        pirate_locations = list(), for example: self.pirate_location_idx = list()
                                                for start_point in initial['pirate_ships'].values():
                                                    self.pirate_location_idx.append(start_point)  # Indices of the initial location of the pirate ships.
        num_treasures_held_per_pirate = list()
        treasures_locations = list(). """
    def __init__(self, pirates, treasures, marine_names):
        self.pirate_locations = {p: loc for p, loc in pirates.items()} # dict of pirates and their locations in the map
        self.treasures_locations = {t: loc for t, loc in treasures.items()} # dict of a treasure name with its location on the map

        # dict of marines and a tuple of location index in track and direction for each (1:going forward in track, -1:going backward in track)
        if marine_names:
            self.marines_position = {m: (0, 1) for m in marine_names}

        # dict of pirates and the number of treasure each one holds
        self.num_treasures_held_per_pirate = {p: 0 for p in pirates}
        # for p in pirates:
        #     self.pirate_locations.append(p)
        # self.marines_locations_indices = [0] * len(num_marines) # Index of track list, which indicates the location of the marine. The marine starts in the first entry of the track list.
        # self.marines_directions = [1] * len(num_marines)  # Direction of the marine w.r.t its track: 1 = next item in list, -1 = previous item in list


    def clone_state(self):
        new_state = State(self.pirate_locations, self.treasures_locations, 0)
        new_state.marines_position = {m: loc_direct for m, loc_direct in self.marines_position.items()}
        new_state.num_treasures_held_per_pirate = {p: num_t for p, num_t in self.num_treasures_held_per_pirate.items()}

        # new_state.pirate_locations = {p: 0 for p in pirates} self.pirate_locations
        # new_state.num_treasures_held_per_pirate = [p for p in self.num_treasures_held_per_pirate]
        # for m in marine_names:
        #     new_state.marines_position[m] = (0,
        #                                 1)  # tuple of location index in track and direction (1:going forward in track, -1:going backward in track)
        #
        # for m in range(len(self.marines_locations_indices)):
        #     new_state.marines_locations_indices[m] = self.marines_locations_indices[m]
        #     new_state.marines_directions[m] = self.marines_directions[m]
        return new_state


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
        self.num_of_treasures = len(initial['treasures'])
        self.num_digits = len(str(self.num_of_treasures))

        self.location_matrix = np.zeros(shape=(self.len_rows, self.len_cols, 5 + self.num_digits), dtype=int)
        # For every location in the map, we will have a list of possibilities that can be done in this location.
        # The possibilities are: base, up, down, left, right, treasure collecting.
        # This list is represented by the following indices: 0, 1, 2, 3, 4, 5.
        # In the 0 to 4 indices will be indicators that represent whether the corresponding action is possible,
        # and in the 5th index will be the number of the treasure that can be collected in this location.

        self.islands_with_treasures = list()
        for t in initial['treasures'].values():
            self.islands_with_treasures.append(t)
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
            treasure_num_digits = len(str(treasure_num))
            i = location[0]
            j = location[1]
            # count = 0
            for b in [-1, 1]:
                if is_valid_location((i + b, j)):
                    self.location_matrix[i + b][j][4 + treasure_num_digits] = self.location_matrix[i + b][j][4 + treasure_num_digits] * (10 ** treasure_num_digits) + treasure_num
                    # update_treasure_collecting_locations(treasure_num, count, i + b, j)
                    # count += 1
                if is_valid_location((i, j + b)):
                    self.location_matrix[i][j + b][4 + treasure_num_digits] = self.location_matrix[i][j + b][4 + treasure_num_digits] * (10 ** treasure_num_digits) + treasure_num
                    # update_treasure_collecting_locations(treasure_num, count, i, j + b)
                    # count += 1
        
        self.marines_tracks = initial['marine_ships'].values()
        self.num_marines = len(initial['marine_ships'])

        initial_state = State(initial['pirate_ships'].values(), self.islands_with_treasures, self.num_marines)
        search.Problem.__init__(self, initial_state)


    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        all_possible_actions = list()
        pirate_locations = state.pirate_locations
        
        for pirate_name in pirate_locations.keys():
            row_location = pirate_locations[pirate_name][0]
            col_location = pirate_locations[pirate_name][1]
            action_options = self.location_dict[(row_location,col_location)]
            
            # Reminder to the indices and their meanings, according to order:
            # b=base, u=up, d=down, l=left, r=right, t=treasure collecting.
            if action_options['b']: all_possible_actions.append(("deposit_treasures", pirate_name))

            if action_options['u']: all_possible_actions.append(('sail', pirate_name, (row_location - 1, col_location)))
            if action_options['d']: all_possible_actions.append(('sail', pirate_name, (row_location + 1, col_location)))
            if action_options['l']: all_possible_actions.append(('sail', pirate_name, (row_location, col_location - 1)))
            if action_options['r']: all_possible_actions.append(('sail', pirate_name, (row_location, col_location + 1)))
            if len(action_options['t']) and (state.num_treasures_held_per_pirate[pirate_name] < 2):
                for t in action_options['t']:
                    all_possible_actions.append(('collect_treasure', pirate_name, t))
            all_possible_actions.append(('wait', pirate_name))
        return all_possible_actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = state.clone_state()
        pirate_name = action[1]
        # pirate_num = int(action[1].split("_")[2])

        # dict of marines and a tuple of location index in track and direction for each (1:going forward in track, -1:going backward in track)
        # self.marines_position = {m: (0, 1) for m in marine_names}


        for marine, track in self.marines_tracks.items():
            # checks if the marine arrived at the last position in his track, if no, he keeps moving forward (direction = 1)
            # checks if the marine arrived at the first position in his track, if no, he keeps moving backwards (direction = -1)
            # if the marine's track is of length 1, the marine doesn't move

            if (state.marines_position[marine][0] < len(track) - 1) and ((state.marines_position[marine][1] == 1) or (state.marines_position[marine][0] == 1[i] == 0)):
                new_state.marines_position[marine] = (state.marines_position[marine][0] + 1, 1)
            elif (state.marines_position[marine][0] > 0) and ((state.marines_position[marine][1] == -1) or (state.marines_position[marine][0] == len(track) - 1)):
                new_state.marines_position[marine] = (state.marines_position[marine][0] - 1, -1)

        # dict of marines and their locations on the map (after the updating of the marines new positions)
        marines_locations = {marine: track[new_state.marines_position[marine][0]] for marine, track in self.marines_tracks.items()}

        if action[0] == 'deposit_treasures':
            new_state.num_treasures_held_per_pirate[pirate_name] = 0  # Updating the number of treasures held by the pirate ship that did this action.
            for treasure_name in new_state.treasures_locations.keys():
                if new_state.treasures_locations[treasure_name] == pirate_name:
                    new_state.treasures_locations[treasure_name] = 'b'

        if action[0] == 'sail':
            new_state.pirate_locations[pirate_name] = action[2]
                    
        if action[0] == 'collect_treasure' and new_state.num_treasures_held_per_pirate[pirate_name] < 2:
            new_state.num_treasures_held_per_pirate[pirate_name] += 1  # Updating the number of treasures held by the pirate ship that did this action.
            new_state.treasures_locations[action[2]] = pirate_name
        
        # if action[0] == 'wait', nothing changes (except the marines - handled below).

        for m in self.marines_tracks.keys():
            if marines_locations[m] == new_state.pirate_locations[pirate_name]:
                new_state.num_treasures_held_per_pirate[pirate_name] = 0
                for treasure_name in new_state.treasures_locations.keys():
                    if new_state.treasures_locations[treasure_name] == pirate_name:
                        new_state.treasures_locations[treasure_name] = self.islands_with_treasures[treasure_name]

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
        return np.sum(min_distances_to_base) #/ num_pirates



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

