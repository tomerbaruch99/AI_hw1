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
    def __init__(self, pirates, treasures, num_marines):
        self.pirate_locations = list()
        for p in pirates:
            self.pirate_locations.append(p)

        self.num_treasures_held_per_pirate = [0] * len(self.pirate_locations)
        self.treasures_locations = [t for t in treasures]
        
        self.marines_locations_indices = np.zeros(shape=num_marines, dtype=int)  # Index of track list, which indicates the location of the marine. The marine starts in the first entry of the track list.
        self.marines_directions = np.ones(shape=num_marines)  # Direction of the marine w.r.t its track: 1 = next item in list, -1 = previous item in list


    def clone_state(self):
        new_state = State(self.pirate_locations, self.treasures_locations, len(self.marines_locations_indices))
        new_state.num_treasures_held_per_pirate = [p for p in self.num_treasures_held_per_pirate]
        for m in range(len(self.marines_locations_indices)):
            new_state.marines_locations_indices[m] = self.marines_locations_indices[m]
            new_state.marines_directions[m] = self.marines_directions[m]
        return new_state


    def __lt__(self, other):
        return id(self) < id(other)  # TODO: Changed tie braker if necessary.


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial_map = initial['map']
        self.location_dict = dict()
        # A dictionary that represents the map. The keys are the indices of the map, 
        # and the values are dictionaries that represent the possibilities in this location.
        # The possibilities are: 'b'=base, 'u'=up, 'd'=down, 'l'=left, 'r'=right, 't'=treasure collecting.
        # These are the keys of the inner dictionaries, and the values in the first five keys are booleans that represent whether the corresponding action is possible in this location.
        # The 6th key, represented by 't', contains a list of all treasure names that can be collected in this location.

        len_rows = len(initial_map)
        len_cols = len(initial_map[0])
        for i in range(len_rows):
            for j in range(len_cols):
                self.location_dict[(i, j)] = dict()  # A dictionary that represents the possibilities in this location, as described above.

                if initial_map[i][j] == 'B':
                    self.base = (i, j)  # The location of the base.
                    self.location_dict[(i, j)]['b'] = True  # base; can deposit treasure here
                else:
                    self.location_dict[(i, j)]['b'] = False  # isn't the base

                for direction, index in zip(['u', 'd', 'l', 'r'], [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]):
                    # Check if the location is valid and not an island, and if so, add the direction up\down\left\right to the location dictionary with the value True, so that we know that we can sail in this direction.
                    self.location_dict[(i, j)][direction] = is_valid_location(index)
                
                self.location_dict[(i, j)]['t'] = list()  # A list of all treasure names that can be collected in this location.
                
        self.islands_with_treasures = initial['treasures']  # A dictionary that represents the treasures and their locations when on their island.

        def is_valid_location(location):
            x, y = location
            return (0 <= x < len_rows) and (0 <= y < len_cols) and (initial_map[x][y] != 'I')

        for treasure, location in initial['treasures'].items():  # For each treasure, update the locations that from them we can collect the treasure.
            i = location[0]
            j = location[1]
            for b in [-1, 1]:
                for location in [(i + b, j), (i, j + b)]:  # The locations that are adjacent to the treasure location.
                    if is_valid_location(location):
                        self.location_dict[location]['t'].append(treasure)  # Add the treasure to the list of treasures that can be collected in this location.
        
        self.marines_tracks = initial['marine_ships']  # A dictionary that represents the tracks of the marine ships.

        initial_state = State(initial['pirate_ships'], initial['treasures'], initial['marine_ships'].keys())  # Create the initial state.
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

            for power, index in enumerate(range(5,self.num_digits+5)):
                if action_options[index] and (state.num_treasures_held_per_pirate[pirate] < 2):
                    t_num = action_options[index] % (10 ** (power+1))
                    while t_num:
                        all_possible_actions.append(('collect_treasure', pirate_name, 'treasure_' + str(t_num)))
                        t_num = action_options[index] // (10 ** (power+1))

            all_possible_actions.append(('wait', pirate_name))

        return all_possible_actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = state.clone_state()
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
        for t in state.treasures_locations.values():
            if t != 'b':  # The goal state is when all the treasures are in the base.
                return False
        return True
    

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return self.h_2(node)
    

    def h_1(self, node):
        """ Summing the indicators of the treasures that are still on their island
        (a treaure that is still on the island is represented by a tuple, while a treasure that isn't is represented by 'b' or a pirate name),
        and dividing by the number of pirates. """
        return sum(1 for t in node.state.treasures_locations.values() if type(t) == tuple) / len(node.state.pirate_locations.keys())


    def h_2(self, node):
        """ Summing the minimum distances from each treasure to the base (using Manhattan Distance),
        and dividing by the number of pirates. """
        num_pirates = len(node.state.pirate_locations.keys())

        # A list of the minimum distances from each treasure to the base.
        # Initialized with infinity values (so that the distance of an unreachable treasure will be infinity) and updated with the actual distances.
        min_distances_to_base = [float('inf')] * len(node.state.treasures_locations.keys())
        
        for t, location in enumerate(node.state.treasures_locations.values()):
            if location == 'b':
                min_distances_to_base[t] = 0  # The distance from the base to the base is 0.
                continue
            elif type(location) == str:
                location = node.state.pirate_locations[location]  # If the treasure is on a pirate ship, get the location of the pirate ship.
            
            # For each direction, check if there is an adjacent sea cell in this direction and if so, update the distance from this adjacent cell to the base - but only if it's shorter than the current distance.
            for direction, index in zip(['u', 'd', 'l', 'r'], [(-1,0), (1,0), (0,-1), (0,1)]):
                x = location[0] + index[0]  # The x coordinate of the adjacent cell.
                y = location[1] + index[1]  # The y coordinate of the adjacent cell.
                if (self.location_dict[location][direction] == True) and (x != self.base[0] or y != self.base[1]):  # If the adjacent cell is a sea cell.
                    temp_dist = abs(self.base[0]-x) + abs(self.base[1]-y)  # The L1-distance from the adjacent cell to the base (Manhattan Distance).
                    
                    # Update the distance from the treasure to the base if the new distance is shorter.
                    if temp_dist < min_distances_to_base[t]:
                        min_distances_to_base[t] = temp_dist

        return sum(min_distances_to_base) / num_pirates


def create_onepiece_problem(game):
    return OnePieceProblem(game)


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


