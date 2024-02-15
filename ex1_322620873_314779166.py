import search_322620873_314779166
import random
import math

ids = ["314779166", "322620873"]

class State:
    """
    state structure:
    pirate_locations = dict(), keys: pirate names, values: tuple of location indices in the map.
    treasures_locations = dict(), keys: treasure names, values: tuple of location indices of its island, the name of the pirate holding them, or 'b' for base.
    marines_position = dict(), keys: marine names, values: tuple of location index in track and direction for each (1:going forward in track, -1:going backward in track).
    num_treasures_held_per_pirate = dict(), keys: pirate names, values: number of treasures each pirate holds.
    """
    def __init__(self, pirates, treasures, marine_names):
        self.pirate_locations = {p: loc for p, loc in pirates.items()} # dict of pirates and their locations in the map
        self.treasures_locations = {t: loc for t, loc in treasures.items()} # dict of a treasure name with its location on the map

        # dict of marines and a tuple of location index in track and direction for each (1:going forward in track, -1:going backward in track)
        self.marines_position = {m: (0, 1) for m in marine_names}

        # dict of pirates and the number of treasure each one holds
        self.num_treasures_held_per_pirate = {p: 0 for p in pirates}
    

    def clone_state(self):
        """ Returns a new state that is a copy of the current state. """
        new_state = State(self.pirate_locations, self.treasures_locations, self.marines_position.keys())
        new_state.marines_position = {m: position for m, position in self.marines_position.items()}  # position is a tuple of location index in track and direction.
        new_state.num_treasures_held_per_pirate = {p: num_t for p, num_t in self.num_treasures_held_per_pirate.items()}
        return new_state


    def __lt__(self, other):
        return id(self) < id(other)
    
    def __eq__(self, other):
        return (self.pirate_locations == other.pirate_locations) and (self.treasures_locations == other.treasures_locations) and (self.num_treasures_held_per_pirate == other.num_treasures_held_per_pirate) and (self.marines_position == other.marines_position)

    def __hash__(self):
        return hash((frozenset(self.pirate_locations.items()), frozenset(self.treasures_locations.items()), frozenset(self.num_treasures_held_per_pirate.items()), frozenset(self.marines_position.items())))


class OnePieceProblem(search_322620873_314779166.Problem):
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
        # The 6th key, represented by 't', contains a tuple of all treasure names that can be collected in this location.

        self.memo = dict()  # A dictionary that represents the memoization of the heuristic function.
        self.memo_distances = dict()  # A dictionary that represents the distances between each pair of locations, so that we don't need to calculate the same distance more than once.
        self.min_distance_of_adjacent_cells = dict()  # A dictionary that represents the minimum distance from an adjacent cell to a location.

        self.len_rows = len(initial_map)
        self.len_cols = len(initial_map[0])

        def is_valid_location(location):
            x, y = location
            return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (initial_map[x][y] != 'I')

        for i in range(self.len_rows):
            for j in range(self.len_cols):
                self.location_dict[(i, j)] = dict()  # A dictionary that represents the possibilities in this location, as described above.

                if initial_map[i][j] == 'B':
                    self.base = (i, j)  # The location of the base.
                    self.location_dict[(i, j)]['b'] = True  # base; can deposit treasure here
                else:
                    self.location_dict[(i, j)]['b'] = False  # isn't the base
                    
        for i in range(self.len_rows):
            for j in range(self.len_cols):
                min_distance_to_base = float('inf')
                for direction, index in zip(['u', 'd', 'l', 'r'], [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]):
                    # Check if the location is valid and not an island, and if so, add the direction up\down\left\right to the location dictionary with the value True, so that we know that we can sail in this direction.
                    if is_valid_location(index):
                        self.location_dict[(i, j)][direction] = is_valid_location(index)

                        # Check if the distance has already been calculated and memoized
                        if (index, self.base) not in self.memo_distances:
                            distance = abs(index[0] - self.base[0]) + abs(index[1] - self.base[1])  # The L1-distance from the adjacent cell to the base (Manhattan Distance).
                            self.memo_distances[(index, self.base)] = distance
                            self.memo_distances[(self.base, index)] = distance  # Distances are symmetric
                            if distance < min_distance:
                                min_distance = distance
                    else:
                        self.location_dict[(i, j)][direction] = False

                if ((i, j), self.base) not in self.memo_distances:
                    self.min_distance_of_adjacent_cells[((i, j), self.base)] = min_distance_to_base  # Setting the minimum distance from the adjacent non-island cells to the base.

                self.location_dict[(i, j)]['t'] = tuple()  # A tuple of all treasure names that can be collected in this location.
        self.islands_with_treasures = initial['treasures']  # A dictionary that represents the treasures and their locations when on their island.
        for treasure, location in initial['treasures'].items():  # For each treasure, update the locations that from them we can collect the treasure.
            i = location[0]
            j = location[1]
            min_distance_to_base = float('inf')

            for b in [-1, 1]:
                for new_loc in [(i + b, j), (i, j + b)]:  # The locations that are adjacent to the treasure location.
                    if is_valid_location(new_loc):
                        self.location_dict[new_loc]['t'] += (treasure,)  # Add the treasure to the tuple of treasures that can be collected in this location.
                        
        self.marines_tracks = initial['marine_ships']  # A dictionary that represents the tracks of the marine ships.
        
        initial_state = tuple()
        for pirate in initial['pirate_ships'].keys():
            initial_state += (State(pirate, initial['treasures'], initial['marine_ships'].keys()),)
        initial_state = State(initial['pirate_ships'], initial['treasures'], initial['marine_ships'].keys())  # Create the initial state.
        search.Problem.__init__(self, initial_state)
    

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        all_possible_actions = tuple()
        pirate_locations = state.pirate_locations
        
        for pirate_name in pirate_locations.keys():
            row_location = pirate_locations[pirate_name][0]
            col_location = pirate_locations[pirate_name][1]
            action_options = self.location_dict[(row_location, col_location)]
            
            # Reminder to the keys and their meanings:
            # b=base, u=up, d=down, l=left, r=right, t=treasure collecting.
            if action_options['b']: all_possible_actions += (("deposit_treasures", pirate_name),)

            if action_options['u']: all_possible_actions += (('sail', pirate_name, (row_location - 1, col_location)),)
            if action_options['d']: all_possible_actions += (('sail', pirate_name, (row_location + 1, col_location)),)
            if action_options['l']: all_possible_actions += (('sail', pirate_name, (row_location, col_location - 1)),)
            if action_options['r']: all_possible_actions += (('sail', pirate_name, (row_location, col_location + 1)),)
            
            if len(action_options['t']) and (state.num_treasures_held_per_pirate[pirate_name] < 2):
                for t in action_options['t']:
                    all_possible_actions += (('collect_treasure', pirate_name, t),)

            all_possible_actions += (('wait', pirate_name),)

        return all_possible_actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = state.clone_state()
        pirate_name = action[1]

        for marine, track in self.marines_tracks.items():
            # Checks if the marine arrived at the last position in its track, if not, it keeps moving forward (direction = 1).
            # Also checks if the marine arrived at the first position in its track, if not, it keeps moving backwards (direction = -1).
            # If the marine's track is of length 1, the marine doesn't move.

            if (state.marines_position[marine][0] < len(track) - 1) and ((state.marines_position[marine][1] == 1) or (state.marines_position[marine][0] == 0)):
                new_state.marines_position[marine] = (state.marines_position[marine][0] + 1, 1)
            elif (state.marines_position[marine][0] > 0) and ((state.marines_position[marine][1] == -1) or (state.marines_position[marine][0] == len(track) - 1)):
                new_state.marines_position[marine] = (state.marines_position[marine][0] - 1, -1)

        if action[0] == 'deposit_treasures':
            new_state.num_treasures_held_per_pirate[pirate_name] = 0  # Updating the number of treasures held by the pirate ship that did this action.
            for treasure_name in new_state.treasures_locations.keys():
                if new_state.treasures_locations[treasure_name] == pirate_name:
                    new_state.treasures_locations[treasure_name] = 'b'

        if action[0] == 'sail':
            new_state.pirate_locations[pirate_name] = action[2]
                    
        if (action[0] == 'collect_treasure') and (new_state.num_treasures_held_per_pirate[pirate_name] < 2):
            new_state.num_treasures_held_per_pirate[pirate_name] += 1  # Updating the number of treasures held by the pirate ship that did this action.
            new_state.treasures_locations[action[2]] = pirate_name
        
        # if action[0] == 'wait', nothing changes (except the marines - handled below).
        
        # Dict of marines and their locations on the map (after the updating of the marines' new positions)
        marines_locations = {marine: track[new_state.marines_position[marine][0]] for marine, track in self.marines_tracks.items()}

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
        for t in state.treasures_locations.values():
            if t != 'b':  # The goal state is when all the treasures are in the base.
                return False
        return True

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return self.h_3(node)
    

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
                if (0 <= x < self.len_rows) and (0 <= y < self.len_cols):
                    if (self.location_dict[location][direction]) and (x != self.base[0] or y != self.base[1]):  # If the adjacent cell is a sea cell.
                        temp_dist = abs(self.base[0]-x) + abs(self.base[1]-y)  # The L1-distance from the adjacent cell to the base (Manhattan Distance).

                        # Update the distance from the treasure to the base if the new distance is shorter.
                        if temp_dist < min_distances_to_base[t]:
                            min_distances_to_base[t] = temp_dist

        return sum(min_distances_to_base) / num_pirates


    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""

    def calculate_distance(self, start, end):
        # Check if the distance has already been calculated and memoized
        if (start, end) in self.memo_distances:
            return self.memo_distances[(start, end)]
        
        # Calculate the distance
        distance = abs(start[0] - end[0]) + abs(start[1] - end[1])

        # Memoize the distance
        self.memo_distances[(start, end)] = distance
        self.memo_distances[(end, start)] = distance  # Distances are symmetric
        return distance
    
    def calculate_min_distance_of_adjacent_cells(self, reference_point, goal_point, treasure_condition=False):
        # Check if the distance has already been calculated and memoized
        if (reference_point, goal_point) in self.min_distance_of_adjacent_cells:
            return self.min_distance_of_adjacent_cells[(reference_point, goal_point)]
        
        # Calculate the minimum distance from all the valid adjacent cells of the reference point to the goal point
        min_distance = float('inf')
        condition_flag = False
        for direction, index in zip(['u', 'd', 'l', 'r'], [(-1,0), (1,0), (0,-1), (0,1)]):
            x = reference_point[0] + index[0]
            y = reference_point[1] + index[1]

            if treasure_condition:
                condition_flag = True if self.location_dict[(x, y)]['t'] else False
            else:
                condition_flag = True if self.location_dict[reference_point][direction] else False
            if condition_flag and (0 <= x < self.len_rows) and (0 <= y < self.len_cols):
                temp_dist = self.calculate_distance((x, y), goal_point)
                if temp_dist < min_distance:
                    min_distance = temp_dist
        return min_distance


    def h_3(self, node):
        # Check if the heuristic has already been calculated and memoized for this node.
        if node in self.memo.keys():
            return self.memo[node]
        
        score = 0  # The heuristic score.
        weights = [0.8, 0.4, 0.1]  # Weights that represent the importance of the amount of treasures held by the pirates.

        # The score will be larger the more the pirate is further from the base and holds less treasures,
        # and will be smaller the more the pirate is closer to the base with more treasures.
        for pirate, location in node.state.pirate_locations.items():
            min_distance_to_base = self.calculate_min_distance_of_adjacent_cells(location, self.base)
            score += (min_distance_to_base * weights[node.state.num_treasures_held_per_pirate[pirate]])**1.8 # ()^2

        # Finds the current locations of all the marines on the map.
        marine_locations = tuple()
        for marine, location_idx in node.state.marines_position.items():
            marine_locations += self.marines_tracks[marine][location_idx[0]]

        for treasure, t_location in node.state.treasures_locations.items():

            # If the pirate is caught by a marine, the score will be larger the more treasures it held and now lost.
            # This penalty for being caught is calculate w.r.t the distance from the pirate to the initial location of the treasure
            # (out of thinking that all of the turns that passed since the pirate collected the treasure and up until now, are a waste).
            if (type(t_location) == str) and (t_location != 'b'):  # If the treasure's location is a name of a pirate ship, it means that the pirate is holding the treasure.
                pirate = t_location
                p_location = node.state.pirate_locations[pirate]
                if p_location in marine_locations:
                    score += weights[2-node.state.num_treasures_held_per_pirate[pirate]]*(self.calculate_min_distance_of_adjacent_cells(self.islands_with_treasures[treasure], p_location, True))
            
            # If the treasure is on an island, the score will be calculated using the distance from it to the nearest pirate, and the number of treasures the pirate currently holds.
            elif type(t_location) == tuple:
                closest_pirate_distance = float('inf')
                for pirate, p_location in node.state.pirate_locations.items():
                    temp_distance = self.calculate_min_distance_of_adjacent_cells(t_location, p_location, True)
                    if temp_distance < closest_pirate_distance:
                        closest_pirate_distance = temp_distance
                score += weights[2-node.state.num_treasures_held_per_pirate[pirate]] * closest_pirate_distance #**2
        
        self.memo[node] = score
        return score


def create_onepiece_problem(game):
     return OnePieceProblem(game)

