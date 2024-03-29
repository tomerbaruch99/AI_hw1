import search
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


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.initial_map = initial['map']
        self.location_dict = dict()
        # A dictionary that represents the map. The keys are the indices of the map, 
        # and the values are dictionaries that represent the possibilities in this location.
        # The possibilities are: 'b'=base, 'u'=up, 'd'=down, 'l'=left, 'r'=right, 't'=treasure collecting.
        # These are the keys of the inner dictionaries, and the values in the first five keys are booleans that represent whether the corresponding action is possible in this location.
        # The 6th key, represented by 't', contains a tuple of all treasure names that can be collected in this location.

        self.len_rows = len(self.initial_map)
        self.len_cols = len(self.initial_map[0])

        def is_valid_location(location):
            x, y = location
            return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.initial_map[x][y] != 'I')

        for i in range(self.len_rows):
            for j in range(self.len_cols):
                self.location_dict[(i, j)] = dict()  # A dictionary that represents the possibilities in this location, as described above.

                if self.initial_map[i][j] == 'B':
                    self.base = (i, j)  # The location of the base.
                    self.location_dict[(i, j)]['b'] = True  # base; can deposit treasure here
                else:
                    self.location_dict[(i, j)]['b'] = False  # isn't the base

                for direction, index in zip(['u', 'd', 'l', 'r'], [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]):
                    # Check if the location is valid and not an island, and if so, add the direction up\down\left\right to the location dictionary with the value True, so that we know that we can sail in this direction.
                    self.location_dict[(i, j)][direction] = is_valid_location(index)
                
                self.location_dict[(i, j)]['t'] = tuple()  # A tuple of all treasure names that can be collected in this location.
                
        self.islands_with_treasures = initial['treasures']  # A dictionary that represents the treasures and their locations when on their island.

        for treasure, location in initial['treasures'].items():  # For each treasure, update the locations that from them we can collect the treasure.
            i = location[0]
            j = location[1]
            for b in [-1, 1]:
                for new_loc in [(i + b, j), (i, j + b)]:  # The locations that are adjacent to the treasure location.
                    if is_valid_location(new_loc):
                        self.location_dict[new_loc]['t'] += (treasure,)  # Add the treasure to the tuple of treasures that can be collected in this location.
        
        self.marines_tracks = initial['marine_ships']  # A dictionary that represents the tracks of the marine ships.

        self.memo = dict()  # A dictionary that represents the memoization of the heuristic function.
        self.memo_distances = dict()  # A dictionary that represents the distances between each pair of locations, so that we don't need to calculate the same distance more than once.
        
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

    def is_island(self, location):
        x, y = location
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.initial_map[x][y] != 'I')
# 0 treasures means need to collect more treasures - small weight
    def h_3(self, node):
        weights = [0.8, 0.4, 0.1]  # weighted num of treasures each pirate holds
        # weights = [0.9, 0.4, 0.225]
        score = 0
        location_next_step = 0

        for location in self.islands_with_treasures.values():
            accessible_treasure = False
            for direction, index in zip(['u', 'd', 'l', 'r'], [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                x = location[0] + index[0]  # The x coordinate of the adjacent cell.
                y = location[1] + index[1]  # The y coordinate of the adjacent cell.
                if self.is_island((x, y)):
                    accessible_treasure = True
            if not accessible_treasure:
                score += float('inf')
                return score

        marines_next_locations = dict()
        for marine, track in self.marines_tracks.items():
            if (node.state.marines_position[marine][0] < len(track) - 1) and (
                    (node.state.marines_position[marine][1] == 1) or (node.state.marines_position[marine][0] == 0)):
                marines_next_locations[marine] = track[node.state.marines_position[marine][0] + 1]
            elif (node.state.marines_position[marine][0] > 0) and (
                    (node.state.marines_position[marine][1] == -1) or (node.state.marines_position[marine][0] == len(track) - 1)):
                marines_next_locations[marine] = track[node.state.marines_position[marine][0] - 1]


        if node in self.memo.keys():
            return self.memo[node]
        # pirate_best_route_to_base = {pirate: (0,0) for pirate in node.state.pirate_locations.keys()}
        for pirate, location in node.state.pirate_locations.items():
            for marine_next_position in marines_next_locations.values():
                min_distances_to_base = float('inf')
                closest_treasure_dist = float('inf')
                for direction, index in zip(['u', 'd', 'l', 'r'], [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    if self.location_dict[location][direction]:
                        x = location[0] + index[0]  # The x coordinate of the adjacent cell.
                        y = location[1] + index[1]  # The y coordinate of the adjacent cell.
                        if (0 <= x < self.len_rows) and (0 <= y < self.len_cols):
                            if node.state.num_treasures_held_per_pirate[pirate] < 2:
                                for treasure in node.state.treasures_locations.values():
                                    if type(treasure) == tuple:
                                        temp_closest_treasure_dist = self.calculate_distance((x, y),treasure)  # The L1-distance from the adjacent cell to the base (Manhattan Distance).
                                        if temp_closest_treasure_dist < closest_treasure_dist:
                                            closest_treasure_dist = temp_closest_treasure_dist
                                            location_next_step = (x,y)
                            else:
                                temp_dist = self.calculate_distance((x, y),self.base)  # The L1-distance from the adjacent cell to the base (Manhattan Distance).
                                if temp_dist < min_distances_to_base:
                                    min_distances_to_base = temp_dist
                                    # next possible location of the pirate
                                    location_next_step = (x, y)
                                score += (min_distances_to_base * weights[node.state.num_treasures_held_per_pirate[pirate]])**1.8 # ()^2

                        num_treasures = node.state.num_treasures_held_per_pirate[pirate]
                        if location_next_step == marine_next_position:
                            for treasure, treasure_location in node.state.treasures_locations.items():
                                if num_treasures == 0:
                                    break
                                if treasure_location == pirate:
                                    num_treasures -= 1
                                    treasure_loc_to_take_again = self.islands_with_treasures[treasure]
                                    # score += self.calculate_distance(next_loc_pirate, treasure_loc_to_take_again) #*(0.5**(2-num_treasures))
                                    score += weights[2-num_treasures] * (self.calculate_distance(location_next_step, treasure_loc_to_take_again))
                                    location_next_step = treasure_loc_to_take_again

                # Update the distance from the treasure to the base if the new distance is shorter.


            # pirate_best_route_to_base[pirate] = (min_arg, min_distances_to_base)


                # next_loc_pirate = min_arg
                # if (node.state.num_treasures_held_per_pirate[pirate] == 2) and (next_loc_pirate == marine_next_position):
                #         for treasure, treasure_location in node.state.treasures_locations.items():
                #             if treasure_location == pirate:
                #                 treasure_loc_to_take_again = self.islands_with_treasures[treasure]
                #                 # score += self.calculate_distance(next_loc_pirate, treasure_loc_to_take_again) #*(0.5**(2-num_treasures))
                #                 score += weights[0] * (
                #                     self.calculate_distance(next_loc_pirate, treasure_loc_to_take_again))
                #                 next_loc_pirate = treasure_loc_to_take_again
                # next_loc_pirate = closest_treasure_next_step_pirate
                # if (node.state.num_treasures_held_per_pirate[pirate] == 1) and (
                #         next_loc_pirate == marine_next_position):
                #     for treasure, treasure_location in node.state.treasures_locations.items():
                #         if treasure_location == pirate:
                #             treasure_loc_to_take_again = self.islands_with_treasures[treasure]
                #             # score += self.calculate_distance(next_loc_pirate, treasure_loc_to_take_again) #*(0.5**(2-num_treasures))
                #             score += weights[2 - num_treasures] * (
                #                 self.calculate_distance(next_loc_pirate, treasure_loc_to_take_again))
                #             next_loc_pirate = treasure_loc_to_take_again
                #

            # for marine_next_position in marines_next_locations.values():
            #     for num_treasures in [2,1,0]:
            #         if (node.state.num_treasures_held_per_pirate[pirate] == num_treasures) and (next_loc_pirate == marine_next_position):
            #             for treasure, treasure_location in node.state.treasures_locations.items():
            #                 if treasure_location == pirate:
            #                     treasure_loc_to_take_again = self.islands_with_treasures[treasure]
            #                     # score += self.calculate_distance(next_loc_pirate, treasure_loc_to_take_again) #*(0.5**(2-num_treasures))
            #                     score += weights[2-num_treasures]*(self.calculate_distance(next_loc_pirate, treasure_loc_to_take_again))
            #                     next_loc_pirate = treasure_loc_to_take_again
            # score += (min_distances_to_base * weights[node.state.num_treasures_held_per_pirate[pirate]])**1.8 # ()^2

        for treasure in node.state.treasures_locations.values():
            closest_pirate_distance = float('inf')
            if type(treasure) == tuple:
                for pirate, location in node.state.pirate_locations.items():
                    for index in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Checking all adjacent cells to see if they can be used to collect the treasure.
                        x = treasure[0] + index[0]
                        y = treasure[1] + index[1]
                        if (0 <= x < self.len_rows) and (0 <= y < self.len_cols):
                            if treasure in self.location_dict[(x, y)]['t']:
                                temp_dist = self.calculate_distance((x, y), location)
                                if temp_dist < closest_pirate_distance:
                                    closest_pirate_distance = temp_dist
                    score += closest_pirate_distance * (weights[2 - node.state.num_treasures_held_per_pirate[pirate]])**2
        self.memo[node] = score
        return score

    import math

#     def calculate_manhattan_distance(self, point1, point2):
#         return abs(int(point1[0]) - int(point2[0])) + abs(int(point1[1]) - int(point2[1]))
#
#     def calculate_nearest_treasure_distance(self, state, pirate):
#         # Assuming state has information about islands and treasures
#         pirate_position = state.pirate_locations[pirate]
#         nearest_treasure_distance = float('inf')
#
#         for treasure_position in state.treasures_locations.values():
#             if type(treasure_position) == tuple:
#                 distance = self.calculate_manhattan_distance(pirate_position, treasure_position)
#                 nearest_treasure_distance = min(nearest_treasure_distance, distance)
#
#         return nearest_treasure_distance
#
#     def calculate_distance_to_base(self, state, pirate):
#         pirate_position = state.pirate_locations[pirate]
#         base_position = self.base
#         return self.calculate_manhattan_distance(pirate_position, base_position)
#
#     def calculate_marine_ship_penalty(self, state, pirate):
#         pirate_position = state.pirate_locations[pirate]
#         marines_locations = {marine: track[state.marines_position[marine][0]] for marine, track in self.marines_tracks.items()}
#
#         # Assuming state has information about marine ships
#         for marine_position in marines_locations.values():
#             if self.calculate_manhattan_distance(pirate_position, marine_position) <= 1:
#                 # Penalize if pirate is too close to a marine ship
#                 return float('inf')  # or another large penalty value
#         return 0
#
#     def calculate_remaining_treasures_value(self, state):
#         remaining_treasures_value = 0
#
#         # Assuming state has information about islands and treasures
#         for treasure_position in state.treasures_locations.values():
#             if type(treasure_position) == tuple:
#                 # If the treasure has not been collected, add its value to the total
#                 remaining_treasures_value += treasure_position[0]+treasure_position[1]
#
#         return remaining_treasures_value
#
#     def h_4(self, node):
#         total_heuristic_value = 0
#
#         for pirate in node.state.pirate_locations.keys():
#                 # Factor 1: Distance to the nearest treasure
#             nearest_treasure_distance = self.calculate_nearest_treasure_distance(node.state, pirate)
#             total_heuristic_value += nearest_treasure_distance
#
#             # Factor 2: Proximity to base
#             base_distance = self.calculate_distance_to_base(node.state, pirate)
#             total_heuristic_value += base_distance
#
#             # Factor 3: Blocking marine ships
#             marine_ship_penalty = self.calculate_marine_ship_penalty(node.state, pirate)
#             total_heuristic_value += marine_ship_penalty
#
#             # Factor 4: Treasures on board
#             treasures_on_board = node.state.num_treasures_held_per_pirate[pirate]
#             total_heuristic_value -= treasures_on_board
#
#         # Factor 5: Remaining treasures on islands
#         remaining_treasures_value = self.calculate_remaining_treasures_value(node.state)
#         total_heuristic_value += remaining_treasures_value
#
#         # Factor 6: Avoiding loops (you may need to implement a loop detection mechanism)
#
#         # Factor 7: Efficient path planning
#
#         # Factor 8: Avoiding conflicts
#
#         return total_heuristic_value
#
#
def create_onepiece_problem(game):
     return OnePieceProblem(game)
#
#
#     """Feel free to add your own functions
#     (-2, -2, None) means there was a timeout"""

    # def calculate_distances_from_treasure(self, num_pirates):
    #     ''' We use BFS to calculate the distance from the base to each treasure. We normalize the distances by the number of pirates.
    #         This function is currently redundent, but we had no heart deleting it :( '''
    #     queue = [(self.base, 0)]  # A list of tuples, each tuple represents the index of a node and its distance from the base node. queue[1] is the distance that we passed so far.
    #     opened = []
    #     normalized_distances = {}  # Dictionary representing the treasure number and its distance from the base.
    #     while queue:
    #         current = queue.pop(0)
    #         opened.append(current[0])
    #         for t_num in range(len(self.num_of_treasures)):
    #             if ('t', t_num+1) in self.location_matrix[current[0][0]][current[0][1]]:  # Changed the location matrix since then
    #                 normalized_distances[t_num+1] = current[1]/num_pirates
    #         if len(normalized_distances.keys()) == len(self.num_of_treasures):
    #             return normalized_distances
    #         for i in [-1,1]:
    #             potential_child_location = (current[0][0] + i, current[0][1])
    #             if self.is_valid_location(potential_child_location) and (potential_child_location not in opened):
    #                 queue.append((potential_child_location, current[1] + 1))
    #             potential_child_location = (current[0][0], current[0][1] + i)
    #             if self.is_valid_location(potential_child_location) and (potential_child_location not in opened):
    #                 queue.append((potential_child_location, current[1] + 1))
    #
    #     for t_num in range(len(self.num_of_treasures)):
    #         if t_num+1 not in normalized_distances.keys():
    #             normalized_distances[t_num+1] = float('inf')
    #     return normalized_distances
    #
    #
