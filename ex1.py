import search
import random
import math
import numpy as np

ids = ["314779166", "322620873"]

class State:
    """ state structure:
        pirate_locations = list(), for example: self.pirate_location_idx = list()
                                                for start_point in initial['pirate_ships'].values():
                                                    self.pirate_location_idx.append(start_point)  # Indices of the initial location of the pirate ships.
        num_treasures_held_per_pirate = list()
        treasures_locations = list(). """
    def __init__(self, pirates, treasures):
        self.pirate_locations = list()
        for p in pirates.values():
            self.pirate_locations.append(p)

        self.num_treasures_held_per_pirate = [0] * len(self.pirate_locations)

        self.treasures_locations = list()
        for t in treasures.values():
            self.treasures_locations.append(t)


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial_map = np.array(initial['map'])
        self.location_matrix = np.array(size=initial_map.shape)
        self.len_rows = self.location_matrix.shape[0]
        self.len_cols = self.location_matrix.shape[1]
        self.num_of_treasures = len(initial['treasures'])

        for i in range(self.len_rows):
            for j in range(self.len_cols):
                self.location_matrix[i][j] = list()

                if initial_map[i][j] == 'B':
                    self.base = (i,j)
                    self.location_matrix[i][j] = 'b'  # base; can deposit treasure here

                if self.is_valid_location((i-1, j)):
                    self.location_matrix[i][j].append('u')  # up
                if self.is_valid_location((i+1, j)):
                    self.location_matrix[i][j].append('d')  # down
                if self.is_valid_location((i, j-1)):
                    self.location_matrix[i][j].append('l')  # left
                if self.is_valid_location((i, j+1)):
                    self.location_matrix[i][j].append('r')  # right

        for treasure, location in initial['treasures'].items():
            i = location[0]
            j = location[1]
            for b in [-1, 1]:
                if self.is_valid_location((i + b, j)):
                    self.location_matrix[i + b][j].append(('t', treasure.split('_')[1]))
                if self.is_valid_location((i, j + b)):
                    self.location_matrix[i][j + b].append(('t', treasure.split('_')[1]))
                 
        self.marine_track = list()
        self.marine_location_idx = list()
        self.direction = list()
        for track in initial['marine_ships'].values():
            self.marine_track.append(track)
            self.marine_location_idx.append(0)  # Index of track list, which indicates the location of the marine. The marine starts in the first entry of the track list.
            self.direction.append('n')  # Direction of the marine w.r.t its track: n = next item in list, p = previous item in list

        search.Problem.__init__(self, State(initial['pirate_ships'], initial['treasures']))

    def is_valid_location(self, location):
        x, y = location
        return (0 <= x < self.len_rows) and (0 <= y < self.len_cols) and (self.location_matrix[x][y] != 'I')


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

            for a in action_options:
                if a == 'b':
                    all_possible_actions.append(("deposit_treasures", pirate_name))

                if a == 'u':
                    all_possible_actions.append(('sail', pirate_name, (row_location - 1, col_location)))
                if a == 'd':
                    all_possible_actions.append(('sail', pirate_name, (row_location + 1, col_location)))
                if a == 'l':
                    all_possible_actions.append(('sail', pirate_name, (row_location, col_location - 1)))
                if a == 'r':
                    all_possible_actions.append(('sail', pirate_name, (row_location, col_location + 1)))

                if type(a) is tuple and a[0] == 't' and state.num_treasures_held_per_pirate[pirate] < 2:
                    all_possible_actions.append(('collect_treasure', pirate_name, 'treasure_' + a[1]))

            all_possible_actions.append(('wait', pirate_name))

        return all_possible_actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

        new_state = state
        pirate_num = action[1].split("_")[2]

        if action[0] == 'deposit_treasures':
            new_state.num_treasures_held_per_pirate[pirate_num - 1] = 0  # Updating the number of treasures held by the pirate ship that did this action.
            for loc_idx in range(len(new_state.treasures_locations)):
                if new_state.treasures_locations[loc_idx] == pirate_num:
                    new_state.treasures_locations[loc_idx] = 'b'

        if action[0] == 'sail':
            new_state.pirate_locations[pirate_num - 1] = action[2]
            for m in range(len(self.marine_track)):
                if self.marine_track[m][self.marine_location_idx[m]] == action[2]:
                    for loc_idx in range(len(new_state.treasures_locations)):
                        if new_state.treasures_locations[loc_idx] == pirate_num:
                            new_state.treasures_locations[loc_idx] = 'I'

        if action[0] == 'collect_treasure' and new_state.num_treasures_held_per_pirate[pirate_num - 1] < 2:
            new_state.num_treasures_held_per_pirate[pirate_num - 1] += 1  # Updating the number of treasures held by the pirate ship that did this action.
            new_state.treasures_locations[action[2].split("_")[1]-1] = pirate_num
        
        if action[0] == 'wait':
            for m in range(len(self.marine_track)):
                if self.marine_track[m][self.marine_location_idx[m]] == new_state.pirate_locations[pirate_num - 1]:
                    for loc_idx in range(len(new_state.treasures_locations)):
                        if new_state.treasures_locations[loc_idx] == pirate_num:
                            new_state.treasures_locations[loc_idx] = 'I'
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
        self.h1(node) , self.h2(node)  # Find balance between the two heuristics
        return 0
    
    def h_1(self, node):
        return sum(1 for t in node.state.treasures_locations if t == 'I') / len(node.state.pirate_locations)


    def calculate_distances_from_treasure(self, num_pirates):
        queue = [(self.base, 0)]  # A list of tuples, each tuple represents the index of a node and its distance from the base node. queue[1] is the distance that we passed so far.
        opened = []
        normalized_distances = {}  # Dictionary representing the treasure number and its distance from the base.
        while queue:
            current = queue.pop(0)
            opened.append(current[0])
            for t_num in range(len(self.num_of_treasures)):
                if ('t', t_num+1) in self.location_matrix[current[0][0]][current[0][1]]:
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
            

    def h_2(self, node):
        num_pirates = len(node.state.pirate_locations)
        return self.calculate_distances_from_treasure(num_pirates)

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_onepiece_problem(game):
    return OnePieceProblem(game)

