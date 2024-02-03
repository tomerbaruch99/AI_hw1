import search
import random
import math
import numpy as np


ids = ["314779166", "322620873"]


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial_map = np.array(initial['map'])
        self.location_matrix = np.array(size=initial_map.shape)
        len_rows = self.location_matrix.shape[0]
        len_cols = self.location_matrix.shape[1]
        
        for i in range(len_rows):
            for j in range(len_cols):
                self.location_matrix[i][j] = list()
                if initial_map[i][j] == 'B':
                    self.location_matrix[i][j] = 'b'  # base; can deposit treasure here
                if i > 0 and initial_map[i-1][j] != 'I':
                    self.location_matrix[i][j].append('u')  # up
                if i < len_rows-1 and initial_map[i+1][j] != 'I':
                    self.location_matrix[i][j].append('d')  # down
                if j > 0 and initial_map[i][j-1] != 'I':
                    self.location_matrix[i][j].append('l')  # left
                if j < len_cols-1 and initial_map[i][j+1] != 'I':
                    self.location_matrix[i][j].append('r')  # right

        for treasure, location in initial['treasures'].items():
            i = location[0]
            j = location[1]
            if i > 0 and self.location_matrix[i-1][j] != 'I':
                self.location_matrix[i-1][j].append(('t', treasure.split('_')[1]))
            if i < len_rows-1 and initial_map[i+1][j] != 'I':
                self.location_matrix[i+1][j].append(('t', treasure.split('_')[1]))
            if j > 0 and self.location_matrix[i][j-1] != 'I':
                self.location_matrix[i][j-1].append(('t', treasure.split('_')[1]))
            if j < len_cols-1 and initial_map[i][j+1] != 'I':
                self.location_matrix[i][j+1].append(('t', treasure.split('_')[1]))      
        self.treasure_count = 0

        self.marine_track = list()
        self.marine_location_idx = list()
        self.direction = list()
        for track in initial['marine_ships'].values():
            self.marine_track.append(track)
            self.marine_location_idx.append(0)  # Index of track list, which indicates the location of the marine. The marine starts in the first entry of the track list.
            self.direction.append('n')  # Direction of the marine w.r.t its track: n = next item in list, p = previous item in list

        search.Problem.__init__(self, initial)  # TODO: Maybe change 'initial' to something else.
        

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        ''' state structure: dictionary of
        pirate_locations = list(), for example: self.pirate_location_idx = list()
                                                for start_point in initial['pirate_ships'].values():
                                                    self.pirate_location_idx.append(start_point)  # Indices of the initial location of the pirate ships.
        num_treasures_held_per_pirate = list()
        treasures_locations = list(). '''

        all_possible_actions = list()
        pirate_locations = state['pirate_locations']
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

                if type(a) is tuple and a[0] == 't' and state['num_treasures_held_per_pirate'] < 2:
                    all_possible_actions.append(('collect_treasure', pirate_name, 'treasure_' + a[1]))

            all_possible_actions.append(('wait', pirate_name))

        return all_possible_actions


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

        ''' state structure: dictionary of
        pirate_locations = list(), for example: self.pirate_location_idx = list()
                                                for start_point in initial['pirate_ships'].values():
                                                    self.pirate_location_idx.append(start_point)  # Indices of the initial location of the pirate ships.
        num_treasures_held_per_pirate = list()
        treasures_locations = list(). '''

        new_state = state
        pirate_num = action[1].split("_")[2]

        if action[0] == 'deposit_treasures':
            new_state['num_treasures_held_per_pirate'][pirate_num - 1] = 0  # Updating the number of treasures held by the pirate ship that did this action.
            for loc_idx in range(len(new_state['treasures_locations'])):
                if new_state['treasures_locations'][loc_idx] == pirate_num:
                    new_state['treasures_locations'][loc_idx] = 'b'

        if action[0] == 'sail':
            new_state['pirate_locations'][pirate_num - 1] = action[2]
            for m in range(len(self.marine_track)):
                if self.marine_track[m][self.marine_location_idx[m]] == action[2]:
                    for loc_idx in range(len(new_state['treasures_locations'])):
                        if new_state['treasures_locations'][loc_idx] == pirate_num:
                            new_state['treasures_locations'][loc_idx] = 'I'

        if action[0] == 'collect_treasure' and new_state['num_treasures_held_per_pirate'][pirate_num - 1] < 2:
            new_state['num_treasures_held_per_pirate'][pirate_num - 1] += 1  # Updating the number of treasures held by the pirate ship that did this action.
            new_state['treasures_locations'][action[2].split("_")[1]-1] = pirate_num
        
        # TODO: wait and notice marine locations.



    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return 0

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_onepiece_problem(game):
    return OnePieceProblem(game)

