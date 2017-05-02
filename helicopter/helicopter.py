import numpy as np
import math
import cv2
from random import randint

STILL = 0
UP = 1
DOWN = 2

NEGATIVE_PELLET_VALUE = -2
POSITIVE_PELLET_VALUE = -1
EMPTY_VALUE = 0
WALL_VALUE = 1
AGENT_VALUE = 2
GOAL_VALUE = 3

REWARD_SCALE = 0.5
GOAL_REWARD = 10 * REWARD_SCALE
POSITIVE_PELLET_REWARD = 2 * REWARD_SCALE
NEGATIVE_PELLET_REWARD = -2 * REWARD_SCALE
STEP_REWARD = 1 * REWARD_SCALE
CRASH_REWARD = -10 * REWARD_SCALE

TRANSFORMATIONS = {STILL:np.array([0,0]), UP:np.array([0,-1]), DOWN:np.array([0,1])}

class HelicopterEnv():
    '''
    Creates a helicopter environment. params is a dictionary, which must have
    the following key-value pairs:
        'length' : The length (width) of the helicopter environment. Must be
                   greater than or equal to the visible_width.
        'height' : Height for the helicopter environment. The playable area
                   will be height-2, because the top and bottom of the
                   environment are padded with a wall.
        'visible_width' : Width of the area observeable to the agent.
        'num_column_obstacles' : Number of column obstacles found throughout
                                 the environment. Must be < (length - 2 *
                                 visible_width)
        'num_positive_pellets' : Number of positive reward pellets in env
        'num_negative_pellets' : Number of negative reward pellets in env
        'flatten_output' : If the output returned by step() should be flattened.
        'padding' : Integer, number of layers of wall padding to apply to floor
                    and ceiling.
    '''
    def __init__(self, params):
        self.length = params['length']
        self.height = params['height']
        self.visible_width = params['visible_width']
        self.num_column_obstacles = params['num_column_obstacles']
        self.flatten_output = params['flatten_output']
        self.padding = params['padding']
        self.num_positive_pellets = params['num_positive_pellets']
        self.num_negative_pellets = params['num_negative_pellets']
        self.num_random_obstacles = params['num_random_obstacles']

        if self.num_random_obstacles != 0:
            assert(self.num_column_obstacles == 0)
        if self.num_column_obstacles != 0:
            assert(self.num_random_obstacles == 0)

        if self.num_column_obstacles != 0:
            assert(self.length > 2*self.visible_width)
            assert(self.num_column_obstacles < (self.length - 2*self.visible_width) / 5)

        self.time = 0
        self.done = True
        self.maze = np.zeros(shape=(self.length, self.height))
        self.agent_coords = np.array([0,0])
        self.column_obstacle_indices = []
        self.action_space = np.array([UP, DOWN, STILL])
        self.observation_space_shape = [self.visible_width * self.height]

    '''
    Returns current state of the environment, as seen by the agent.
    If flatten is:
        false: array of size [visible_width, height] returned
        true: array of size visible_width * height returned
    '''
    def get_state(self, flatten=False):
        offset = self.agent_coords[0]
        state = np.zeros(shape=(self.visible_width, self.height))

        state = np.copy(self.environment[offset:offset+self.visible_width])
        state[int(self.agent_coords[0] - offset), int(self.agent_coords[1])] = AGENT_VALUE

        if flatten:
            state = np.ndarray.flatten(state)
        return state.T


    '''
    Resets environment, returns first observation.
    '''
    def reset(self):
        self.time = 0
        self.done = False
        self.environment = np.zeros(shape=(self.length + self.visible_width, self.height))
        self.agent_coords = np.array([0,self.height / 2])

        # Populate goal
        goal_height = 1 + self.padding + np.random.randint(self.height - 2 - 2*self.padding)
        self.environment[self.length - 1, goal_height] = GOAL_VALUE

        # Put walls on top & bottom and add padding
        self.environment[:,0] = WALL_VALUE
        self.environment[:,self.height-1] = WALL_VALUE
        for i in range(self.padding):
            self.environment[:,1+i] = WALL_VALUE
            self.environment[:,self.height-2-i] = WALL_VALUE

        # Generate column obstacles
        self.column_obstacle_indices = []
        for i in range(self.num_column_obstacles):
            columns_start = self.visible_width
            index = columns_start + np.random.randint(self.length - 2*self.visible_width)
            col_regen_attempts = 0
            while ((index-2) in self.column_obstacle_indices) or ((index-1) in self.column_obstacle_indices) \
                  or (index in self.column_obstacle_indices) or ((index+1) in self.column_obstacle_indices) \
                  or ((index+2) in self.column_obstacle_indices):
                  index = columns_start + np.random.randint(self.length - 2*self.visible_width)
                  col_regen_attempts += 1
                  if col_regen_attempts > 500:
                      print("Something wrong with column regen")
            self.column_obstacle_indices.append(index)
            column_pass_width = 1 + np.random.randint(4)
            column_pass_base = 1 + self.padding + np.random.randint((self.height / 2) - (column_pass_width / 2))
            assert(column_pass_base != 0)
            assert(column_pass_base != self.height - 1)
            self.environment[index, ] = WALL_VALUE
            self.environment[index, column_pass_base:column_pass_base+column_pass_width] = EMPTY_VALUE

        # Generate random single point obstacles
        # empty_x, empty_y = np.where(self.maze == EMPTY_VALUE)
        # path_indices = np.random.choice(np.arange(self.height), size=(self.num_paths), replace=False)

        # Generate random obstacles
        if self.num_random_obstacles > 0:
            empty_x, empty_y = np.where(self.environment == EMPTY_VALUE)
            obstacle_indices = np.random.choice(np.arange(len(empty_x)),
                                              size=(self.num_random_obstacles),
                                              replace=False)
            for i in obstacle_indices:
                self.environment[empty_x[i], empty_y[i]] = WALL_VALUE

        # Generate goal pellets
        empty_x, empty_y = np.where(self.environment == EMPTY_VALUE)
        pellet_indices = np.random.choice(np.arange(len(empty_x)),
                                          size=(self.num_positive_pellets + self.num_negative_pellets),
                                          replace=False)
        for i in pellet_indices[0:self.num_positive_pellets]:
            self.environment[empty_x[i], empty_y[i]] = POSITIVE_PELLET_VALUE
        for i in pellet_indices[self.num_positive_pellets:]:
            self.environment[empty_x[i], empty_y[i]] = NEGATIVE_PELLET_VALUE

        return self.get_state(flatten=self.flatten_output)

    '''
    Steps environment with the specified action.
    Returns observation, reward, done.
    '''
    def step(self, action):
        assert(action in self.action_space)
        assert(not self.done)
        self.time = self.time + 1
        transformation = TRANSFORMATIONS[action]
        new_position = self.agent_coords + transformation
        new_position[0] += 1
        reward = 0

        new_pos_val = self.environment[new_position[0], new_position[1]]
        if new_pos_val == WALL_VALUE:
            self.done = True
            reward = CRASH_REWARD
        elif new_pos_val == GOAL_VALUE:
            self.done = True
            reward = GOAL_REWARD
        elif new_position[0] > self.length - 1:
            self.done = True
            reward = STEP_REWARD
        elif new_pos_val == POSITIVE_PELLET_VALUE:
            reward = POSITIVE_PELLET_REWARD
        elif new_pos_val == NEGATIVE_PELLET_VALUE:
            reward = NEGATIVE_PELLET_REWARD
        else:
            reward = STEP_REWARD

        self.agent_coords = new_position
        return self.get_state(flatten=self.flatten_output), reward, \
               self.done, "info_not_implemented"

    '''
    Prints the current state of the environment, as seen by the agent.
    '''
    def display_grid(self):
        print(self.get_state(flatten=False))

    def close(self):
        print('todo: any cleanup?')
