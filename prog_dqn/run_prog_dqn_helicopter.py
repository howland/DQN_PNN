import argparse
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

# Hack to import dqn from parallel directory
import os
import sys
from os import path
sys.path.append("../")

import prog_dqn
from dqn_utils import *
from helicopter import helicopter
import prog_nn

Q_FUNC_PARAM_COL_NUM = 'column_number'
CHECKPOINT_BASE_PATH = 'checkpoint_path'
CHECKPOINTS_LIST = 'checkpoints_list'


'''
Function passed to dqn used for creating the progressive neural network column
for q value estimation.
'''
def helicopter_model(input_shape, num_actions, session, prog_q_func_params, scope, reuse=False):
    column_number = prog_q_func_params[Q_FUNC_PARAM_COL_NUM]
    checkpoint_base_path = prog_q_func_params[CHECKPOINT_BASE_PATH]
    checkpoint_list = prog_q_func_params[CHECKPOINTS_LIST]
    # TODO: parametrize topology in prog_q_func_params

    if column_number < 1:
        # Assuming input_shape is length 1 (haven't implemented convolution yet)
        topology1 = [input_shape[0], 128, 128, 64, num_actions]
        activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]

        column = prog_nn.InitialColumnProgNN(topology1, activations, session, checkpoint_base_path, dtype=tf.float32)
    else:
        # TODO: SEPERATE, STORE, AND LOAD FOR EACH INDIVIDUAL COLUMN BEING RESTORED!!!
        topology1 = [input_shape[0], 128, 128, 64, num_actions]
        activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]

        # Restore previous columns here
        prev_columns = []
        for i in range(column_number):
            print("reconstructing column i: ", i)
            if i == 0:
                col_i = prog_nn.InitialColumnProgNN(topology1, activations, session, checkpoint_base_path, dtype=tf.float32)
            else:
                col_i = prog_nn.ExtensibleColumnProgNN(topology1, activations, session, checkpoint_base_path, prev_columns, dtype=tf.float32)

            col_i.restore_weights(checkpoint_list[i])
            prev_columns.append(col_i)
            print("column successfully restored")

        print("previous columns are: ", prev_columns)

        column = prog_nn.ExtensibleColumnProgNN(topology1, activations, session, checkpoint_base_path, prev_columns, dtype=tf.float32)

    return column

'''
Trains a column of the PNN using DQN for 100000000 time steps. After
num_timesteps, exploration is fixed at 0.01.

See sample invocations in main()
'''
def helicopter_learn(env, session, num_timesteps, q_func_params):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = prog_dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        return t > 100000000

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.2),
            ((1e6)/2, 0.01),
            (num_iterations / 10, 0.01),
        ], outside_value=0.01
    )

    prog_dqn.learn(
        env,
        prog_q_func=helicopter_model,
        prog_q_func_params=q_func_params,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    return session

'''
Deals with running pnn dqn. In the current state, the run configuration is hard
coded in main. To change the column to train, specify the environment parameters
for that column, and also the q function parameters. Sample configurations for
the first three columns (0, 1, 2) can be found below, although only column 0 is
trained.
'''
def main():
    # Establish parameters for specific column
    env_0_params = { # Col 0 params
            'length' : 10,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 0,
            'flatten_output' : True,
            'padding' : 0,
    }

    env_1_params = {
            'length' : 30,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 1,
            'flatten_output' : True,
            'padding' : 0,
    }

    env_2_params = {
            'length' : 100,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 15,
            'flatten_output' : True,
            'padding' : 0,
    }

    col_0_q_params = {
        Q_FUNC_PARAM_COL_NUM : 0,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [],
    }

    # For second column
    col_1_q_params = {
        Q_FUNC_PARAM_COL_NUM : 1,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [60], # Change to latest checkpoint for col 0
    }

    # For third column
    col_2_q_params = {
        Q_FUNC_PARAM_COL_NUM : 2,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [60, 40], # Change to latest checkpoints for col 0 and 1
    }

    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    # Specify params for  col 0, 1, or 2
    env = helicopter.HelicopterEnv(env_0_params)
    set_global_seeds(seed)
    # env.seed(seed)
    session = get_session()
    # Specify params for  col 0, 1, or 2
    helicopter_learn(env, session, num_timesteps=int(4e7), q_func_params=col_0_q_params)

if __name__ == "__main__":
    main()
