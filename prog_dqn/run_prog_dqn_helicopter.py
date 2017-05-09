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
RESTORE_FROM_SAVE = 'resore_from_save'
RESTORE_CHECKPOINT = 'restore_from_save_path'


'''
Function passed to dqn used for creating the progressive neural network column
for q value estimation.
'''
def helicopter_model(input_shape, num_actions, session, prog_q_func_params, scope, reuse=False):
    column_number = prog_q_func_params[Q_FUNC_PARAM_COL_NUM]
    checkpoint_base_path = prog_q_func_params[CHECKPOINT_BASE_PATH]
    checkpoint_list = prog_q_func_params[CHECKPOINTS_LIST]
    resume_training = prog_q_func_params[RESTORE_FROM_SAVE]
    # TODO: parametrize topology in prog_q_func_params
    parameter_reduction_experiment = False

    if column_number < 1:
        # Assuming input_shape is length 1 (haven't implemented convolution yet)
        if parameter_reduction_experiment:
            topology1 = [input_shape[0], 128, 64, num_actions]
            activations = [tf.nn.relu, tf.nn.relu, None]
        else:
            topology1 = [input_shape[0], 128, 128, 64, num_actions]
            activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]
        column = prog_nn.InitialColumnProgNN(topology1, activations, session, checkpoint_base_path, dtype=tf.float32)
    else:
        # Restore previous columns here
        prev_columns = []
        for i in range(column_number):
            if parameter_reduction_experiment:
                if i == 1:
                    topology1 = [input_shape[0], 128, 64, num_actions]
                elif i == 2:
                    topology1 = [input_shape[0], 64, 32, num_actions]
                elif i == 3:
                    topology1 = [input_shape[0], 32, 32, num_actions]
                elif i == 4:
                    topology1 = [input_shape[0], 16, 16, num_actions]
                elif i > 4:
                    assert(False)
                activations = [tf.nn.relu, tf.nn.relu, None]
            else:
                # TODO: SEPERATE, STORE, AND LOAD FOR EACH INDIVIDUAL COLUMN BEING RESTORED!!!
                topology1 = [input_shape[0], 128, 128, 64, num_actions]
                activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]
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

    if resume_training:
        restore_checkpoint = prog_q_func_params[RESTORE_CHECKPOINT]
        column.restore_weights(restore_checkpoint)


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

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    return session

def get_env_params():
    # Establish parameters for specific column
    env_0_params = { # Col 0 params
            'length' : 10,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 0,
            'flatten_output' : True,
            'padding' : 0,
            'num_positive_pellets' : 0,
            'num_negative_pellets' : 0,
            'num_random_obstacles' : 0,
    }

    env_1_params = {
            'length' : 30,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 1,
            'flatten_output' : True,
            'padding' : 0,
            'num_positive_pellets' : 0,
            'num_negative_pellets' : 0,
            'num_random_obstacles' : 0,
    }

    env_2_params = {
            'length' : 100,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 15,
            'flatten_output' : True,
            'padding' : 0,
            'num_positive_pellets' : 0,
            'num_negative_pellets' : 0,
            'num_random_obstacles' : 0,
    }

    env_3_params = {
            'length' : 100,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 0,
            'flatten_output' : True,
            'padding' : 0,
            'num_positive_pellets' : 0,
            'num_negative_pellets' : 0,
            'num_random_obstacles' : 30,
    }

    env_4_params = {
            'length' : 100,
            'height' : 10,
            'visible_width' : 10,
            'num_column_obstacles' : 0,
            'flatten_output' : True,
            'padding' : 0,
            'num_positive_pellets' : 5,
            'num_negative_pellets' : 0,
            'num_random_obstacles' : 30,
    }
    return [env_0_params, env_1_params, env_2_params, env_3_params, env_4_params]

def get_col_params():
    # For column number 0
    col_0_q_params = {
        Q_FUNC_PARAM_COL_NUM : 0,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [],
        RESTORE_FROM_SAVE : False,
        # RESTORE_CHECKPOINT : 153,
    }

    # For column number 1
    col_1_q_params = {
        Q_FUNC_PARAM_COL_NUM : 1,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [67], # Change to latest checkpoint for col 0
        RESTORE_FROM_SAVE : False,
        # RESTORE_CHECKPOINT : 67,
    }

    # For column number 2
    col_2_q_params = {
        Q_FUNC_PARAM_COL_NUM : 2,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [67, 153], # Change to latest checkpoints for col 0 and 1
        RESTORE_FROM_SAVE : False,
        # RESTORE_CHECKPOINT : 0,
    }

    # For column number 3
    col_3_q_params = {
        Q_FUNC_PARAM_COL_NUM : 3,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [67, 153, 68], # Change to latest checkpoints for col 0 and 1
        RESTORE_FROM_SAVE : False,
        # RESTORE_CHECKPOINT : 0,
    }

    # For column number 4
    col_4_q_params = {
        Q_FUNC_PARAM_COL_NUM : 4,
        CHECKPOINT_BASE_PATH : 'helicopter_test',
        CHECKPOINTS_LIST : [67, 153, 68, -1], # Change to latest checkpoints for col 0 and 1
        RESTORE_FROM_SAVE : False,
        # RESTORE_CHECKPOINT : 0,
    }
    return [col_0_q_params, col_1_q_params, col_2_q_params, col_3_q_params, col_4_q_params]


'''
Deals with running pnn dqn. In the current state, the run configuration is hard
coded in main. To change the column to train, specify the environment parameters
for that column, and also the q function parameters. Sample configurations for
the first three columns (0, 1, 2) can be found below, although only column 0 is
trained.
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--column', type=int, default=0)
    args = parser.parse_args()
    assert(args.column >= 0 and args.column < 5)

    # To change the environments, you must modify the parameters returned by
    # get_env_params()!!!
    # Additionally, the checkpoints to use for the frozen columns MUST be hardcoded
    # in get_col_params.
    env_parameters = get_env_params()
    col_parameters = get_col_params()
    env = helicopter.HelicopterEnv(env_parameters[args.column])
    session = get_session()
    helicopter_learn(env, session, num_timesteps=int(4e7), q_func_params=col_parameters[args.column])

if __name__ == "__main__":
    main()
