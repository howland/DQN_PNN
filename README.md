# DQN_PNN
Deep-Q Networks using a Progressive Neural Network on a toy helicopter environment.

## Usage
There are three steps for training this implementation PNN-DQN on the helicopter environment. Note that columns in the PNN are ZERO indexed, i.e. the first column is referred to as column 0, the second as column 1, and so on.
1. The base policy must be trained (the first column in the PNN)
2. The desired weight checkpoint for the first column (0) must be hardcoded into `col_1_q_params` found in `run_prog_dqn_helicopter:get_col_params`
3. The second column can be trained (column 1). The desired weight checkpoint then must be hardcoded into `col_2_q_params`
4. Repeat for higher order columns

To train a column: `python run_prog_dqn_helicopter.py --column=0`

## Results
We were able to demonstrate that vanilla DQN becomes unstable with more complex versions of the helicopter environment (with minimal hyperparameter tuning and the default helicopter state space representation), and that PNN-DQN trained on simpler subtasks can improve final performance and stability while learning.

![Performance on Environment 2](/images/env2.png)

For complete results, see the report.

## Credit
The initial PNN implementation comes from synpon's prog_nn implementation, written by odellus. The license can be found at prog_dqn/prog_nn_LICENSE.
* https://github.com/synpon/prog_nn/

The initial DQN implementation comes from berkeleydeeprlcourse's DQN implementation. The license can be found at prog_dqn/dqn_LICENSE.
* https://github.com/berkeleydeeprlcourse/homework

## Future Work:
* The option for Convolutional layers needs to be implemented in the PNN. Until Conv layers are added, the DQN-PNN implementation will not work well for visual observations. This will require elaborating the specified topology for the networks, but should not be too difficult.
