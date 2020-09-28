# RL Agents for Python
_Grayson Schaer\
09/01/2020_

# Description
Included are five reinforcement learning classes: `Policy_Iteration_Agent.py`, `Value_Iteration_Agent.py`, `SARSA_Agent.py`,  `QLearning_Agent.py`, and `DQN_Agent.py`. These classes implement the policy iteration, value iteration, SARSA, Q-learning algorithms, and Deep Q-Learning respectively. Also included are three testing environments, `Gridworld_Test_Env.py`, `Pendulum_Test_Env.py`, `Discrete_Action_Pendulum_Env.py`. `Gridworld_Test_Env.py` provides a simulation environment of the classical gridworld. The states are defined as indices in the gridworld, the actions are directions of travel, and the reward is based on hitting or missing a reward/teleport tile. `Pendulum_Test_Env.py` provides a simulation environment of a single pendulum with discretized states and actions. The states are related to the angular position and angular rate of the pendulum, the actions are related to the applied torque at the fulcrum of the pendulum, and the rewards are proportional to how close the pendulum is to standing upright. `Discrete_Action_Pendulum_Env.py` provides a simulation environment of a single pendulum with discretized action space and continuous state space. The state is the angular position and angular rate of the pendulum and the actions are related to the applied torque at the fulcrum of the pendulum. The reward is small and positive when the pendulum is to standing upright +- 3.6 deg, proportional to how close the pendulum is to standing upright, and taxed for high angular rates when standing upright. Usage of the learning classes is demonstrated in the test scripts, `Gridworld_Test.py`, `Pendulum_Test.py`, and `Discrete_Action_Pendulum_Test.py`.

# Usage

## `Policy_Iteration_Agent.py` Usage
```Python
import Gridworld_Test_Env as gte
import Policy_Iteration_Agent as pa

# Initialize the testing environment
env = gte.GridWorld()
n_episodes = 10

# Gather data from the environment
num_states = env.num_states
num_actions = env.num_actions
num_rows = int(math.sqrt(n_s)) #This defines the size of the x direction of the planar state-space
num_cols = n_rows #This defines the size of the y direction of the planar state-space

# Define other agent parameters
gamma = 0.95 # Discount rate

# Create and train agent (Training is done before simulation because this is a model based agent)
p_agent = pa.Policy_Iteration_Agent(num_states, num_actions, num_rows, num_cols, gamma, env)
p_agent.train() # This will save the training curve as a .png

# Simulate agent over n_episodes of episodes
    for curr_episode in range(n_episodes):

        # Initialize simulation environment
        s1 = env.reset()
        a1 = p_agent.get_action(s1)

        # Simulate until episode is done
        done = False
        while not done:

            # Do the selected action and get a new action
            (s2, r, done) = env.step(a1)
            a2 = p_agent.get_action(s2)

            # Update state and action
            s1 = s2
            a1 = a2

        # After an episode is done, update the logbook
        p_agent.end_episode()

    # Once an episode set is complete, update the logbook and terminate the current log
    p_agent.terminate_agent()

# Gather the results from the test in the logbook
results = p_agent.logbook
```

## `Value_Iteration_Agent.py` Usage
```Python
import Gridworld_Test_Env as gte
import Value_Iteration_Agent as va

# Initialize the testing environment
env = gte.GridWorld()
n_episodes = 10

# Gather data from the environment
num_states = env.num_states
num_actions = env.num_actions
num_rows = int(math.sqrt(n_s)) #This defines the size of the x direction of the planar state-space
num_cols = n_rows #This defines the size of the y direction of the planar state-space

# Define other agent parameters
gamma = 0.95 # Discount rate

# Create and train agent (Training is done before simulation because this is a model based agent)
v_agent = va.Value_Iteration_Agent(num_states, num_actions, num_rows, num_cols, gamma, env)
v_agent.train() # This will save the training curve as a .png

# Simulate agent over n_episodes of episodes
    for curr_episode in range(n_episodes):

        # Initialize simulation environment
        s1 = env.reset()
        a1 = v_agent.get_action(s1)

        # Simulate until episode is done
        done = False
        while not done:

            # Do the selected action and get a new action
            (s2, r, done) = env.step(a1)
            a2 = v_agent.get_action(s2)

            # Update state and action
            s1 = s2
            a1 = a2

        # After an episode is done, update the logbook
        v_agent.end_episode()

    # Once an episode set is complete, update the logbook and terminate the current log
    v_agent.terminate_agent()

# Gather the results from the test in the logbook
results = v_agent.logbook
```
## `SARSA_Agent.py` Usage
```Python
import Gridworld_Test_Env as gte
import SARSA_Agent as sa

# Initialize the testing environment
env = gte.GridWorld()
n_episodes = 10

# Gather data from the environment
num_states = env.num_states
num_actions = env.num_actions
num_rows = int(math.sqrt(n_s)) #This defines the size of the x direction of the planar state-space
num_cols = n_rows #This defines the size of the y direction of the planar state-space

# Define other agent parameters
alpha = 0.50 # Learning rate
gamma = 0.95 # Discount rate
epsilon = 0.10 # Exploration rate

# Create agent
s_agent = sa.SARSA_Agent(num_states, num_actions, num_rows, num_cols, alpha, gamma, epsilon, env)

# Train agent over n_episodes of episodes
for curr_episode in range(n_episodes):

    # Initialize simulation
    s1 = env.reset()
    a1 = random.randint(0, env.num_actions - 1)

    # Simulate until episode is done
    done = False
    while not done:

        # Do the selected action and get a new action
        (s2, r, done) = env.step(a1)
        a2 = s_agent.get_action(s2)

        # Update agent's Q estimate
        s_agent.update_Q(s1, a1, s2, a2, r)

        # Update state and action
        s1 = s2
        a1 = a2

    # After an episode is done, update the logbook
    s_agent.end_episode()

# Once an episode set is complete, update the logbook and terminate the current log
s_agent.terminate_agent()

# Gather the results from the test in the logbook
results = s_agent.logbook
```
## `QLearning_Agent.py` Usage
```Python
import Gridworld_Test_Env as gte
import QLearning_Agent as qa

# Initialize the testing environment
env = gte.GridWorld()
n_episodes = 10

# Gather data from the environment
num_states = env.num_states
num_actions = env.num_actions
num_rows = int(math.sqrt(n_s)) #This defines the size of the x direction of the planar state-space
num_cols = n_rows #This defines the size of the y direction of the planar state-space

# Define other agent parameters
alpha = 0.50 # Learning rate
gamma = 0.95 # Discount rate
epsilon = 0.10 # Exploration rate

# Create agent
q_agent = qa.QLearn_Agent(num_states, num_actions, num_rows, num_cols, alpha, gamma, epsilon, env)

# Train agent over n_episodes of episodes
for curr_episode in range(n_episodes):

    # Initialize simulation
    s1 = env.reset()

    # Simulate until episode is done
    done = False
    while not done:

        # Do the selected action and get a new action
        a1 = q_agent.get_action(s1)
        (s2, r, done) = env.step(a1)

        # Update agent's Q estimate
        q_agent.update_Q(s1, a1, s2, r)

        # Update state and action
        s1 = s2

    # After an episode is done, update the logbook
    q_agent.end_episode()

# Once an episode set is complete, update the logbook and terminate the current log
q_agent.terminate_agent()

# Gather the results from the test in the logbook
results = q_agent.logbook
```
## `DQN_Agent.py` Usage
```Python
import Discrete_Action_Pendulum_Env as dape
import DQN_Agent as dqn
import numpy as np
import math
import torch

# Defines what a set of episodes is
def run_set(curr_set, n_sets, n_episodes, env, agent):

    # Start Training
    print("Training DQN agent " + str(curr_set) + " of " + str(n_sets-1) + "...")  

    # Train agent over n_episodes of episodes
    for curr_episode in range(n_episodes):

        # Initialize simulation
        s1 = env.reset()
        agent.add_state_to_sequence(s1)

        # Simulate until episode is done
        done = False
        while not done:

            # With probability e select a random action a1, otherwise select a1 = argmax_a Q(s1, a; theta)
            a1 = agent.get_action(s1)

            # Execute action a1 in emulator and observer reward r and next state s2
            (s2, r, done) = env.step(a1)

            # Update state sequence buffer, store experience in data_set
            agent.add_experience_to_data_set(s1, a1, r, s2)

            # Sample a random minibatch from the data_set, define targets, perform gradient descent, and ocasionally update Q_Target_Network
            agent.learn()

            # Update state and action
            s1 = s2

        # After an episode is done, update the logbook
        agent.end_episode()

    # Onces an episode set is complete, update the logbook and terminate the current log
    agent.terminate_agent()

    return agent


if __name__ == '__main__':
    # Environment
    env = dape.Pendulum()
    num_actions = env.num_actions
    state_dimension = 2

    # Simulation parameters
    n_sets = 100
    n_episodes = 1005                         
    n_steps = env.max_num_steps

    # Agent parameters
    max_data_set_size = 50000                 
    start_data_set_size = 500              
    sequence_size = 1                     
    minibatch_size = 32                    
    num_hidden_layers = 2             
    num_neurons_in_layer = 64               
    clone_interval = 1000                 
    alpha = 0.0025                           
    gamma = 0.95                        
    epsilon_start = 1.00                  
    epsilon_end = 0.10                   
    epsilon_depreciation_factor = 0.99977

    # Create agent
    agent = dqn.DQN_Agent(num_actions, state_dimension, max_data_set_size, start_data_set_size, sequence_size,
                          minibatch_size, num_hidden_layers, num_neurons_in_layer, clone_interval,
                          alpha, gamma, epsilon_start, epsilon_end, epsilon_depreciation_factor)  

    # Run the defined number of sets and update the average
    for curr_set in range(n_sets):

        # Run a set of episodes
        agent = run_set(curr_set, n_sets, n_episodes, env, agent)

```
# License
MIT License

Copyright (c) 2020 Grayson Schaer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
