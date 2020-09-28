import Discrete_Action_Pendulum_Env as dape 
import DQN_Agent as dqn
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import pickle

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
    n_episodes = 1005                          # Best = 505
    n_steps = env.max_num_steps
    
    # Agent parameters
    max_data_set_size = 50000                 # Best = 50000
    start_data_set_size = 500                 # Best = 500
    sequence_size = 1                         # Best = 1 (likely because we already have state derivative data)
    minibatch_size = 32                       # Best = 32
    num_hidden_layers = 2                     # Constrained = 2
    num_neurons_in_layer = 64                 # Constrained = 64
    clone_interval = 1000                     # Best = 1000
    alpha = 0.0025                            # Best = 0.0025
    gamma = 0.95                              # Constrained = 0.95
    epsilon_start = 1.00                      # Best = 1.00
    epsilon_end = 0.10                        # Best = 0.10
    epsilon_depreciation_factor = 0.99977    # Best = 0.99977
    
    # Create agent
    agent = dqn.DQN_Agent(num_actions, state_dimension, max_data_set_size, start_data_set_size, sequence_size, 
                          minibatch_size, num_hidden_layers, num_neurons_in_layer, clone_interval, 
                          alpha, gamma, epsilon_start, epsilon_end, epsilon_depreciation_factor)  
    
    # Run the defined number of sets and update the average
    start = time.time()
    for curr_set in range(n_sets):
        # Run a set of episodes
        agent = run_set(curr_set, n_sets, n_episodes, env, agent)
    
    elapsed = time.time() - start
    print("Simulation took:", f'{elapsed:.3f}', "seconds.")
    
    # plot results
    print("Plotting training and trajectory data...")
    start = time.time()
    
    # Calculate locations of cloning and terminal epsilon
    last_data_point = len(agent.logbook['r_tot_discount_avg'])-1
    num_times_cloned = int(float(last_data_point) // float(clone_interval))
    cloning_points = np.linspace(clone_interval, last_data_point, num_times_cloned)
    final_exploration_frame = math.log(epsilon_end) // math.log(epsilon_depreciation_factor)
    
    # Get best agent, define its policy and value function, make gif
    best_Q = agent.logbook['best_Q']
    policy = lambda s : int(torch.argmax(best_Q.forward(torch.Tensor(s))))
    value = lambda s : float(torch.max(best_Q.forward(torch.Tensor(s))).detach())
    env.video(policy, filename='Results/dqn_trj.gif')
    
    # Make spaces to plot value function and policy
    theta = np.linspace(-2*np.pi, 2*np.pi, 1000)
    theta_dot = np.linspace(-1*env.max_thetadot, env.max_thetadot, 1000)
    X,Y = np.meshgrid(theta, theta_dot)
    policy_out = np.zeros(np.shape(X))
    value_out = np.zeros(np.shape(X))
    for i in range(len(X)):
        for j in range(len(Y)):
            policy_out[i][j] = env._a_to_u(policy((X[i][j],Y[i][j])))
            value_out[i][j] = value((X[i][j],Y[i][j]))
    
    # Save all useful variables
    outputs = {
        'n_sets' : n_sets, 
        'n_episodes' : n_episodes, 
        'env' : env, 
        'best_agent' : agent,
        }
    with open('Results/outputs', 'wb') as f:
        pickle.dump(outputs, f)
    
    # Plot Policy
    plt.clf()
    title_str = "DQN Policy: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.xlabel("θ")
    plt.ylabel("θ'")
    c = plt.pcolor(X, Y, policy_out, shading = 'auto')
    cbar = plt.colorbar(c)
    cbar.set_label('Torque')
    plt.savefig('Results/dqn_pol_fnc.png', dpi = 200)
    plt.close()
        
    # Plot Policy
    plt.clf()
    title_str = "DQN Value Fucntion: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.xlabel("θ")
    plt.ylabel("θ'")
    c = plt.pcolor(X, Y, value_out, shading = 'auto')
    cbar = plt.colorbar(c)
    cbar.set_label('Expected Return')
    plt.savefig('Results/dqn_val_fnc.png', dpi = 200)
    plt.close()
    
    # Plot Trajectory
    plt.clf()
    title_str = "DQN Trajectory: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.plot([*range(len(agent.logbook['best_s']))], agent.logbook['best_s'])
    plt.axhline(env.max_theta_for_upright, c='k', linestyle=':', linewidth=0.5, label=None)
    plt.axhline(-1 * env.max_theta_for_upright, c ='k', linestyle=':', linewidth=0.5, label=None)
    plt.legend(["θ", "θ'", "Target θ"])
    plt.xlabel('Simulation Step')
    plt.ylabel('[Rad], [Rad/S]')
    plt.savefig('Results/dqn_trj.png', dpi = 200)
    plt.close()
    
    # Plot learning curve
    plt.clf()
    title_str = "DQN Learning Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.plot([*range(len(agent.logbook['r_tot_discount_avg']))], agent.logbook['r_tot_discount_avg'], label="Discount Reward")
    for cloning_point in cloning_points:
        if cloning_point in [clone_interval]:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label="Cloning")
        else:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label=None)
    plt.axvline(final_exploration_frame, c='k', linestyle=':', linewidth=2, label="ε Stable")
    plt.legend()
    plt.xlabel('Simulation Step')
    plt.ylabel('Total Discounted Reward')
    plt.savefig('Results/dqn_lrn_cur.png', dpi = 200)
    plt.close()
    
    # Plot reward curve    
    plt.clf()
    title_str = "DQN Reward Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.plot([*range(len(agent.logbook['r_tot_avg']))], agent.logbook['r_tot_avg'], label="Total Reward")
    for cloning_point in cloning_points:
        if cloning_point in [clone_interval]:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label="Clone")
        else:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label=None)
    plt.axvline(final_exploration_frame, c='k', linestyle=':', linewidth=2, label="ε Stable")
    plt.legend()
    plt.xlabel('Simulation Step')
    plt.ylabel('Total Reward')
    plt.savefig('Results/dqn_rwd_cur.png', dpi = 200)
    
    # plot loss curve
    plt.clf()
    title_str = "DQN Loss Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon_start) + " → " + str(epsilon_end)
    plt.title(title_str)
    plt.plot([*range(len(agent.logbook['loss_avg']))], agent.logbook['loss_avg'], label="Loss")
    for cloning_point in cloning_points:
        if cloning_point in [clone_interval]:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label="Clone")
        else:
            plt.axvline(cloning_point, c='r', linestyle=':', linewidth=0.5, label=None)
    plt.axvline(final_exploration_frame, c='k', linestyle=':', linewidth=2, label="ε Stable")
    plt.legend()
    plt.xlabel('Simulation Step')
    plt.ylabel('Loss')
    plt.savefig('Results/dqn_los_cur.png', dpi = 200)
    plt.close()
    
    elapsed = time.time() - start
    print("Plotting took:", f'{elapsed:.3f}', "seconds.")