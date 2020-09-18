import random
import Pendulum_Test_Env as pte
import QLearning_Agent as qa
import SARSA_Agent as sa
import numpy as np
import matplotlib.pyplot as plt
import time

def run_set(curr_set, n_states, n_actions, state_space_dimension, alpha, gamma, epsilon, n_episodes, n_sets):
    # Create environment
    #
    #   By default, both the state space (theta, thetadot) and the action space
    #   (tau) are discretized with 31 grid points in each dimension, for a total
    #   of 31 x 31 states and 31 actions.
    #
    #   You can change the number of grid points as follows (for example):
    #
    #       env = discrete_pendulum.Pendulum(n_theta=11, n_thetadot=51, n_tau=21)
    env = pte.Pendulum(n_theta = n_states, n_thetadot = n_states, n_tau = n_actions)

    # Start Training
    print("Training SARSA agent " + str(curr_set) + " of " + str(n_sets-1) + "...")
    
    # Build agent
    s_agent = sa.SARSA_Agent(n_states ** state_space_dimension, n_actions, n_states, n_states, alpha, gamma, epsilon)    
      
    # Train agent over n_episodes of episodes
    for curr_episode in range(n_episodes):
        
        # Initialize simulation
        s1 = env.reset()
        a1 = random.randint(0, n_actions - 1)
        
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

    # Start Training
    print("Training Q-Learning agent " + str(curr_set) + " of " + str(n_sets-1) + "...")
    
    # Build agent
    q_agent = qa.QLearn_Agent(n_states ** state_space_dimension, n_actions, n_states, n_states, alpha, gamma, epsilon)    
      
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
    
    return s_agent, q_agent

if __name__ == '__main__':
    # Simulation parameters
    n_sets = 50
    n_episodes = 5000
    n_steps = 100
    
    # Environment parameters
    n_states = 31
    n_actions = 31
    state_space_dimension = 2
    
    # Agent parameters
    alpha = 0.40
    gamma = 0.95
    epsilon = 0.05
    
    # Init SARSA averages
    s_V_avg = 0
    s_p_avg = 0
    s_avg_val_avg = 0
    s_learning_avg = 0
    s_discount_avg = 0
    
    # init qlearn averages
    q_V_avg = 0
    q_p_avg = 0
    q_avg_val_avg = 0
    q_learning_avg = 0
    q_discount_avg = 0
    
    # Run the defined number of sets and update the average
    start = time.time()
    for i in range(n_sets):
        # Run a set of episodes
        s_agent, q_agent = run_set(i, n_states, n_actions, state_space_dimension, alpha, gamma, epsilon, n_episodes, n_sets)
        
        # Gather SARSA results from episode set
        s_log_curr = s_agent.get_log()
        s_V_curr = s_agent.get_V()
        s_p_curr = s_agent.get_p(-5.0, 10.0/(n_actions-1))
        
        # Gather qlearn results from episode set
        q_log_curr = q_agent.get_log()
        q_V_curr = q_agent.get_V()
        q_p_curr = q_agent.get_p(-5.0, 10.0/(n_actions-1))
        
        # Update SARSA averages
        s_V_avg = np.add(s_V_avg, (1/n_sets) * s_V_curr)
        s_p_avg = np.add(s_p_avg, (1/n_sets) * s_p_curr)
        s_avg_val_avg = np.add(s_avg_val_avg, (1/n_sets) * np.asarray(s_log_curr['avg_value']))
        s_learning_avg = np.add(s_learning_avg, (1/n_sets) * np.asarray(s_log_curr['r_tot']))
        s_discount_avg = np.add(s_discount_avg, (1/n_sets) * np.asarray(s_log_curr['r_tot_discount']))
        
        # Update qlearn averages
        q_V_avg = np.add(q_V_avg, (1/n_sets) * q_V_curr)
        q_p_avg = np.add(q_p_avg, (1/n_sets) * q_p_curr)
        q_avg_val_avg = np.add(q_avg_val_avg, (1/n_sets) * np.asarray(q_log_curr['avg_value']))
        q_learning_avg = np.add(q_learning_avg, (1/n_sets) * np.asarray(q_log_curr['r_tot']))
        q_discount_avg = np.add(q_discount_avg, (1/n_sets) * np.asarray(q_log_curr['r_tot_discount']))
        
    elapsed = time.time() - start
    print("Training set took:", f'{elapsed:.3f}', "seconds.")
    
    # plot results
    print("Plotting training and trajectory data...")
    s_agent.visualize_log(True, False, False, False, int((n_episodes - 1)*n_steps), int((n_episodes)*n_steps - 1))
    s_agent.visualize_V(s_V_avg, 'Theta [deg]', 'Theta_dot [deg/s]', 'SARSA Value Estimation', -180.0, -859.44, 360.0/(n_states-1), 1718.89/(n_states-1))
    s_agent.visualize_p(s_p_avg, 'Theta [deg]', 'Theta_dot [deg/s]', 'SARSA Policy', -180.0, -859.44, 360.0/(n_states-1), 1718.89/(n_states-1), 'Torque', -5.0, 10.0/(n_actions-1))

    plt.clf()
    title_str = "SARSA Discounted Learning Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(s_discount_avg))], s_discount_avg)
    plt.xlabel('Learning Step')
    plt.ylabel('Total Discounted Reward')
    plt.savefig('sarsa_discount_learning_curve.png', dpi = 500)
    plt.close()
    
    plt.clf()
    title_str = "SARSA Learning Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(s_learning_avg))], s_learning_avg)
    plt.xlabel('Learning Step')
    plt.ylabel('Total Reward')
    plt.savefig('sarsa_learning_curve.png', dpi = 500)
    plt.close()
    
    plt.clf()
    title_str = "SARSA Average Value Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(s_avg_val_avg))], s_avg_val_avg)
    plt.xlabel('Learning Step')
    plt.ylabel('Average of Value Function')
    plt.savefig('sarsa_avg_val_curve.png', dpi = 500)
    plt.close()
            
    
    q_agent.visualize_log(True, False, False, False, int((n_episodes - 1)*n_steps), int((n_episodes)*n_steps - 1))
    q_agent.visualize_V(q_V_avg, 'Theta [deg]', 'Theta_dot [deg/s]', 'Q-Learning Value Estimation', -180.0, -859.44, 360.0/(n_states-1), 1718.89/(n_states-1))
    q_agent.visualize_p(q_p_avg, 'Theta [deg]', 'Theta_dot [deg/s]', 'Q-Learning Policy', -180.0, -859.44, 360.0/(n_states-1), 1718.89/(n_states-1), 'Torque', -5.0, 10.0/(n_actions-1))
    
    plt.clf()
    title_str = "Q-Learning Discounted Learning Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(q_discount_avg))], q_discount_avg)
    plt.xlabel('Learning Step')
    plt.ylabel('Total Discounted Reward')
    plt.savefig('qlearn_discount_learning_curve.png', dpi = 500)
    plt.close()
    
    plt.clf()
    title_str = "Q-Learning Learning Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(q_learning_avg))], q_learning_avg)
    plt.xlabel('Learning Step')
    plt.ylabel('Total Reward')
    plt.savefig('qlearn_learning_curve.png', dpi = 500)
    plt.close()
    
    plt.clf()
    title_str = "Q-Learning Average Value Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(q_avg_val_avg))], q_avg_val_avg)
    plt.xlabel('Learning Step')
    plt.ylabel('Average of Value Function')
    plt.savefig('qlearn_avg_val_curve.png', dpi = 500)
    plt.close()
    
    print("Plotting complete!")
