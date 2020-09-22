import random
import Pendulum_Test_Env as pte
import QLearning_Agent as qa
import SARSA_Agent as sa
import time
import math
import matplotlib.pyplot as plt
import numpy as np

def run_set(curr_set, n_sets, n_episodes, env, s_agent, q_agent):
            
    # Start Training
    print("Training SARSA agent " + str(curr_set) + " of " + str(n_sets-1) + "...")  
      
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

    # Onces an episode set is complete, update the logbook and terminate the current log
    s_agent.terminate_agent()
    
    # Start Training
    print("Training Q-Learning agent " + str(curr_set) + " of " + str(n_sets-1) + "...")
      
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
    
    # Onces an episode set is complete, update the logbook and terminate the current log
    q_agent.terminate_agent()
    
    return s_agent, q_agent

if __name__ == '__main__':
    # Environment
    env = pte.Pendulum()
    n_s = env.num_states
    n_a = env.num_actions
    
    # Simulation parameters
    n_sets = 2
    n_episodes = 2
    n_steps = env.max_num_steps
    
    # Agent parameters
    alpha = 0.50
    gamma = 0.95
    epsilon = 0.10
    n_rows = int(math.sqrt(n_s))
    n_cols = n_rows
        
    # Plotting parameters
    X = [*range(n_cols)]
    Y = [*range(n_rows)]
    grid = np.meshgrid(X, Y)
    
    # Create agents
    s_agent = sa.SARSA_Agent(n_s, n_a, n_rows, n_cols, alpha, gamma, epsilon)  
    q_agent = qa.QLearn_Agent(n_s, n_a, n_rows, n_cols, alpha, gamma, epsilon)  
    
    # Run the defined number of sets and update the average
    start = time.time()
    for curr_set in range(n_sets):
        # Run a set of episodes
        s_agent, q_agent = run_set(curr_set, n_sets, n_episodes, env, s_agent, q_agent)
    
    elapsed = time.time() - start
    print("Simulation took:", f'{elapsed:.3f}', "seconds.")
    
    # plot results
    print("Plotting training and trajectory data...")
    
    # Convert trajectory data to real data (this is done outside of the agent class to ensure agent generalizability)
    sarsa_s = np.asarray(s_agent.logbook['best_s'])
    sarsa_i = sarsa_s // env.n_thetadot
    sarsa_j = env.n_thetadot * ((sarsa_s / env.n_thetadot) - (sarsa_s // env.n_thetadot))
    sarsa_theta = ((2 * np.pi * sarsa_i) / env.n_theta) - np.pi
    sarsa_theta_dot = ((2 * env.max_thetadot * sarsa_j) / env.n_thetadot) - env.max_thetadot
    
    qlrn_s = np.asarray(q_agent.logbook['best_s'])
    qlrn_i = qlrn_s // env.n_thetadot
    qlrn_j = env.n_thetadot * ((qlrn_s / env.n_thetadot) - (qlrn_s // env.n_thetadot))
    qlrn_theta = ((2 * np.pi * qlrn_i) / env.n_theta) - np.pi
    qlrn_theta_dot = ((2 * env.max_thetadot * qlrn_j) / env.n_thetadot) - env.max_thetadot
    
    # Plot Value functions
    plt.clf()
    c = plt.pcolor(grid[0], grid[1], q_agent.logbook['avg_V'], shading = 'auto')
    cbar = plt.colorbar(c)
    cbar.set_label('Expected Return')
    title_str = "Q-Learning Value Function: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.savefig('Example_Results/Pendulum/qln_val_fnc.png', dpi = 200)
    
    plt.clf()
    c = plt.pcolor(grid[0], grid[1], s_agent.logbook['avg_V'], shading = 'auto')
    cbar = plt.colorbar(c)
    cbar.set_label('Expected Return')
    title_str = "SARSA Value Function: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.savefig('Example_Results/Pendulum/sar_val_fnc.png', dpi = 200)
        
    # Plot policies
    plt.clf()
    c = plt.pcolor(grid[0], grid[1], q_agent.logbook['avg_p'], shading = 'auto')
    cbar = plt.colorbar(c)
    cbar.set_label('Expected Return')
    title_str = "Q-Learning Policy: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.savefig('Example_Results/Pendulum/qln_pol.png', dpi = 200)
    
    plt.clf()
    c = plt.pcolor(grid[0], grid[1], s_agent.logbook['avg_p'], shading = 'auto')
    cbar = plt.colorbar(c)
    cbar.set_label('Expected Return')
    title_str = "SARSA Policy: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.savefig('Example_Results/Pendulum/sar_pol.png', dpi = 200)
    
    # Plot average of Value function curves
    plt.clf()
    title_str = "Value Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(q_agent.logbook['avg_val_avg']))], q_agent.logbook['avg_val_avg'])
    plt.plot([*range(len(s_agent.logbook['avg_val_avg']))], s_agent.logbook['avg_val_avg'])
    plt.legend(["QLN", "SAR"])
    plt.xlabel('Simulation Step')
    plt.ylabel('Average of Value Function')
    plt.savefig('Example_Results/Pendulum/val_cur.png', dpi = 200)
    plt.close()
    
    # Plot reward curves    
    plt.clf()
    title_str = "Reward Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(q_agent.logbook['r_tot_avg']))], q_agent.logbook['r_tot_avg'])
    plt.plot([*range(len(s_agent.logbook['r_tot_avg']))], s_agent.logbook['r_tot_avg'])
    plt.legend(["QLN", "SAR"])
    plt.xlabel('Simulation Step')
    plt.ylabel('Total Reward')
    plt.savefig('Example_Results/Pendulum/rwd_cur.png', dpi = 200)
    plt.close()
    
    # Plot learning curves
    plt.clf()
    title_str = "Learning Curve: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(q_agent.logbook['r_tot_discount_avg']))], q_agent.logbook['r_tot_discount_avg'])
    plt.plot([*range(len(s_agent.logbook['r_tot_discount_avg']))], s_agent.logbook['r_tot_discount_avg'])
    plt.legend(["QLN", "SAR"])
    plt.xlabel('Simulation Step')
    plt.ylabel('Total Discounted Reward')
    plt.savefig('Example_Results/Pendulum/lrn_cur.png', dpi = 200)
    plt.close()
    
    # Plot trajectories
    plt.clf()
    title_str = "Q-Learning Trajectory: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(q_agent.logbook['best_s']))], q_agent.logbook['best_s'])
    plt.plot([*range(len(q_agent.logbook['best_a']))], q_agent.logbook['best_a'])
    plt.plot([*range(len(q_agent.logbook['best_r']))], q_agent.logbook['best_r'])
    plt.legend(["s", "a", "r"])
    plt.xlabel('Simulation Step')
    plt.savefig('Example_Results/Pendulum/qln_trj.png', dpi = 200)
    plt.close()

    plt.clf()
    title_str = "Q-Learning True Trajectory: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(qlrn_theta))], qlrn_theta)
    plt.plot([*range(len(qlrn_theta_dot))], qlrn_theta_dot)
    plt.legend(["\u03B8", "\u03B8'"])
    plt.xlabel('Simulation Step')
    plt.savefig('Example_Results/Pendulum/qln_tru_trj.png', dpi = 200)
    plt.close()    

    plt.clf()
    title_str = "SARSA Trajectory: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(s_agent.logbook['best_s']))], s_agent.logbook['best_s'])
    plt.plot([*range(len(s_agent.logbook['best_a']))], s_agent.logbook['best_a'])
    plt.plot([*range(len(s_agent.logbook['best_r']))], s_agent.logbook['best_r'])
    plt.legend(["s", "a", "r"])
    plt.xlabel('Simulation Step')
    plt.savefig('Example_Results/Pendulum/sar_trj.png', dpi = 200)
    plt.close()
    
    plt.clf()
    title_str = "SARSA True Trajectory: \u03B1 = " + str(alpha) + ", \u03B3 = " + str(gamma) + ", \u03B5 = " + str(epsilon)
    plt.title(title_str)
    plt.plot([*range(len(sarsa_theta))], sarsa_theta)
    plt.plot([*range(len(sarsa_theta_dot))], sarsa_theta_dot)
    plt.legend(["\u03B8", "\u03B8'"])
    plt.xlabel('Simulation Step')
    plt.savefig('Example_Results/Pendulum/sar_tru_trj.png', dpi = 200)
    plt.close()  
    
    print("Plotting complete!")
