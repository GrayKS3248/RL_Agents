"""
Created on Tuesday Sep 15 10:56:33 CST 2020

@author: Grayson Schaer
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class QLearn_Agent():
    
    # Agent constructor
    # @param n_s - integer size of the state space
    # @param n_a - integer size of the action space
    # @param n_rows - Size of the state space in the x direction
    # @param n_cols - Size of the state space in the y direction
    # @param alpha - learning rate
    # @param gamma - discount ratio
    # @param epsilon - elpsilon-greedy action choice parameter
    def __init__(self, n_s, n_a, n_rows, n_cols, alpha, gamma, epsilon):
        # Sets the size of the state space
        self.n_s = n_s
        
        # Sets the size of the action space
        self.n_a = n_a
        
        # Sets the size of the x direction in the state space
        self.n_rows = n_rows
        
        # Sets the size of the y direction in the state space
        self.n_cols = n_cols
        
        # Defines the learning rate for the Q-learning agent
        self.alpha = alpha
        
        # Defines the discounting rate defined by the problem definition
        self.gamma = gamma
        
        # Defines the greedy ratio for exploration - exploitation (epsilon-greedy)
        self.epsilon = epsilon
        
        # Initializes estimated Value function (updated by TD(0))
        self.V = [0.0]*self.n_s
        
        # Initializes estimated Quality function (updated by Q-learning algorithm)
        self.Q = np.zeros((self.n_s, self.n_a))
        
        # Initializes policy (deterministic based on Quality function)
        self.p = np.zeros((self.n_s, self.n_a))
        for i in range(self.n_s):
            self.p[i][np.argmax(self.Q[i])] = 1.0
        
        # Defines the variables used in the log
        self.r_tot = 0
        self.r_tot_discount = 0
        self.curr_step = 0
        
        # Creates a log that tracks learning
        self.log = {
            's': [],
            'a': [],
            'r': [],
            'r_tot': [],
            'r_tot_discount': [],
            'avg_value': [], 
            'curr_step': [],
            }
    
    # Public getter for n_s
    # @return number of states
    def get_n_s(self):
        return self.n_s
    
    # Public getter for n_a
    # @return number of actions
    def get_n_a(self):
        return self.n_a
    
    # Public getter for alpha
    # @return learning rate
    def get_alpha(self):
        return self.alpha
    
    # Public getter for gamma
    # @return discount ratio
    def get_gamma(self):
        return self.gamma
    
    # Public getter for epsilon
    # @return epsilon-greedy ratio
    def get_epsilon(self):
        return self.epsilon
    
    # Public getter for the estimated value function
    # @return estimate value function
    def get_V(self):
        
        # Init planar representation of the Value function
        planar_V = np.zeros((self.n_rows, self.n_cols))
        
        # Assign each cell in planar_V its associated Value function output
        index = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                planar_V[i][j] = self.V[index]
                index += 1
                
        # Return the pretty version of the Value function
        return planar_V
    
    # Public getter for the estimated Quality function
    # @return estimated Quality function
    def get_Q(self):
        return self.Q
    
    # Public getter for the current policy
    # @param action_min - numerical value of the first action in the action space
    # @param action_space - float spacing between two actions (assumes uniform action spacing)
    # @return current policy
    def get_p(self, action_min, action_space):
        
        # Init planar representation of the policy function
        planar_p = np.zeros((self.n_rows, self.n_cols))
        
        # Get best policy data
        self.update_p()
        
        # Assign each cell in planar_p its associated action based on the policy, action_min, and action_space
        index = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                action_index = np.argmax(self.p[index])
                action = action_min + action_index * action_space
                planar_p[i][j] = action
                index += 1
        
        # Return the pretty representation of the policy
        return planar_p
    
    # Public getter for the learning log
    # @return learning log
    def get_log(self):
        return self.log
    
    # Updates the value function estimate based on the TD(0) algorithm
    # @param s1 - The state before an action is taken
    # @param s2 - The state after an action is taken
    # @param r - The reward returned traveling from s1 to s2
    # @return Current estimate of value function
    def update_V(self, s1, s2, r):
        # Implementation of TD(0) algorithm
        self.V[s1] = self.V[s1] + self.alpha * (r + self.gamma * self.V[s2] - self.V[s1])
        return self.V
              
    
    # Updates the quality function estimate based on the Q-learning algorithm
    # @param s1 - The state before an action is taken
    # @param a1 - The action taken in the second state
    # @param s2 - The state after an action is taken
    # @param r - The reward returned traveling from s1 to s2
    # @return Current estimate of quality function
    def update_Q(self, s1, a1, s2, r):
        # Implementation of Q-learning algorithm
        self.Q[s1][a1] = self.Q[s1][a1] + self.alpha*(r + self.gamma*self.Q[s2][np.argmax(self.Q[s2])] - self.Q[s1][a1])
        
        # Update the current Value function estimate
        self.update_V(s1, s2, r)
        
        #Update the log
        self.r_tot = self.r_tot + r
        self.r_tot_discount = self.r_tot_discount + (self.gamma ** (self.curr_step)) *  r
        avg_value = sum(self.V) / len(self.V)
        self.log['s'].append(s1)
        self.log['a'].append(a1)
        self.log['r'].append(r)
        self.log['r_tot'].append(self.r_tot)
        self.log['r_tot_discount'].append(self.r_tot_discount)
        self.log['avg_value'].append(avg_value)
        self.log['curr_step'].append(self.curr_step)
        self.curr_step = self.curr_step + 1
        
        return self.Q
    
    # Updates the policy function based on the Quality function
    # @return Current policy
    def update_p(self):
        # Update the deterministic policy function based on the new Quality function
        self.p = np.zeros((self.n_s, self.n_a))
        for i in range(self.n_s):
            self.p[i][np.argmax(self.Q[i])] = 1.0
        return self.p
    
    # Ask for an action based on the current quality fnc, epsilon-greedy parameters, and the current state
    # @param s - The current state
    # @return - The calculated epsilon-greedy action
    def get_action(self, s):
        
        # Choose a random variable to determine action
        choice = random.random()
        a = -1
        
        # If the choice variable is smaller than epsilon, take a random action in the action space
        if (choice <= self.epsilon):
            a = random.randint(0, self.n_a-1)
            
        # Otherwise, choose the best action based on the current Q function
        else:
            
            # Get the argmax a from Q
            a = np.argmax(self.Q[s])
            
            # Checks if any quality fnc outputs are the same
            identical = np.zeros(self.n_a)
            for i in range(self.n_a):
                    if (self.Q[s][i] == self.Q[s][a]):
                        identical[i] = 1
            
            # If there are identities, randomly sample from the collection of identies
            if sum(identical) != 1:
                
                # Chose a random index of one of the identical actions
                ident_a_index = random.randint(0, sum(identical) - 1)
                
                # Convert the identical array from truth array to identical action index+1 array
                identical = np.matmul(identical,np.diag(range(1,self.n_a + 1)))
                
                # Remove all indices that are zero (ie actions do not have identical outcomes)
                identical = identical[identical != 0]
                
                # Convert the identical array from identical action index+1 array to identical action index array
                identical -= 1
                
                # Select the random sampled action from the space of all identical outcome actions
                a = int(identical[ident_a_index])
        
        # Return the selected action
        return a
    
    # Visualizes the value function as a heatmap over 2D state-space
    # @param x_label - label placed on x-axis (defines columns' name)
    # @param y_label - label placed on y-axis (defines rows' name)
    # @param title - title of the plot
    # @param x_min - minimum value of the first state space dimension
    # @param y_min - minimum value of the second state space dimension
    # @param x_space - spacing between each action in the x state-space dimension
    # @param y_space - spacing between each action in the y state-space dimension
    # @param path - png save filepath and name
    # @return 2d array representation of the Value function
    def visualize_V(self, V, x_label, y_label, title, x_min, y_min, x_space, y_space, path=''):
        
        # Get the number of rows and columns in the state space
        n_rows = len(V[:][0])
        n_cols = len(V[0][:])
        
        # Create arrays used to define X and Y coords
        X = [*range(n_cols)]
        X = [n * x_space + x_min for n in X]
        Y = [*range(n_rows)]
        Y = [n * y_space + y_min for n in Y]
        grid = np.meshgrid(X, Y)
        
        # Create and save plot
        plt.clf()
        c = plt.pcolor(grid[0], grid[1], V, shading = 'auto')
        cbar = plt.colorbar(c)
        cbar.set_label('Expected Return')
        plt.title(str(title))
        plt.xlabel(str(x_label))
        plt.ylabel(str(y_label))
        save_path = str(path) + 'qlearn_value.png'
        plt.savefig(save_path, dpi = 500)
    
    # Visualizes the probability function as a heatmap over 2D state-space
    # @param x_label - label placed on x-axis (defines columns' name)
    # @param y_label - label placed on y-axis (defines rows' name)
    # @param title - title of the plot
    # @param x_min - minimum value of the first state space dimension
    # @param y_min - minimum value of the second state space dimension
    # @param x_space - spacing between each action in the x state-space dimension
    # @param y_space - spacing between each action in the y state-space dimension
    # @param action_title - name of the action space
    # @param action_min - numerical value of the first action in the action space
    # @param action_space - float spacing between two actions (assumes uniform action spacing)
    # @param path - png save filepath
    # @return 2d array representation of the policy
    def visualize_p(self, p, x_label, y_label, title, x_min, y_min, x_space, y_space, action_title, action_min, action_space, path=''):
        
        # Get the number of rows and columns in the state space
        n_rows = len(p[:][0])
        n_cols = len(p[0][:])
        
        # Create arrays used to define X and Y coords
        X = [*range(n_cols)]
        X = [n * x_space + x_min for n in X]
        Y = [*range(n_rows)]
        Y = [n * y_space + y_min for n in Y]
        grid = np.meshgrid(X, Y)
        
        # Create and save plot
        plt.clf()
        c = plt.pcolor(grid[0], grid[1], p, shading = 'auto')
        cbar = plt.colorbar(c)
        cbar.set_label(str(action_title))
        plt.title(str(title))
        plt.xlabel(str(x_label))
        plt.ylabel(str(y_label))
        save_path = str(path) + 'qlearn_policy.png'
        plt.savefig(save_path, dpi = 500)

    # Creates visualization of all parts of the log (trajectory, return, discounted return)
    # @param plot_trajectory - boolean that determines whether to plot trajectory
    # @param plot_total_reward - boolean that determines whether to plot total reward
    # @param plot_discounted_reward - boolean that determines whether to plot total dicounted reward
    # @param plot_avg_val - boolean that determines whether to plot the average value curve
    # @param initial_step - defines the first step at which the trajectory is plotted
    # @param final_stel - defines the last step at which the trajectory is plotted
    # @param path - png save filepath
    def visualize_log(self, plot_trajectory, plot_total_reward, plot_discounted_reward, plot_avg_val, initial_step, final_step, path = ''):
        
        sub_log = {
            's': self.log['s'][initial_step:final_step],
            'a': self.log['a'][initial_step:final_step],
            'r': self.log['r'][initial_step:final_step]
            }
        
        #plot the trajectory vs simulation step
        if plot_trajectory == True:
            plt.clf()
            title_str = "Qlearning Trajectory: \u03B1 = " + str(self.alpha) + ", \u03B3 = " + str(self.gamma) + ", \u03B5 = " + str(self.epsilon)
            plt.title(title_str)
            plt.plot([*range(len(sub_log['s']))], sub_log['s'])
            plt.plot([*range(len(sub_log['a']))], sub_log['a'])
            plt.plot([*range(len(sub_log['r']))], sub_log['r'])
            plt.legend(['s', 'a', 'r'])
            plt.xlabel('Learning Step')
            save_path = str(path) + 'qlearn_trajectory.png'
            plt.savefig(save_path, dpi = 500)
            plt.close()
        
        # plot the total reward vs simulation step
        if plot_total_reward == True:
            plt.clf()
            title_str = "Qlearning Learning Curve: \u03B1 = " + str(self.alpha) + ", \u03B3 = " + str(self.gamma) + ", \u03B5 = " + str(self.epsilon)
            plt.title(title_str)
            plt.plot([*range(len(self.log['r_tot']))], self.log['r_tot'])
            plt.xlabel('Learning Step')
            plt.ylabel('Total Reward')
            save_path = str(path) + 'qlearn_learning_curve.png'
            plt.savefig(save_path, dpi = 500)
            plt.close()
            
        # plot the discounted reward vs simulation step
        if plot_discounted_reward == True:
            plt.clf()
            title_str = "Qlearning Discounted Learning Curve: \u03B1 = " + str(self.alpha) + ", \u03B3 = " + str(self.gamma) + ", \u03B5 = " + str(self.epsilon)
            plt.title(title_str)
            plt.plot([*range(len(self.log['r_tot_discount']))], self.log['r_tot_discount'])
            plt.xlabel('Learning Step')
            plt.ylabel('Total Discounted Reward')
            save_path = str(path) + 'qlearn_discount_learning_curve.png'
            plt.savefig(save_path, dpi = 500)
            plt.close()
            
        # plot the average value vs simulation step
        if plot_avg_val == True:
            plt.clf()
            title_str = "Qlearning Averag Value Curve: \u03B1 = " + str(self.alpha) + ", \u03B3 = " + str(self.gamma) + ", \u03B5 = " + str(self.epsilon)
            plt.title(title_str)
            plt.plot([*range(len(self.log['avg_value']))], self.log['avg_value'])
            plt.xlabel('Learning Step')
            plt.ylabel('Value Function  Average')
            save_path = str(path) + 'qlearn_avg_val_curve.png'
            plt.savefig(save_path, dpi = 500)
            plt.close()
            