"""
Created on Tuesday Sep 15 10:56:33 CST 2020

@author: Grayson Schaer
"""

import numpy as np
import random

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
        
        # Defines the learning rate for the agent
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
        
        # Defines the variables used in the logbook
        self.entry_number = 1
        self.highest_r_tot = -1
        self.last_episode_last_index = -1
        
        # Creates a log that tracks learning during a single episode set
        # @entry s - the state at each iteration step
        # @entry a - the action taken at each iteration step
        # @entry r - the reward given at each iteration step
        # @entry r_tot - the sum of all rewards given up to the current iteration step
        # @entry r_tot_discount - the sum of all the discounted rewards given up to the current iteration step
        # @entry avg_value - the average of the value function at each iteration step
        # @curr_step - the iteration step at which all other values are measured
        self.log = {
            's': [],
            'a': [],
            'r': [],
            'r_tot': [],
            'r_tot_discount': [],
            'avg_value': [], 
            }
        
        # Creates a book that tracks results from previous episode sets
        # @entry logs - records each log for each set of episodes run by an agent
        # @entry avg_Q - tracks the average Quality function generated by each agent
        # @entry avg_V - tracks the average Value function generated by each agent
        # @entry avg_p - tracks the average policy function generated by each agent
        # @entry alpha - learning rate of episode set
        # @entry gamma - discount factor of episode set
        # @entry epsilon - exploration rate of episode set
        self.logbook = {
            'best_s': [],
            'best_a': [],
            'best_r': [],
            'r_tot_avg': 0,
            'r_tot_discount_avg': 0,
            'avg_val_avg': 0,
            'avg_Q': 0,
            'avg_V': 0,
            'avg_p': 0,
            'alpha': [],
            'gamma': [],
            'epsilon': [],
            }
    
    # Public getter for the estimated value function
    # @return estimate value function in planar state space form
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
    
    # Public getter for the current policy
    # @return current policy in planar state space form
    def get_p(self):
        
        # Init planar representation of the policy function
        planar_p = np.zeros((self.n_rows, self.n_cols))
        
        # Get best policy data
        self.update_p()
        
        # Assign each cell in planar_p its associated action based on the policy, action_min, and action_space
        index = 0
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                action_index = np.argmax(self.p[index])
                planar_p[i][j] = action_index
                index += 1
        
        # Return the pretty representation of the policy
        return planar_p
   
    # Updates the best trajectory at the end of each episode
    def end_episode(self):
        # Update the best trajectory log
        if self.log['r_tot'][-1] > self.highest_r_tot:
            self.highest_r_tot = self.log['r_tot'][-1]
            self.logbook['best_s'] = self.log['s'][(self.last_episode_last_index + 1):(len(self.log['r_tot']) - 1)]
            self.logbook['best_a'] = self.log['a'][(self.last_episode_last_index + 1):(len(self.log['r_tot']) - 1)]
            self.logbook['best_r'] = self.log['r'][(self.last_episode_last_index + 1):(len(self.log['r_tot']) - 1)]
            
        self.last_episode_last_index = len(self.log['r_tot']) - 1 
   
    # Archives the log from the current episode to the log book. Resets agent to initial conditions
    # @param alpha - learning rate
    # @param gamma - discount ratio
    # @param epsilon - elpsilon-greedy action choice parameter
    def terminate_agent(self):
        # Update the average learning curves
        n = self.entry_number
        self.entry_number += 1
        r_tot_avg = (1/n) * ((n - 1) * self.logbook['r_tot_avg'] + np.asarray(self.log['r_tot']))
        r_tot_discount_avg = (1/n) * ((n - 1) * self.logbook['r_tot_discount_avg'] + np.asarray(self.log['r_tot_discount']))
        avg_val_avg = (1/n) * ((n - 1) * self.logbook['avg_val_avg'] + np.asarray(self.log['avg_value']))
        avg_Q = (1/n) * ((n - 1) * self.logbook['avg_Q'] + self.Q)
        avg_V = (1/n) * ((n - 1) * self.logbook['avg_V'] + self.get_V())
        avg_p = (1/n) * ((n - 1) * self.logbook['avg_p'] + self.get_p())
        
        # In the case of the first set, the average value is just the current value
        if n==1:
            avg_Q = self.Q
            avg_V = self.get_V()
            avg_p = self.get_p()
            r_tot_avg = self.log['r_tot']
            r_tot_discount_avg = self.log['r_tot_discount'] 
            avg_val_avg = self.log['avg_value']
            
        # Input the average function values
        self.logbook['r_tot_avg'] = r_tot_avg
        self.logbook['r_tot_discount_avg'] = r_tot_discount_avg
        self.logbook['avg_val_avg'] = avg_val_avg       
        self.logbook['avg_Q'] = avg_Q
        self.logbook['avg_V'] = avg_V
        self.logbook['avg_p'] = avg_p
        
        # Input the agent parameters
        self.logbook['alpha'].append(self.alpha)
        self.logbook['gamma'].append(self.gamma)
        self.logbook['epsilon'].append(self.epsilon)
        
        # Reset the log
        self.log = {
            's': [],
            'a': [],
            'r': [],
            'r_tot': [],
            'r_tot_discount': [],
            'avg_value': [], 
            }
        
        # Initializes estimated Value function (updated by TD(0))
        self.V = [0.0]*self.n_s
        
        # Initializes estimated Quality function (updated by SARSA algorithm)
        self.Q = np.zeros((self.n_s, self.n_a))
        
        # Initializes policy (deterministic based on Quality function)
        self.p = np.zeros((self.n_s, self.n_a))
        for i in range(self.n_s):
            self.p[i][np.argmax(self.Q[i])] = 1.0
            
        # Defines the variables used in the log
        self.r_tot = 0
        self.r_tot_discount = 0
        self.curr_step = 0
        
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
    
    # Updates the value function estimate based on the TD(0) algorithm
    # @param s1 - The state before an action is taken
    # @param s2 - The state after an action is taken
    # @param r - The reward returned traveling from s1 to s2
    # @return Current estimate of value function
    def update_V(self, s1, s2, r):
        # Implementation of TD(0) algorithm
        self.V[s1] = self.V[s1] + self.alpha * (r + self.gamma * self.V[s2] - self.V[s1])
        return self.V
    
    # Ask for an action based on the current quality fnc, epsilon-greedy parameters, and the current state
    # @param s - The current state
    # @return - The calculated epsilon-greedy action
    def get_action(self, s):
        
        # Choose a random variable to determine action
        choice = random.random()
        
        # Init action to impossible value
        # the variable a necesarily must be updated, so it is placed outside of the if loop so that it can be returned
        # i.e., I don't know how scope works in Python so I want to make sure a is scoped to the proper field
        # but I also want to make sure if it isn't selected properly the env throws an error.
        # This is because I am too lazy to do error checks here
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