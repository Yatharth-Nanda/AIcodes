# -*- coding: utf-8 -*-
"""
Name: yatharth nanda 
2021AAPS1291G
"""
import numpy as np
import math 
import matplotlib.pyplot as plt 
import string
class Bandit: 
    '''This class implements the 3 algorithms we want to try  '''
    def __init__(self,value):
        self.q_values=np.random.normal(size=(2000,10))
        #created testbed 
        self.curr_q_est=[[value for _ in range(10)] for _ in range(2000)]
        #created an action value 
        self.count=np.zeros((2000,10))
        #created an array to store count        
    
    
    def epsilon_greedy(self,epsilon,time_steps,step_size):
        '''This is a combined implementation for epsilon greedy and optimisitc initial values 
        We can set the initial estimates during object instantiation to simulate the optimisitic initial values 
        step_size =0 defaults to a step size of 1/n '''
        indexes=np.arange(10)
        #creates an array for storing rewards averaged for each timestep
        self.reward_samples=[]
        most_optimal_action=[]
        #creating an array for most optimal actions 
        for i in range(2000):
            most_optimal_action.append(np.argmax(self.q_values[i]))
        
        #stores optimal action percentage 
        self.percentage=[]
        for t in range(time_steps):
            #for every timestep i call exploit or explore for each row 
            #calculate reward 
            sum_reward=0 # stored my reward across bandits for a single timestep 
            percent_action=0 # stores number of bandits that took optimal action for that timestep
            for i in range(2000): #iterate over bandits 
                if(np.random.rand()<epsilon):
                    curr_index=np.random.choice(indexes)
                else :
                    curr_index=np.argmax(self.curr_q_est[i])
                
                #checking if the current action is the most optimal 
                if(curr_index==most_optimal_action[i]):
                    percent_action+=1
                
                
                #updating the index 
                curr_reward=self.q_values[i][curr_index]+np.random.normal()
                self.count[i][curr_index]+=1
                value=self.curr_q_est[i][curr_index]
                if(step_size==0): # step_size 0 means use 1/n as step_size where n is number of times the action has been accessed 
                    self.curr_q_est[i][curr_index]=value+(1/self.count[i][curr_index])*(curr_reward-value)
                else :
                    self.curr_q_est[i][curr_index]=value+(step_size)*(curr_reward-value)
                #current estimate updated and count updated 
                sum_reward+=curr_reward                
               
            self.reward_samples.append(sum_reward/2000)            
            self.percentage.append(percent_action/20)
        return self.reward_samples,self.percentage
    
    
    def upper_confidence_bound(self,time_steps,c):
        '''This function implements upper confidence bound '''
        self.reward_samples=[]
        self.percentage=[]
        most_optimal_action=[]
        for i in range(2000):
            most_optimal_action.append(np.argmax(self.q_values[i]))
        for t in range(time_steps):
            #for every timestep i call exploit or explore for each row 
            #calculate reward 
            sum_reward=0
            percent_action=0              
            for i in range(2000):
                max_value=0
                for j in range(10): # explore each arm to determine max arm , if any arm has zero , that is the max and break loop
                    if(self.count[i][j]==0):
                        curr_lever=j
                        break
                    elif (max_value<self.curr_q_est[i][j]+c*math.sqrt(math.log(t)/self.count[i][j])):
                        max_value=self.curr_q_est[i][j]+c*math.sqrt(math.log(t)/self.count[i][j])
                        curr_lever=j
                if(curr_lever==most_optimal_action[i]):
                    percent_action+=1
                curr_reward=self.q_values[i][curr_lever]+np.random.normal()
                self.count[i][curr_lever]+=1
                value=self.curr_q_est[i][curr_lever]
                
                self.curr_q_est[i][curr_lever]=value+(1/self.count[i][curr_lever])*(curr_reward-value)
                
                #current estimate updated and count updated 
                sum_reward+=curr_reward 
            self.reward_samples.append(sum_reward/2000)
           
            self.percentage.append(percent_action/20)
        return self.reward_samples,self.percentage                       
     
               
class MRP :          
        
    def __init__(self):
        #creating estimates for my terminal states and normal states 
        self.curr_q_est = np.full(7, 0.5)
        self.curr_q_est[0]=self.curr_q_est[6]=0        
        self.true_values=[1/6 ,2/6,3/6,4/6,5/6] # true values of states 
       
    def episode_generator(n):
        # 0  1 2 3 4 5 6 
        # 3 is my starting state 
        #pick action if not at 0 or 6
        # if at 0 or 6 terminate episode and add to list 
        
        episodes=[]
        for i in range(n):
            curr_state=3
            curr_episode=[]
            while curr_state!=0 and curr_state!=6:
                #add current action to episode 
                curr_episode.append(curr_state)
                #pick action                 
                curr_state+= np.random.choice([-1,1])# random choice between one step to right and one to the left 
            curr_episode.append(curr_state)
            episodes.append(curr_episode)
        return episodes 
    
    #generates a reward based on state transitions
    def reward_generator(curr_state,next_state):
        if(curr_state==5 and next_state==6):
            return 1
        else :
            return 0       
    
    
    def temporal_difference(self,n,alpha,gamma):
        episodes=MRP.episode_generator(n)
        
        rms_values=[]
        #iterating over episodes
        for ep in episodes:
           
            for i in range(len(ep)):
                curr_state=ep[i]
                next_state=0
                if(i+1<len(ep)):
                    next_state=ep[i+1]                
                
                #is not at terminal state , perform update
                if(curr_state!=0 and curr_state!=6):# unterminated                    
                    self.curr_q_est[curr_state]=self.curr_q_est[curr_state]+alpha*(MRP.reward_generator(curr_state,next_state)+ gamma*self.curr_q_est[next_state]-self.curr_q_est[curr_state])
                    
            rms=0#to store root mean square error value 
            for i in range (1,6):
                rms+=pow(self.curr_q_est[i]-self.true_values[i-1],2)
            rms_values.append(math.sqrt(rms/5))
                
        output=[]
        #extract estimates for estimates
        for i in range(1,6):
            output.append(self.curr_q_est[i])
            
        
        return output ,rms_values  
 
def main():
    
    
    #EPSILON GREEDY FOR 4 VALUES OF EPSILON , 1000 timesteps , intial estimates are zero 
    b1=Bandit(0)
    b2=Bandit(0)
    b3=Bandit(0)
    b4=Bandit(0)
    b5=Bandit(0)
    avg_reward1,percent_action1=b1.epsilon_greedy(0.1,1000,0)
    avg_reward2,percent_action2=b2.epsilon_greedy(0.01,1000,0)
    avg_reward3,percent_action3=b3.epsilon_greedy(0.03,1000,0)
    avg_reward4,percent_action4=b4.epsilon_greedy(0,1000,0)
    avg_reward5,percent_action5=b5.epsilon_greedy(1,1000,0)
    x_axis=range(1,1001)
    plt.figure(figsize=(12,6))
    plt.plot(x_axis,avg_reward1,label='E=0.1')
    plt.plot(x_axis,avg_reward2,label='E=0.01')
    plt.plot(x_axis,avg_reward3,label='E=0.03')
    plt.plot(x_axis,avg_reward4,label='E=0')
    plt.plot(x_axis,avg_reward5,label='E=1')
    plt.ylabel('Average reward ')
    plt.xlabel('Timesteps')
    plt.title('Average reward for Epsilon greedy action selection ')    
    plt.ylim(0,1.75)
    plt.legend()
    
    plt.figure(figsize=(12,6))
    plt.plot(x_axis,percent_action1,label='E=0.1')
    plt.plot(x_axis,percent_action2,label='E=0.01')
    plt.plot(x_axis,percent_action3,label='E=0.03')
    plt.plot(x_axis,percent_action4,label='E=0')
    plt.plot(x_axis,percent_action5,label='E=1')
    plt.ylabel('Percentage Optimal action  ')
    plt.xlabel('Timesteps')
    plt.title(' Percentage Optimal action for different values of epsilon ')    
    plt.ylim(0, 100)
    plt.legend()
    
    
    #OPTIMISTIC INITIAL VALUES 
    a1=Bandit(0)
    a2=Bandit(5)
    a3=Bandit(3)
    avg_reward11,percent_action11=a1.epsilon_greedy(0.1,1000,0)
    avg_reward12,percent_action12=a2.epsilon_greedy(0,1000,0.1)
    avg_reward13,percent_action13=a3.epsilon_greedy(0,1000,0.1)
    
    x_axis=range(1,1001)
    plt.figure(figsize=(12,6))
    plt.plot(x_axis,avg_reward11,label='Epsilon greedy with E=0.1')
    plt.plot(x_axis,avg_reward12,label='optimisitic initial values ,intial value =5,E=0.1 ')   
    plt.plot(x_axis,avg_reward13,label='optimisitic initial values ,intial value =3,E=0.1 ')   
    plt.ylabel('Average reward for optimisitic initial values ')
    plt.xlabel('Timesteps')
    plt.title(' Average reward for optimisitic initial values vs epsilon greedy ')  
    plt.ylim(0,1.75)
    plt.legend()
    
    plt.figure(figsize=(12,6))
    plt.plot(x_axis,percent_action11,label='E=0.1')
    plt.plot(x_axis,percent_action12,label='optimistic initial values ,intial value =5  ')
    plt.plot(x_axis,percent_action13,label='optimistic initial values ,initial value =3  ')
    plt.ylabel('Percentage Optimal action  ')
    plt.xlabel('Timesteps')
    plt.title(' Percentage Optimal action for Optimistic Initial Values   ') 
    plt.ylim(0, 100)
    plt.legend()
    
    
    #UPPERCONFIDENCE BOUND AND EPSILON GREEDY 
   # Create two Bandit instances with different initial values
    c1 = Bandit(0)
    c2 = Bandit(0)
    c3=Bandit(0)
    
    # Apply epsilon-greedy and UCB algorithms to these Bandits
    avg_reward21, percent_action21 = c1.epsilon_greedy(0.1, 1000, 0)
    avg_reward22, percent_action22 = c2.upper_confidence_bound(1000, 2)
    avg_reward23, percent_action23 = c3.upper_confidence_bound(1000, 5)
    
    # Plot the results
    x_axis = range(1, 1001)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, avg_reward21, label='Epsilon-Greedy (E=0.1)')
    plt.plot(x_axis, avg_reward22, label='UCB (c=2)')
    plt.plot(x_axis, avg_reward23, label='UCB (c=5)')
    plt.ylabel('Average reward for UCB   ')
    plt.xlabel('Timesteps')
    plt.title(' Average reward for different values of UCB  ') 
    plt.ylim(0, 1.75)
    plt.legend()
    plt.title('Average Reward')
    
    plt.subplot(2, 1, 2)
    plt.plot(x_axis, percent_action21, label='Epsilon-Greedy (E=0.1)')
    plt.plot(x_axis, percent_action22, label='UCB (c=2)')
    plt.plot(x_axis, percent_action23, label='UCB (c=5)')
    plt.ylabel('Percentage Optimal action  ')
    plt.xlabel('Timesteps')
    plt.title(' Percentage Optimal action for different values of UCB ') 
    plt.ylim(0, 100)
    plt.legend()
    plt.title('Percent Action')
    
    plt.tight_layout()

    
    #MARKOV REWARD PROCESS 
    d1=MRP()
    d2=MRP()
    d3=MRP()
    d4=MRP()
    
    estimate_1,rms1=d1.temporal_difference(100, 0.1,1)
    estimate_2,rms2=d2.temporal_difference(10, 0.1,1)
    estimate_3,rms3=d3.temporal_difference(0, 0.1,1)
    estimate_4,rms4=d4.temporal_difference(1, 0.1,1)
    estimate_5=d1.true_values
    
    x_labels = list(string.ascii_uppercase[:5])  # Generates ['A', 'B', 'C', 'D', 'E']

    # Plot the temporal difference estimates and true values on the same graph
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_labels, estimate_1, label=' (100 steps)')
    plt.plot(x_labels, estimate_2, label=' (10 steps)')
    plt.plot(x_labels, estimate_3, label=' (0 steps)')
    plt.plot(x_labels, estimate_4, label=' (1 step)')
    plt.plot(x_labels, estimate_5, label=' True Values', linestyle='--')  # True values as a dashed line
    plt.xlabel('state-value  estimates ')
    plt.ylabel('Temporal Difference Estimate / True Values')
    plt.title('Temporal Difference Estimates over different number of epsiodes and True Values')
    plt.legend()
    
    plt.tight_layout()
  
    # Calculate temporal difference estimates for each instance
    d8 = MRP()
    d10 = MRP()
    d11 = MRP()
    d12=MRP()
    
    estimate_8,rms8 = d8.temporal_difference(100, 0.03,1)
    estimate_10,rms10 = d10.temporal_difference(100, 0.1,1)
    estimate_11,rms11 = d11.temporal_difference(100, 0.15,1)
    estimate_12,rms12 = d12.temporal_difference(100, 0.05,1)
    
    plt.figure(figsize=(10, 6))    
    plt.plot(x_labels, estimate_8, label='alpha =0.03 ')    
    plt.plot(x_labels, estimate_10, label='alpha =0.1')
    plt.plot(x_labels, estimate_11, label='alpha =0.15')
    plt.plot(x_labels, estimate_12, label='alpha=0.05 ')
    plt.plot(x_labels, estimate_5, label=' True Values', linestyle='--')
    
    plt.xlabel('State-Value  estimates ')
    plt.ylabel('Temporal Difference Estimate / True Values')
    plt.title('Temporal Difference Estimates over different values of learning rate   and True Values')
    plt.legend()
    
    plt.tight_layout() 
    
    # Plot the RMSE values for the second set using x_axis
    plt.figure(figsize=(12, 6))
    x_axis2=range(1,101)
  
    plt.plot(x_axis2, rms8, label='RMSE alpha=0.03')
    plt.plot(x_axis2, rms10, label='RMSE alpha=0.01')
    plt.plot(x_axis2, rms11, label='RMSE alpha=0.15')
    plt.plot(x_axis2, rms12, label='RMSE alpha=0.05')
    plt.xlabel('Number of Episodes ')
    plt.ylabel('Root Mean Square Error (RMSE) Value')
    plt.title('RMSE Values for Temporal Difference Estimates for different values of learning rate  ')
    plt.legend()
    
    plt.tight_layout()
    plt.show()   

if __name__ == "__main__":
    main()
