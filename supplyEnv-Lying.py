from gym.spaces import Discrete,MultiDiscrete,Box
import datetime
import os 
 
# import time 
import csv

import numpy as np 
import time
from gym import utils 
 
from gym.utils import seeding 
from ray.rllib.utils.typing import AgentID
from ray.tune.logger import pretty_print
# from ray.rllib.algorithms.algorithm import Algorithm
# from ray.rllib.env.policy_server_input import PolicyServerInput
# from ray import tune
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from ray.rllib.env.multi_agent_env import MultiAgentEnv 
from scipy.stats import poisson, binom, randint, geom 

from itertools import chain
import collections

ep_count = 0
class POMultiAgent1W1F(MultiAgentEnv): 
    def __init__(self,  *args, **kwargs):#
        # super().__init__(*args, **kwargs)
        self.factory_capacity = 60
        self.retailer_capacity = 20 
        self.price_capacity = 60
        factory_capacity = self.factory_capacity 
        retailer_capacity = self.retailer_capacity 
        price_capacity = self.price_capacity 
        self.product_types_num = 1
        factory_size=1
        #self.v_f = 2 # 1:the factory does not show its stock; 2:factory can choose to show its stock or not.
        self.lying = 60
        low_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
        high_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
        high_act[0] = self.retailer_capacity
        high_act[1] = self.price_capacity
        high_act[2] = self.lying
        self.action_space = Box(low=low_act, high=high_act, dtype=np.float32,seed=2024)
        #self.action_space = MultiDiscrete([retailer_capacity + 1,price_capacity+1,self.v_f])  #[order , price, show]
        self.periods = 30 #simulation days
        low_obs = np.zeros((8,), dtype=np.float32)
        high_obs = np.zeros((8,), dtype=np.float32)
        high_obs[:factory_size*7] = factory_capacity
        high_obs[factory_size*7:factory_size*7+1] = self.periods
        self.observation_space = Box(low=low_obs, high=high_obs, dtype=np.float32,seed=2024)
        #self.observation_space = MultiDiscrete([factory_capacity+1,factory_capacity+1,factory_capacity+1,factory_capacity+1,factory_capacity+1,factory_capacity+1,factory_capacity+1,self.periods+1]) # [inventory,backlog,stockout,Factory_inventory]
        self.state =  {1:np.array([10,0,0,0,0,0,0,0]), 2: np.array([10,0,0,0,0,0,0,0])}  # an state example
        self.r = {1:0, 2:0} 
        self.info = {1: {'retalier obs': self.state[1]}, 2: {'factory obs': self.state[2]}} 
        self.k = 5 
        self.c = 2 
        self.h = 2 
        self.b = 5
        self.s = [200,100]
        self.re_num = 0
        self.day =  {1: 0, 2:0}
        self.days =  {1: self.periods, 2:self.periods}
        self.stockoutCount =  {1: 0, 2:0}
        self.stockoutCountS = {1: 6, 2:6}
        self._seed() 
        self.dist = 6  #This parameter can be selected for 1-5 different demand distributions.
        self.ALPHA = 0.5
        self.dist_param = {'mu':10,'muDemand':2,'stdDemand':1}  
        self.d_max = 30
        self.d_var = 5
        self.task = 1 #1: rewards set remains the same, 2: factory rewards need to be subtracted from retailer rewards, 3: retailer incentives need to be subtracted from factory rewards
        self.demand_len=3
        self.done = {1: False, 2: False}
        # self.g = 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.demand_history = []
        self.percent_history = []
        self.reset() 
        
        self.nocoop_count = 0
        self.lying_count =0
        self.trust_count = 0
        self.agent1_action_1_count = 0
        self.agent1_action_0_count = 0
       # self.agent2_action_1_count = 0
      #  self.agent2_action_0_count = 0
        self.total_steps = 0 

    def demand(self):  
        distributions = {  
            1: {'dist': poisson, 'params': {'mu': self.dist_param['mu']}},  
            2: {'dist': binom, 'params': {'n': 10, 'p': 0.5}},  
            3: {'dist': randint, 'params': {'low': 0, 'high': 50}},  
            4: {'dist': geom, 'params': {'p': 0.5}},  
            5: {'dist': 'seasonal', 'params': None},
            6: {'dist': 'lasc', 'params': {'muDemand': self.dist_param['muDemand'], 'stdDemand': self.dist_param['stdDemand']}}  
        }  
    
        if self.dist in distributions:
            dist_info = distributions[self.dist]  
            dist_instance = dist_info['dist']  
            dist_params = dist_info['params']  
            
            # Handle standard distribution cases (1 to 4)
            if self.dist < 5 and dist_instance is not None:  
                return dist_instance.rvs(**dist_params)  
    
            # Handle seasonal demand (dist == 5)
            elif self.dist == 5:  
                current_date = datetime.datetime.now()  
                t = current_date.timetuple().tm_yday  # Day of the year (1 to 365)
                demand = np.round(  
                    self.d_max / 3 +  
                    self.d_max / 5 * np.cos(4 * np.pi * (2 + t) / self.periods) +  
                    np.random.randint(0, self.d_var + 1)  
                )  
                demand = min(demand, 30) 
                return int(demand)  # Return as an integer
            
            # Handle custom normal distribution (dist == 6)
            elif self.dist == 6:
                muDemand = self.dist_param['muDemand']
                stdDemand = self.dist_param['stdDemand']
                demand = np.floor(max(0, np.random.normal(muDemand, stdDemand)))
                return int(demand)  # Ensure it returns an integer
        else:
            raise ValueError(f"Invalid distribution type: {self.dist}")
           
  
    
    def transition(self,agent_ids, x, a_factory, a_retailer,V_F,d):  
         I = [value[0] for value in x.values()] 
         BL = [value[1] for value in x.values()] 
         SO = [value[2] for value in x.values()] 
    
         total_inventory_factory = a_factory + I[1] 
         factory_stockout = False 
         if  0 <= total_inventory_factory <= self.factory_capacity:  #### blance 
             I[1] = total_inventory_factory 
             if  I[1] < a_retailer: #### stockout 
                 SO[1] = a_retailer-I[1] 
                 if SO[1] > 0: 
                     factory_stockout = True 
                     for agent_id in agent_ids: 
                         self.stockoutCount[agent_id] += 1 
                         a_retailer = I[1] #*********     order changed = real order/demand 
                 I[1] = 0 
                 BL[1] = 0 
             else: 
                 I[1] -= a_retailer 
                 SO[1] = 0 
                 BL[1] = 0 
    
         elif total_inventory_factory > self.factory_capacity: 
                 BL[1] = total_inventory_factory-self.factory_capacity 
                 I[1] = total_inventory_factory 
                 if I[1] < a_retailer: #### stockout 
                     SO[1] = a_retailer-I[1] 
                     if SO[1] > 0: 
                         factory_stockout = True 
                         for agent_id in agent_ids: 
                             self.stockoutCount[agent_id] += 1 
                     I[1] = 0        
                 else: 
                     I[1] -= a_retailer 
                     SO[1] = 0 
                     BL[1] = np.where(I[1] < self.factory_capacity, 0, BL[1])  
    
         I[1] = np.clip(I[1],0, self.factory_capacity-1) 
         BL[1] = np.clip(BL[1],0, self.factory_capacity-1) 
         SO[1] = np.clip(SO[1],0, self.factory_capacity-1) 
         if not factory_stockout: 
             total_inventory_retailer = a_retailer + I[0] 
             if  0 < total_inventory_retailer <= self.retailer_capacity: 
                 I[0] = total_inventory_retailer 
                 if  I[0] < d: 
                     SO[0] = d-I[0] 
                     if SO[0] > 0: 
                         for agent_id in agent_ids: 
                             self.stockoutCount[agent_id] += 1 
                             d = I[0] #*********     order changed = real order/demand 
                     I[0] = 0 
                     BL[0] = 0 
                 else: 
                     I[0] -= d 
                     SO[0] = 0 
                     BL[0] = 0 
    
             elif total_inventory_retailer > self.retailer_capacity: 
                 # self.retailer_capacity < a_retailer+I[0]: 
                     BL[0] = total_inventory_retailer-self.retailer_capacity 
                     I[0] = total_inventory_retailer 
    
                     if d > I[0]: 
                         SO[0] = d-I[0] 
                         if SO[0] > 0: 
                             for agent_id in agent_ids: 
                                 self.stockoutCount[agent_id] += 1 
                         I[0] = 0 
                         # BL[1] = 0 
                     else: 
                         I[0] -= d 
                         SO[0] = 0 
                         BL[0] = np.where(I[0] < self.retailer_capacity, 0, BL[0]) 
    
    
             I[0] = np.clip(I[0],0, self.retailer_capacity-1) 
             BL[0] = np.clip(BL[0],0, self.retailer_capacity-1) 
             SO[0] = np.clip(SO[0],0, self.retailer_capacity-1) 
    
         else: 
             a_retailer = 0 #*********     order changed = real order/demand 
             total_inventory_retailer = I[0] 
             if  I[0] < d: 
                 SO[0] = d-I[0] 
                 if SO[0] > 0: 
                     for agent_id in agent_ids: 
                         self.stockoutCount[agent_id] += 1 
                         d = I[0] #*********     order changed = real order/demand 
                 I[0] = 0 
                 BL[0] = 0 
             else: 
                 I[0] -= d 
                 SO[0] = 0 
                 BL[0] = 0 
    
              
    
         self.o_history.append(np.array([[a_retailer]], dtype=np.int64)) 
         o_history = np.hstack(list(chain(*chain(*self.o_history)))) 
         self.d_history.append(np.array([[d]], dtype=np.int64)) 
         d_history = np.hstack(list(chain(*chain(*self.d_history)))) 
         self.period +=1 
    
         
    
         x = {1: np.array([I[0],BL[0],SO[0],d_history[0],d_history[1],d_history[2],V_F,self.period]),
               2: np.array([I[1],BL[1],SO[1],o_history[0],o_history[1],o_history[2],0,self.period])}  
    
          

         return x 

 
    def reward(self, x, a_retailer,p_retailer,a_factory, p_factory,d, y): 
      
        k = self.k 
        c = self.c 
        h = self.h 
       
 
        factory_reward =  - c * max(min(y[2][0] + a_factory, self.factory_capacity) - y[2][0], 0)-h * y[2][0] + p_factory * max(min(x[2][0] + x[2][5], self.factory_capacity) - x[2][0], 0) 
        if self.task == 2:
            factory_reward -= self.s[0]*y[1][2]
        
       
        self.day[1] += 1
 
        retailer_reward = - p_factory * max(min(y[1][0] + y[2][5], self.retailer_capacity) - y[1][0], 0)-h * y[1][0]   + p_retailer * max(min(x[1][0] + x[1][5], self.retailer_capacity) - x[1][0], 0) 
        if self.task == 3:
            retailer_reward -= self.s[1]*y[2][2]
            
            
        
        if self.task == 4:
            factory_reward -= self.s[0]*y[1][2]
            retailer_reward -= self.s[1]*y[2][2]
        
        
        self.r[1] = -self.s[0]*y[1][2] * 0.7 -self.b * y[1][1] * 0.2 + retailer_reward*0.1
        
        self.r[2] = -self.s[1]*y[2][2] * 0.7 - self.b*y[2][1] * 0.2 +factory_reward*0.1

 
       
        self.day[2] += 1
        
        
       
        return {1: self.r[1], 2: self.r[2]} 
 
    def _seed(self, seed=None): 
        
 
        self.np_random, seed = seeding.np_random(seed) 
 
        return [seed] 
 
    def step(self, actions): 
        global ep_count
        ep_count += 1
       
        agent_ids = actions.keys() 
        
        obs_dict = self.state.copy() 
    
 
        demand = self.demand() 
        
        actions = {i:np.floor(a) for i , a in actions.items()}
        for key in actions:
            if not actions[key].flags.writeable:
                actions[key] = actions[key].copy()
        
        actions = actions.copy()
        #if self.v_f == 2:
           # actions[2][2] = np.where(actions[2][2]>=1,1,0)
        #elif self.v_f == 3:
           # actions[2][2] = 1
        #else:
          #  actions[2][2] = 0
          
        #if actions[2][2] <1:
          #  self.lying = actions[2][2]
        #elif 1<=actions[2][2] <2:
           # actions[2][2] = 1
       # else:
           # actions[2][2] = 0
            
       
        

        observations_dict = self.transition(agent_ids,obs_dict, actions[2][0], actions[1][0],actions[2][2], demand) 

 
        
 
        self.state = observations_dict 

        self.info = {1: {'retalier obs': self.state[1]}, 2: {'factory obs': self.state[2]}} 
 
         
 
        rewards = self.reward(obs_dict, actions[1][0],actions[1][1],  actions[2][0], actions[2][1], demand,observations_dict) 
        #processed_rewards = self.get_processed_reward(rewards)
 
         
 
        self.done = {i: self.is_done(i) for i in agent_ids} 
 
 
 
 
        self.done["__all__"] = all(self.done.values()) 
        self.state_history.append(self.state.copy())
        self.reward_history.append(rewards.copy())
        self.demand_history.append(demand)
        # 检查并修改可写性
        

        # 修改元素
        actions[1][2] = 0
        
        if actions[1][2] == 0:
            self.agent1_action_1_count = 0
            self.agent1_action_0_count += 1

        #if actions[2][2] == 1:
           # self.agent2_action_1_count += 1
        #else:
            #self.agent2_action_0_count += 1
            
        
        
        self.total_steps += 1
        actions[1][1]=0.0
       
        self.action_history.append(actions.copy())
        
        
        
        
        
     
        
        self.render()
 
      
    

        return observations_dict, rewards, self.done, self.info 
    

    def get_processed_reward(self,rewards):
        processed_rewards = {}
        rewards_list = list(rewards.values())
        mean_reward = np.mean(rewards_list)
        for key, value in rewards.items():
            processed_rewards[key] =self.ALPHA * value + (1 - self.ALPHA) * mean_reward
        return processed_rewards
    
    
    def is_done(self, agent_id): 
        if self.stockoutCount[agent_id] >= self.stockoutCountS[agent_id] or self.day[agent_id] >= self.days[agent_id]:

            done = True
        else:
            done = False
 
        return done 
    
    
    def save_to_csv(self, filename=None):
     
        data = []
        for i in range(len(self.state_history)):
            row = {
                "state_1_inventory": self.state_history[i][1][0],
                "state_1_backlog": self.state_history[i][1][1],
                "state_1_stockout": self.state_history[i][1][2],
                "state_1_demand1": self.state_history[i][1][3],
                "state_1_demand2": self.state_history[i][1][4],
                "state_1_demand3": self.state_history[i][1][5],
                "state_1_SHOW_Factory_inv": self.state_history[i][2][6],
                "state_1_steps": self.state_history[i][1][7],

                "action_1_order": self.action_history[i][1][0],
                "action_1_price": self.action_history[i][1][1],
                "action_1_SHOW": self.action_history[i][1][2], ### always 0
                
                "reward_1": self.reward_history[i][1],

                "state_2_inventory": self.state_history[i][2][0],
                "state_2_backlog": self.state_history[i][2][1],
                "state_2_stockout": self.state_history[i][2][2],
                "state_2_order1": self.state_history[i][2][3],
                "state_2_order2": self.state_history[i][2][4],
                "state_2_order3": self.state_history[i][2][5],
                "state_2_SHOW_Factory_inv": self.state_history[i][2][6],
                "state_2_steps": self.state_history[i][2][7],

                "action_2_order": self.action_history[i][2][0],
                "action_2_price": self.action_history[i][2][1],
                "action_2_SHOW": self.action_history[i][2][2],  ### 0 or 1
          
                "reward_2": self.reward_history[i][2],  
              
            }
            data.append(row)
    

        with open(filename, 'w', newline="") as csvfile:
            fieldnames = [
                "state_1_inventory", "state_1_backlog", "state_1_stockout","state_1_demand1","state_1_demand2","state_1_demand3",
                "state_1_SHOW_Factory_inv","state_1_steps",
                "action_1_order",
                "action_1_price",
                "action_1_SHOW",
                "reward_1",
                "state_2_inventory", "state_2_backlog", "state_2_stockout","state_2_order1","state_2_order2","state_2_order3",
                "state_2_SHOW_Factory_inv","state_2_steps",
                "action_2_order",
                "action_2_price",
                "action_2_SHOW",
                "reward_2"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def reset(self):
        self.re_num += 1
        self.stockoutCount =  {1: 0, 2:0}
        self.day =  {1: 0, 2:0}
        self.d_history = collections.deque(maxlen=self.demand_len)
        for d in range(self.demand_len):
            self.d_history.append(np.zeros((1,1), dtype = np.float32))

        self.o_history = collections.deque(maxlen=self.demand_len)
        for d in range(self.demand_len):
            self.o_history.append(np.zeros((1,1), dtype = np.float32))
       
        
        self.state = {1: np.array([10,0, 0,0, 0,0, 0,0]), 2: np.array([10, 0, 0,0,0,0, 0,0])}
        self.period = 0
        self.r = {1: 0, 2: 0}
      
        self.days = {1: self.periods, 2: self.periods}
        #self.agent1_action_1_count = 0
        #self.agent1_action_0_count = 0
        self.nocoop_count =0
        self.trust_count = 0
        self.lying_count =0

        #self.agent2_action_1_count = 0
        #self.agent2_action_0_count = 0
        self.total_steps = 0
      
       
        return self.state#,self.done

    
    def render(self, mode="human", log_dir='./plots'):
        # tomato aquamarine pink dodgerblue lightskyblue dodgerblue skyblue lightsteelblue  darkkhaki  darksalmon 1 powderblue khaki skyblue 
        current_date = datetime.datetime.now()
        # num_periods = ep_count // self.periods
        global ep_count
        # ep_count += 1
        p = self.periods
        p = 500 if p <= 500 else p       
        if ep_count % p == 0 and len(self.state_history) % p==0:
       
            start = max(0,len(self.state_history)-p)
           
            end = len(self.state_history)
          
            
            fig = plt.figure(figsize=(5 * 2, 10))  # 2 agents
            colors = list(mcolors.TABLEAU_COLORS.keys())   
            agent_labels = ["Retailer", "Factory"]  
          

            for i in range(2):
                ax = fig.add_subplot(5, 2, i + 1)
                if i==0:
                    ax.plot(range(start , end),[s[i+1][5] for s in self.state_history[start:end]], label=f'Customer Demand',  color='b', alpha=0.8, linewidth=0.8)
                else:  
                    ax.plot(range(start , end),[s[i+1][5] for s in self.state_history[start:end]], label=f'Retailer Order',  color='b', alpha=0.8, linewidth=0.8)
                #if i==0:
                    #ax.plot(range(start, end), [a[i+1][0] for a in self.action_history[start:end]], label=f'Factory Order', color='mediumslateblue',alpha=0.85, linewidth=0.8)
                #else:
                    #ax.plot(range(start, end), [a[i+1][0] for a in self.action_history[start:end]], label=f'Supplier Order', color='mediumslateblue',alpha=0.85, linewidth=0.8)
                    
                ax.plot(range(start, end), [a[i+1][0] for a in self.action_history[start:end]], label=f'{agent_labels[i]} Order', color='#F79EB3',alpha=0.85, linewidth=0.8)
                if i==0:
                    ax.plot(range(start, end), [a[i+1][1] for a in self.action_history[start:end]], color="ghostwhite", alpha=0.1, linewidth=0.1)
                else:
                    # Extract prices
                    prices = [a[i+1][1] for a in self.action_history[start:end]]
                    
                    # Calculate mean and standard deviation
                    mean_price = sum(prices) / (end - start)
                    std_price = np.std(prices, ddof=1)
                    ax.plot(range(start, end), [a[i+1][1] for a in self.action_history[start:end]], label=f'{agent_labels[i]} Price: {mean_price:.2f}, Std: {std_price:.2f}', color="#BCBBD9", alpha=0.4, linewidth=1.7)
                
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Demand & Order & Price ', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1]) 

            for i in range(2):
                ax = fig.add_subplot(5, 2, i + 3)
                mean_inv = np.mean([s[i+1][0] for s in self.state_history[start:end]])
                ax.plot(range(start, end), [s[i+1][0] for s in self.state_history[start:end]], label=f'{agent_labels[i]} Inventory: {mean_inv:.2f}',  color='#DAF3F2',alpha=0.8, linewidth=2)
                
                mean_bac = np.mean([s[i+1][1] for s in self.state_history[start:end]])
                ax.plot(range(start, end), [s[i+1][1] for s in self.state_history[start:end]], label=f'{agent_labels[i]} Backlog: {mean_bac:.2f}', color='#E8859D', linewidth=1)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Inventory & Backlog', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1])
            
            for i in range(2):
                ax = fig.add_subplot(5, 2, i + 5)
                ax.plot(range(start, end), [s[i+1][0] for s in self.state_history[start:end]], label=f'{agent_labels[i]} Inventory', color='#DAF3F2',alpha=0.8, linewidth=2)
                
                mean_sto = np.mean([s[i+1][2] for s in self.state_history[start:end]])
                ax.plot(range(start, end), [s[i+1][2] for s in self.state_history[start:end]], label=f'{agent_labels[i]} Stockout: {mean_sto:.2f}', color='purple', linewidth=1)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Inventory & Stockout', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1])

            for i in range(2):
                ax = fig.add_subplot(5, 2, i + 7)
                mean_profit = sum([r[i+1] for r in self.reward_history[start:end]]) / (end - start)
                ax.plot(range(start, end), [r[i+1] for r in self.reward_history[start:end]], label=f'{agent_labels[i]} Mean Profit: {mean_profit:.2f}', color="#D8AFDD", alpha=0.8,linewidth=0.65)
               
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Profit', fontsize='medium')
                ax.set_xlabel('Step')

           # for i in range(2):
             #   ax = fig.add_subplot(5, 2, i + 9)
                #count_of_ones = count_of_zeros = count_of_lying = 0
                 # Initialize counters for ones, zeros, and lying
                
                
                # One pass over the data to calculate all counts
                #for a in self.action_history[start:end]:
                    #if a[i+1][2] == 1:
                      #  count_of_ones += 1
                  #  elif a[i+1][2] == 0:
                       # count_of_zeros += 1
                 #   else:
                      #  count_of_lying += 1
                        
                        
                        
                        
                    #if actions[2][2] < 1:
                       # self.lying_count +=1
        
                    #elif actions[2][2] == 0:
                     #   self.trust_count += 1
                    #else:
                       # self.nocoop_count += 1
                
                # Calculate percentages
               # count_1_percent = (count_of_ones / (end - start)) * 100
                #count_0_percent = (count_of_zeros / (end - start)) * 100
                #count_lying_percent = (count_of_lying / (end - start)) * 100

                #count_of_ones = count_of_zeros = count_of_lying = 0
                #count_of_ones = sum(1 for a in self.action_history[start:end] if a[i+1][2] == 1)
                #count_1_percent = (count_of_ones/(end-start))*100
                #count_of_zeros = sum(1 for a in self.action_history[start:end] if a[i+1][2] == 0)
                #count_0_percent = (count_of_zeros/(end-start))*100
                #count_of_lying = sum(1 for a in self.action_history[start:end] if a[i+1][2] < 1)
                #count_lying_percent = (count_of_lying/(end-start))*100
                
                
                #mean_Lying = sum([a[i+1][2] for a in self.action_history[start:end]]) / (end - start)
                #ax.plot(range(start , end),[a[i+1][2] for a in self.action_history[start:end]],
                       # label=f'Lying: {mean_Lying:.2f}%',
                      #  color='#CEE0EE',alpha=0.5,linestyle='--', linewidth=0.73)
               # ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
              #  if i == 0:
                   # ax.set_ylabel('Percent', fontsize='medium')
               # ax.set_xlabel('Step')

          
        
            plt.tight_layout()
            

          
            if log_dir:
                
                os.makedirs(log_dir, exist_ok=True)
                filename = os.path.join(log_dir, f'Step{ep_count}_ep{self.re_num}_T{self.task}_{current_date}.png')
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
                
                
        if ep_count % p == 0 and len(self.state_history) % p==0:
       
            start = max(0,len(self.state_history)-p)
         
            end = len(self.state_history)
         

            fig = plt.figure(figsize=(3 * 2, 6)) 
            colors = list(mcolors.TABLEAU_COLORS.keys())    

            for i in range(2):
                ax = fig.add_subplot(3, 2, i + 1)
                if i==0:
                    ax.plot(range(start , end),[s[i+1][5] for s in self.state_history[start:end]], label=f'Customer Demand',  color='b', alpha=0.85, linewidth=0.8)
                else:
                    ax.plot(range(start , end),[s[i+1][5] for s in self.state_history[start:end]], label=f'Retailer Order',  color='b', alpha=0.85, linewidth=0.8)
                #ax.plot(range(start , end),[s[i+1][5] for s in self.state_history[start:end]], label=f'Demand',  color='b', alpha=0.85, linewidth=0.8)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Demand', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1]) 

            for i in range(2):
                ax = fig.add_subplot(3, 2, i + 3)
                ax.plot(range(start, end), [a[i+1][0] for a in self.action_history[start:end]], label=f'{agent_labels[i]} Order', color='#F79EB3',alpha=0.85, linewidth=0.8)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Order', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1]) 
                
            for i in range(2):
                ax = fig.add_subplot(3, 2, i + 5)
                ax.plot(range(start, end), [a[i+1][1] for a in self.action_history[start:end]], label=f'{agent_labels[i]} Price', color="#BCBBD9", alpha=0.4, linestyle='--',linewidth=1.7)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Price', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1]) 
        
            plt.tight_layout()
            

       
            if log_dir:
                
                os.makedirs(log_dir, exist_ok=True)
                filename = os.path.join(log_dir, f'New_Step{ep_count}_ep{self.re_num}_T{self.task}_{current_date}.png')
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

        filename = f"POVT{self.task}.csv"
    
        # Assuming self.save_to_csv method exists in your environment class
        self.save_to_csv(filename)

  
 
        
class POMultiAgent1W1F_V1T1(POMultiAgent1W1F):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.v_f = 1
        self.task = 1

             
class POMultiAgent1W1F_V1T2(POMultiAgent1W1F):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.v_f = 1
        self.task = 2

class POMultiAgent1W1F_V1T3(POMultiAgent1W1F):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.v_f = 1
        self.task = 3
        
class POMultiAgent1W1F_V1T4(POMultiAgent1W1F):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.v_f = 1
        self.task = 4
        
        
#class POMultiAgent1W1F_V2T1(POMultiAgent1W1F):
   # def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
   #     self.v_f = 2
    #    self.task = 1
        
#class POMultiAgent1W1F_V2T2(POMultiAgent1W1F):
  #  def __init__(self, *args, **kwargs):
  #      super().__init__(*args, **kwargs)
  #      self.v_f = 2
  #      self.task = 2
        
#class POMultiAgent1W1F_V2T3(POMultiAgent1W1F):
   # def __init__(self, *args, **kwargs):
   #     super().__init__(*args, **kwargs)
    #    self.v_f = 2
    #    self.task = 3
        
        
#class POMultiAgent1W1F_V3T1(POMultiAgent1W1F):
   # def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
     #   self.v_f = 3
    #    self.task = 1
     #   self.factory_capacity = 60
     #   self.retailer_capacity = 20 
    #    self.price_capacity = 60
    #    factory_capacity = self.factory_capacity 
    #    retailer_capacity = self.retailer_capacity 
     #   price_capacity = self.price_capacity 
    #    self.product_types_num = 1
    #    factory_size=1
    #    low_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
    #    high_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
     #   high_act[0] = self.retailer_capacity
     #   high_act[1] = self.price_capacity
    #    high_act[2] = self.v_f-1
    #    self.action_space = Box(low=low_act, high=high_act, dtype=np.float32,seed=2024)
        
#class POMultiAgent1W1F_V3T2(POMultiAgent1W1F):
  #  def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
   #     self.v_f = 3
     #   self.task = 2
      #  self.factory_capacity = 60
      #  self.retailer_capacity = 20 
       # self.price_capacity = 60
     #   factory_capacity = self.factory_capacity 
    #    retailer_capacity = self.retailer_capacity 
   #     price_capacity = self.price_capacity 
    #    self.product_types_num = 1
   #     factory_size=1
   #     low_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
   #     high_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
   #     high_act[0] = self.retailer_capacity
   #     high_act[1] = self.price_capacity
   #     high_act[2] = self.v_f-1
   #     self.action_space = Box(low=low_act, high=high_act, dtype=np.float32,seed=2024)
        
#class POMultiAgent1W1F_V3T3(POMultiAgent1W1F):
   # def __init__(self, *args, **kwargs):
     #   super().__init__(*args, **kwargs)
    #    self.v_f = 3
     #   self.task = 3
    #    self.factory_capacity = 60
    #    self.retailer_capacity = 20 
    #    self.price_capacity = 60
   #     factory_capacity = self.factory_capacity 
    #    retailer_capacity = self.retailer_capacity 
   #     price_capacity = self.price_capacity 
   #     self.product_types_num = 1
   #     factory_size=1
   #     low_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
    #    high_act = np.zeros(((2 + 1) * self.product_types_num), dtype=np.float32)
    #    high_act[0] = self.retailer_capacity
    #    high_act[1] = self.price_capacity
    #    high_act[2] = self.v_f-1
     #   self.action_space = Box(low=low_act, high=high_act, dtype=np.float32,seed=2024)
        
        
  

if __name__ == '__main__':
    env = POMultiAgent1W1F_V2T1()
    episodes = 1
    
    for episode in range(episodes):
        # print(f'++++++++++++++ Episode {episode} ++++++++++++++')
        
        # Reset the environment at the start of each episode
        obs = env.reset()
        
        # Initialize the `done` dictionary for all agents
        done = {1: False, 2: False}
        i=0
    
        while not done.get("__all__", False):  # Safely check for "__all__"
        
            print('+++++++++++++++++++++++++',i,'++++++++++++++++++++++++++')
            # Take a step in the environment with random actions for both agents
            actions = {
                1: env.action_space.sample(),
                2: env.action_space.sample()
            }
            obs, rew, done, info = env.step(actions)
            env.render()
    
            # Ensure "__all__" is calculated if not provided by env
            done["__all__"] = all(done.values())
    
            # Display observations, rewards, and other information
            print(f'Observations: {obs}')
            print(f'Rewards: {rew}')
            print(f'Done: {done}')
            print(f'Info: {info}')
            i += 1
    
        print(f'Episode {episode} completed.\n')
    
    print("All episodes completed.")

    # env = POMultiAgent1W1F_V2T1()
    # episodes = 50
    
    # for episode in range(episodes):
    #     print(f'++++++++++++++ Episode {episode} ++++++++++++++')
        
    #     # Reset environment at the start of each episode
    #     obs, done = env.reset()
    #     print("done",done) #done {1: False, 2: False}
    
    #     while not done["__all__"]:
    #         # Take a step in the environment with random actions for both agents
    #         actions = {
    #             1: env.action_space.sample(),
    #             2: env.action_space.sample()
    #         }
    #         obs, rew, done, info = env.step(actions)
    
    #         # Display observations, rewards, and other information
    #         print(f'Observations: {obs}')
    #         print(f'Rewards: {rew}')
    #         print(f'Done: {done}')
    #         print(f'Info: {info}')
    
    #     print(f'Episode {episode} completed.\n')
    
    # print("All episodes completed.")

   
    # env = POMultiAgent1W1F_V2T1()
    # episides = 50
    # obs = env.reset()
    # for i in range(episides): 
        
    #     # while True:
    #     print('+++++++++++++++++egthrget5++++++++',i,'++++++++++++++++++++++++++')
    
    #     obs, rew, done, info = env.step( 
    
    #         {1: env.action_space.sample(), 2: env.action_space.sample()} 
    
    #     ) 
    #     print('obs',obs) 
    
    #     print('rew',rew) 
    
    #     print('done',done) 
    
    #     print('info',info) 
    # #         if done :
    # #             break
    # # assert done

    

   
