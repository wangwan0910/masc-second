 
 
 ####finish same max days
 
# Run successful on MAC computer 
 
# -*- coding: utf-8 -*- 
 
""" 
 
Created on Tue Apr  9 14:11:52 2024 
 
@author: wangw  
 
gymnasium  0.28.1 
 
gym     0.23.1 
 
ray      2.2.0 

This current environment can be viewed as one factory and one retailer with 5 different demand distribution(user-specified), procceed rewards,
with links to its original environment, and the original environment can be viewed as just one retailer.
The observation space for each agent is the [inventory,backlog, stockout] and the action space is the value of [orders,price].
https://github.com/paulhendricks/gym-inventory/blob/master/gym_inventory/envs/inventory_env.py

""" 
 
# -*- coding: utf-8 -*- 
 
""" 
 
Spyder Editor 
 
  
 
This is a temporary script file. 
 
""" 
 
from typing import Set
from ray.rllib.algorithms.dqn import DQNConfig 
from ray.rllib.algorithms.ppo import PPOConfig 
from ray.rllib.env.multi_agent_env import MultiAgentEnv 
 
import gymnasium as gym 
from collections import OrderedDict
from gym.spaces import Discrete,MultiDiscrete,Dict
 
import os 
 
import time 
 
import numpy as np 
 
from gym import utils 
 
from gym.utils import seeding 
from ray.rllib.utils.typing import AgentID
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray import tune 
from scipy.stats import poisson, binom, randint, geom 
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env



class MultiAgent1W1F(MultiAgentEnv): 
 
  
 
    def __init__(self,  *args, **kwargs):#max_inventory=20, factory_capacity=100, k=5, c=2, h=2, p=3, lam=8 
        self.factory_capacity = 100 
        self.retailer_capacity = 10 
        self.price_capacity = 10
        factory_capacity = self.factory_capacity 
        retailer_capacity = self.retailer_capacity 
        price_capacity = self.price_capacity 
        # self.state =  {1:np.array([10,0,0]), 2: np.array([10,0,0])} 
        self.I = [self.retailer_capacity,self.retailer_capacity]
        self.BL = [0,0]
        self.SO = [0,0] 
        state = {1: np.array([self.I[0],self.BL[0],self.SO[0]]), 2:  np.array([self.I[1],self.BL[1],self.SO[1]])}
        self.state = OrderedDict(state)
        self._agent_ids = list(self.state.keys())
        self.action_space =  Dict(
            {
                x: MultiDiscrete([retailer_capacity+1, price_capacity+1], dtype=np.int64)
                for x in self._agent_ids
            }
        )
        self.observation_space = Dict(
            {
                x: MultiDiscrete([factory_capacity*2+1, factory_capacity*2+1, factory_capacity*2+1], dtype=np.int64)
                for x in self._agent_ids
            }
        )
        
        
        
        

        
        # self.action_space = MultiDiscrete([retailer_capacity + 1,price_capacity+1])  #[order , price]
        # self.observation_space = MultiDiscrete([factory_capacity*2+1,factory_capacity*2+1,factory_capacity*2+1]) # [inventory,backlog,stockout]
        self.periods = 30 
        

        self.r = {1:0, 2:0} 
        # self.info = {1: {'obs': np.array([20,0,0])}, 2: {'obs': np.array([50,0,0])}} 
        
        
        self.info = {1:  {'obs': np.array([self.I[0],self.BL[0],self.SO[0]])}, 2:  {'obs': np.array([self.I[1],self.BL[1],self.SO[1]])} }
        
        self.k = 5 
        self.c = 2 
        self.h = 2 
        self.b = 5
        self.s = 5
        self.lam = 8 
        self.days =  {1: self.periods, 2:self.periods}
        self.day =  {1: 0, 2:0}
        self._seed() 
        self.dist = 1
        self.ALPHA = 0.5
        self.dist_param = {'mu':20}
        self.reset() 


    def demand(self):
        user_D = np.zeros(self.days[1])
        distributions = {
            1: {'dist':poisson,'params':{'mu':self.dist_param['mu']}},
            2: {'dist':binom, 'params': {'n':30, 'p':0.5}},
            3: {'dist':randint,'params':{'low':0,'high':100}},
            4: {'dist':geom, 'params': {'p':0.5}},
            5: {'dist':user_D, 'params':None}
        }
        if self.dist <5:
            dist_info = distributions[self.dist]
            dist_instance = dist_info['dist']
            dist_params = dist_info['params']
            if dist_instance is not None:
                return dist_instance.rvs(**dist_params)
            else:
                assert len(user_D) == self.days[1],'the length of the user-specified distribution is not equal to the number of days'
                return user_D
        else:
            assert len(user_D) == self.days[2],'the length of the user-specified distribution is not equal to the number of days'
            return user_D
  
 
    def transition(self, x, a_factory, a_retailer,d): 
        # x = [inventory，backlog,stockout ]
        print('-------X------',x) # {1: array([10,  0,  0]), 2: array([10,  0,  0])}
        # x = np.array(x)
        I = [value[0] for value in x.values()]
        # print('-------I------',I)
        BL = [value[1] for value in x.values()]
        SO = [value[2] for value in x.values()]


        if  0 <= a_factory+I[1] <= self.factory_capacity:  #### blance
            I[1] = a_factory+I[1]
            BL[1] = 0
            SO[1] = 0
            if  0 < a_retailer-I[1]: #### stockout
                SO[1] = a_retailer-I[1]
                I[1] = 0
                BL[1] = 0
        else:
            # self.factory_capacity < a_factory+I[1]: #### backlog
            BL[1] = max((I[1]-self.factory_capacity),0)
            I[1] = np.where(a_factory + I[1] > 2 * self.factory_capacity, 2 * self.factory_capacity, a_factory + I[1])
            SO[1] = 0
            if 0 < a_retailer-I[1]: #### stockout
                SO[1] = a_retailer-I[1]
                I[1] = 0
                BL[1] = 0
        # else:
        #     raise ValueError

        if  0 < a_retailer+I[0] <= self.retailer_capacity:
            I[0] = a_retailer+I[0]
            BL[0] = 0
            SO[0] = 0
            if  0 < d-I[0]:
                SO[0] = d-I[0]
                I[0] = 0
                BL[0] = 0
        else:
            # self.retailer_capacity < a_retailer+I[0]:
            BL[0] = max((I[0]-self.retailer_capacity),0)
            SO[0] = 0
            I[0] = np.where(a_retailer + I[0] > 2 * self.retailer_capacity, 2 * self.retailer_capacity, a_retailer + I[0])
            if 0 < a_retailer-I[1]:
                SO[1] = a_retailer-I[1]
                I[1] = 0
                BL[1] = 0
        # else:
        #     raise ValueError
        self.I = I
        self.BL = BL
        self.SO = SO
       
        x = {1: np.array([I[0],BL[0],SO[0]]), 2: np.array([I[1],BL[1],SO[1]])} 
        x = OrderedDict(x)
        print('-------np.array(x)22222------',x) #{1: array([19,  0,  0]), 2: array([18,  0,  0])}

        # self.state =  x
        return x 
 
    def reward(self, x, a_factory, p_factory,a_retailer,p_retailer,d, y): 
        k = self.k 
        c = self.c 
        h = self.h 
        print( a_factory, a_retailer) 
 
        factory_reward = -k * (a_factory > 0) - c * max(min(x[2][0] + a_factory, self.factory_capacity) - x[2][0], 0)-h * x[2][0] -self.b*x[2][1]-self.s*x[2][2]+ p_factory * max(min(x[2][0] + a_retailer, self.retailer_capacity) - y[2][0], 0) 
        self.r[2] = factory_reward
        # print('factory_reward',factory_reward) 
        # print('p_factory',p_factory) 
        self.day[1] += 1
 
        retailer_reward = -k * (a_retailer > 0) - p_factory * max(min(x[1][0] + a_retailer, self.retailer_capacity) - x[1][0], 0)-h * x[1][0] -self.b*x[1][1]-self.s*x[1][2]  + p_retailer * max(min(x[1][0] + d, self.retailer_capacity) - y[1][0], 0) 
        self.r[1] = retailer_reward

 
        # print('retailer_reward',retailer_reward) 
        # print('p_retailer',p_retailer) 
        self.day[2] += 1
        
        
        # print('self.day',self.day) 
        return {1: self.r[1], 2: self.r[2]} 
 
    def _seed(self, seed=None): 
        
 
        self.np_random, seed = seeding.np_random(seed) 
 
        return [seed] 
 
    def step(self, actions): 
        agent_ids = actions.keys() 
        print('--------actions---------',actions)# {1: array([12,  8]), 2: array([8, 6])}
        obs_dict = self.state.copy() 
    
 
        demand = self.demand() 
 
        observations_dict = self.transition(obs_dict, actions[1][0], actions[2][0], demand) 

 
        
 
        self.state = observations_dict.copy() 

        # state = OrderedDict(self.state)

        self.info = {1: {'obs': self.state[1]}, 2: {'obs': self.state[2]}} 
 
         
 
        rewards = self.reward(obs_dict, actions[1][0], actions[2][0], actions[1][1], actions[2][1], demand,observations_dict) 
        processed_rewards = self.get_processed_reward(rewards)
 
     
 
        done = {i: self.is_done(i) for i in agent_ids} 
 
 
 
 
        done["__all__"] = all(done.values()) 

        

 
        # print('observations_dict',observations_dict) 
 
        # print('rewards',rewards) 
 
        # print('done',done) 
 
        print('self.info',self.info) 
 
  
 
        return self.state, processed_rewards, done, self.info 
    

    def get_processed_reward(self,rewards):
        processed_rewards = {}
        rewards_list = list(rewards.values())
        mean_reward = np.mean(rewards_list)
        for key, value in rewards.items():
            processed_rewards[key] =self.ALPHA * value + (1 - self.ALPHA) * mean_reward
        return processed_rewards
    
    
    def is_done(self, agent_id): 
            
        if self.day[agent_id] >= self.days[agent_id]:
            done = True
        else:
            done = False
 
        return done 
 
 
 
    def reset(self): 
        self.day =  {1: 0, 2:0}
        self.r =  {1:0, 2: 0} 
        self.I = [self.retailer_capacity,self.retailer_capacity]
        self.BL = [0,0]
        self.SO = [0,0] 
        state = {1: np.array([self.I[0],self.BL[0],self.SO[0]]), 2:  np.array([self.I[1],self.BL[1],self.SO[1]])}
        self.state = OrderedDict(state)
        self.info = {1:  {'obs': np.array([self.I[0],self.BL[0],self.SO[0]])}, 2:  {'obs': np.array([self.I[1],self.BL[1],self.SO[1]])} }


        # a_factory = 0
        # p_factory = 0
        # a_retailer = 0
        # p_retailer = 0
        # d = 2
     
        # self.reward(x, a_factory, p_factory,a_retailer,p_retailer,d, y)

        # print('----tttttt-----',self.transition(x, a_factory, a_retailer,d) )
        # return self.transition(self.state,a_factory, a_retailer,d) 
        return self.state
       
 
  
 
 
  
 
if __name__ == '__main__': 
    register_env("MultiSupplyEnv", lambda config: MultiAgent1W1F())

    config = PPOConfig()
    # .rollouts(num_rollout_workers=2, create_env_on_local_worker=True))

    config = config.environment(env="MultiAgent1W1F")

    test_env = MultiAgent1W1F()

    # player_roles = ['1','2']

    # policies = {}
    # for role in player_roles:
    #     policies[role] = PolicySpec(
    #         observation_space=test_env.observation_space,
    #         action_space=test_env.action_space,
    #     )
    # config = config.multi_agent(
    #     policies=policies,
    #     policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
    # )

    pretty_print(config.to_dict())

    algo = config.build()

    for i in range(10):
        result = algo.train()

    print(pretty_print(result))
 
 
 
  
 
    # env = MultiAgent1W1F() 
    
    
    # search_space = {
    #     'weight': tune.uniform(0,1),
    #     'discount_factory':tune.uniform(0, 1)}
    
    
  
    # a=  {1: env.action_space.sample(), 2: env.action_space.sample()} 
    # print('a----------------',a) # {1: array([4, 5], dtype=int32), 2: array([16,  1], dtype=int32)} ,order and price
 
    # OBS= {1: env.observation_space.sample(), 2: env.observation_space.sample()} 
 
    # print('OBS----------------',OBS) 
 
    # obs, rew, done, info = env.step(a) 
 
    # print('obs------',obs) 
 
    # print('rew------',rew) 
 
     
 
    # # # while True: 
    
    # # episides = 60
    # # for i in range(episides): 
    # #     obs = env.reset()
    # #     while True:
    # #         print('+++++++++++++++++++++++++',i,'++++++++++++++++++++++++++')

    # #         obs, rew, done, info = env.step(env.action_space.sample()  ) 
    # #         if done :
    # #             break
    # # assert done

    # # print('obs',obs) 

    # # print('rew',rew) 

    # # print('done',done) 

    # # print('info',info) 


    # # print('rew------') 
    
    
    
 
    # config = (PPOConfig().environment(MultiAgent1W1F)
    #           .rollouts(num_rollout_workers=2,
    #                     create_env_on_local_worker=True))
    
    # pretty_print(config.to_dict())
    
    # algo = config.build()
    # for i in range(10):
    #     result = algo.train()
        
    # pretty_print(result)
 
    # print('rew------111') 
    
   
    # print('rew------222') 
    
    # checkpoint = algo.save()
    # print(checkpoint)
    # evaluation = algo.evaluate()
    # print(pretty_print(evaluation))
    
    # algo.stop()
    # restored_algo = Algorithm.from_checkpoint(checkpoint)
    
    # print('rew------333') 
    
    # simple_trainer = DQNConfig().environment(env=MultiAgent1W1F).build() 
    # for i in range(20):
        
 
    #     result = simple_trainer.train() 
    
    #     print(pretty_print(result))
    
    
    
 
    
    
    
    # checkpoint =  algo.evaluate()
    
 
     
 
    # algo = PPOConfig().environment(env=MultiAgent1W1F).multi_agent( 
    #         policies={ 
 
    #             "policy_1": ( 
 
    #                 None, env.observation_space, env.action_space, {"gamma": 0.80} 
 
    #             ), 
 
    #             "policy_2": ( 
 
    #                 None, env.observation_space, env.action_space, {"gamma": 0.95} 
 
    #             ), 
 
    #         }, 
 
    #         policy_mapping_fn = lambda agent_id:
    #     f"policy_{agent_id}", 
    #     ).build() 
 
     
 
    # print(algo.train()) 
 
    print('rew------333') 
    