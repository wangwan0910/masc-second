#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    
"""


@author: wangwan
"""
import gym
from gym import spaces
from gym.spaces import Box
import math
import numpy as np
import itertools
from itertools import chain
import collections






# Alexandera Environment Class

class State:
    def __init__(self,inventory,reorder_point,pendingDelivery,daysToDelivery,
                 stockouts,processingTimes,demand_history,T=30,t=0):
        # self.inventory = np.array([ 0, 0, 10],dtype=np.float64)
        # self.reorder_point=np.array([ 0, 0, 5],dtype=np.float64)
        # self.pendingDelivery =np.array([ 0, 0, 0],dtype=np.float64)
        # self.daysToDelivery = np.array([ 0, 0, 0],dtype=np.float64)
        # self.stockouts = np.array([ 0, 0, 0],dtype=np.float64)
        # self.processingTimes = np.array([ 0, 0, 0],dtype=np.float64)
        # self.T=T
        # self.demand_history = demand_history
        # self.t = 0
        self.inventory = inventory
        self.reorder_point = reorder_point
        self.pendingDelivery = pendingDelivery
        self.daysToDelivery = daysToDelivery
        self.stockouts = stockouts
        self.processingTimes = processingTimes
        self.demand_history = demand_history
        self.T=T
        self.t = t
        
    
    def to_array(self):
        state=[self.inventory,self.reorder_point,
               self.pendingDelivery,self.daysToDelivery,
               self.stockouts,
               self.processingTimes]
        state=np.array(state)
        
        return state
    
    # def update_state(self, action_obj):
    #    self.x = action_obj[0]
    #    self.y = action_obj[1]
    #    self.z = action_obj[2]
    
   
        

        

class Action:
    def __init__(self,serviceTime,orderToSupplier,reorderPoint):
        
        # self.serviceTime = np.array([ 0, 0, 0],dtype=np.float64)
        # self.orderToSupplier = np.array([ 0, 0, 0],dtype=np.float64)
        # self.reorderPoint = np.array([ 0, 0, 0],dtype=np.float64)
        self.serviceTime = serviceTime
        self.orderToSupplier = orderToSupplier
        self.reorderPoint = reorderPoint
        
    
    def to_array(self):
       
        action=[self.serviceTime,self.orderToSupplier,
               self.reorderPoint]
        
        action=np.array(action)
        
        return action
   
        
        
        

class Environment():
    def __init__(self):
        self.inventory = np.array([ 0, 0, 10],dtype=np.float64)
        self.reorder_point=np.array([ 0, 0, 5],dtype=np.float64)
        self.pendingDelivery =np.array([ 0, 0, 0],dtype=np.float64)
        self.daysToDelivery = np.array([ 0, 0, 0],dtype=np.float64)
        self.stockouts = np.array([ 0, 0, 0],dtype=np.float64)
        self.processingTimes = np.array([ 1, 3, 0],dtype=np.float64)
        self.inventoryCost=np.array([1000, 5, 1000])
        self.muDemand=2
        self.stdDemand=0.1
        self.retailerOrder=10
        self.stockoutCost=10000
        self.N=1000
        self.T =31 
        self.max = 30
        self.serviceTime = np.array([ 0, 0, 0],dtype=np.float64)
        self.orderToSupplier = np.array([ 0, 0, 0],dtype=np.float64)
        self.reorderPoint = np.array([ 0, 0, 0],dtype=np.float64)
        self.reset()
        self.price = 100
      
        
     
    def reset(self, demand_history_len=3):     
       # (five) demand values observed
       self.demand_history = collections.deque(maxlen=demand_history_len)
       for d in range(demand_history_len):
          self.demand_history.append(self.demand())
       self.t = 0 
       # return self.demand_history 
       
 
   
    def demand(self):
         
         demand = max(0, np.random.normal(self.muDemand, self.stdDemand))
         # make sure demand is integer
         demand = np.floor(demand)
         
         return demand
     
    def initial_state(self):
        
        return State(self.inventory,self.reorder_point,self.pendingDelivery,self.daysToDelivery,
                     self.stockouts,self.processingTimes,list(self.demand_history))

       
    def initial_action(self):
       
       return Action(self.serviceTime, self.orderToSupplier, self.reorderPoint)
      


    def step(self, state, action):
        
        
        newState = state.copy()
        # check if daysToDelivery has been fulfilled
        for i in range(3):
            # check if waiting for any delivery
            if newState[3,i] > 0:
                # subtract daysToDelivery
                newState[3,i] -= 1

            # fulfill order
            if newState[3,i] == 0:
                # reduce from supplier when inventory is 0
                newState[0,i] += newState[2,i]
                # cut supplier's inventory
                if i > 0:
                    newState[0,i-1] -= newState[2,i]

                # set pending delivery to 0
                newState[2,i] = 0
            
                
                
         # check for stockout
        retailerState = newState[:, 2]
     
        if retailerState[0] >= self.demand():
         # consume inventory
            retailerState[0] -= self.demand()
        else:
         # add to stockout
         # print(retailerState[4])
         # print(demand[0][0])
            retailerState[4] +=self.demand() - retailerState[0]
         # set inventory to zero
            retailerState[0] = 0
            
                
            """
            execute action if there's any
            """
            # def execute(self, state, action):
                
            # new_s = State(self.inventory,self.reorder_point,self.T,
            #                    self.pendingDelivery,self.daysToDelivery,
            #                      self.stockouts,self.processingTimes,
            #                      list(self.demand_history))
            
            # newState = new_s.to_array()

        
        # check for stockout
        # retailerState = newState[:, 2]

        # trigger for aking action: retailer's inventory < reorderPoint & not waiting for any delivery
        actionTrigger = (retailerState[0] <= retailerState[1]) & (retailerState[3] == 0)

        # get rewards
        # include state[4] accumulated number of stockouts + state[0] long-term inventory (after delivered to customer)
        # e.g. just before the next reorderPoint
        reward = 0.0
        
        reward += np.dot(self.price , np.sum(newState[2], axis=0))  # revenue
                 
        print('reward',reward)
        
        reward -= newState[4,2] * self.stockoutCost  # total stockouts this period x stockoutPrice
        # reward -= ( newState[2,1]-min(newState[0,2] + newAction[1,2],m) )* self.stockoutCost
        # reward -= (newState[0,0] * self.inventoryCost[0] + newState[0,1] * self.inventoryCost[1])
        # inventory
        print('reward1',reward)
        reward -= (max(min(newState[0,0] + newAction[1,0],m) - newState[2,1],0) * self.inventoryCost[0] + 
                   max(min(newState[0,1] + newAction[1,1],m) - newState[2,2],0) * self.inventoryCost[1])  # inventory

        print('reward2',reward)
        reward -=  max(min(newState[0,2] + newAction[1,2],m) - self.demand(),0) * self.inventoryCost[2] # include unused safety stock at retailer

        
        print('reward3',reward)
        # reward = - newState[4, 2] * self.stockoutCost  - (newState[0, 0] * self.inventoryCost[0] + newState[0, 1] * self.inventoryCost[1])  - newState[0, 2] * self.inventoryCost[2] # include unused safety stock at retailer
        # float(reward)
        self.t += 1
        
        retailerState = newState[:, 2]
        s1State = newState[:, 1]
        s1Action = newAction[:, 1]
        s0State = newState[:, 0]
        s0Action = newAction[:, 0]

        # pendingDelivery = ordered items
        newState[2] = newAction[1]

        # daysToDelivery = serviceTimes + processingTimes
        s0State[3] = 0 + s0State[5]  # s0 has 0 supplier serviceTime
        s1State[3] = s0Action[0] + s1State[5]
        retailerState[3] = s1Action[0]

        # set next reorder point
        newState[1, 2] = newAction[2, 2]

        # reset stockout
        newState[4] = 0
        # actionTrigger
        return (newState,reward, self.t==self.T-1)
    
    
    
    
    
    
    
        new_s = State(self.inventory,self.reorder_point,
                           self.pendingDelivery,self.daysToDelivery,
                             self.stockouts,self.processingTimes,
                             list(self.demand_history))
        
        newState = new_s.to_array().copy()
        # print(newState)
        
        
        
        new_a = Action(self.serviceTime, self.orderToSupplier, 
                       self.reorderPoint)
        
        newAction = new_a.to_array().copy()
        m = self.max
    
    # state[0] =  max(min(newState[0] + newAction[1],m) - newState[0],0)
    # newState[0]= max(min(newState[0] + newAction[1],m) - newState[0],0)
    
    # newAction[1] = newAction + - self.demand()
    
   #  """
   #  natural state update
   #  """
   #  def step(self, state, demand):
       
   #      # get newState
   #      newState = State(self.inventory,self.reorder_point,self.T,
   #                         self.pendingDelivery,self.daysToDelivery,
   #                           self.stockouts,self.processingTimes,
   #                           list(self.demand_history))
        
        

   #      # check if daysToDelivery has been fulfilled
   #      for i in range(3):
   #          # check if waiting for any delivery
   #          if newState.daysToDelivery[i] > 0:
   #              # subtract daysToDelivery
   #              newState.daysToDelivery[i] -= 1

   #          # fulfill order
   #          if newState.daysToDelivery[i] == 0:
   #              # reduce from supplier when inventory is 0
   #              newState.inventory[i] += newState.pendingDelivery[i]
   #              # cut supplier's inventory
   #              if i > 0:
   #                  newState.inventory[i-1] -= newState.pendingDelivery[i]

   #              # set pending delivery to 0
   #              newState.pendingDelivery[i] = 0

   #      # check for stockout
   #      # retailerState = newState[:, 2]
        
   #      if newState.inventory[2] >= demand[0]:
   #          # consume inventory
   #          newState.inventory[2] -= demand[0]
   #      else:
   #          # add to stockout
   #          # print(retailerState[4])
   #          # print(demand[0][0])
   #          newState.stockouts[2] += demand[0]- newState.inventory[2]
   #          # set inventory to zero
   #          newState.inventory[2] = 0
            
   #      # if retailerState[0].any(lambda x: x >= demand):
   #      #     print('yes')
   #      # else:
   #      #     print('no')
   # # do something

   #      # trigger for aking action: retailer's inventory < reorderPoint & not waiting for any delivery
   #      actionTrigger = (newState.inventory[2] <= newState.reorder_point[2]) & (newState.daysToDelivery[2] == 0)
     
   #      # get rewards
   #      # include state[4] accumulated number of stockouts + state[0] long-term inventory (after delivered to customer)
   #      # e.g. just before the next reorderPoint
   #      reward = 0.0
   #      reward -= newState.stockouts[2] * self.stockoutCost  # total stockouts this period x stockoutPrice
   #      reward -= (newState.inventory[0] * self.inventoryCost[0] + newState.inventory[1] * self.inventoryCost[1])  # inventory
   #      reward -= newState.inventory[2] * self.inventoryCost[2] # include unused safety stock at retailer
   #      # reward = - newState[4, 2] * self.stockoutCost  - (newState[0, 0] * self.inventoryCost[0] + newState[0, 1] * self.inventoryCost[1])  - newState[0, 2] * self.inventoryCost[2] # include unused safety stock at retailer
   #      # float(reward)
   #      self.t += 1
        
        
   #      # done = False
   #      # newState=state
        
   #      return (newState,actionTrigger,reward, self.t==self.T-1)#(3,)
    
    #self.t == self.T-1
    
  
    
    

    
    
    
#Supply Chain Gym Wrapper
class SupplyChain(gym.Env):
    
    def __init__(self):
        self.reset()
        self.inventory = self.supply_chain.inventory
        self.reorder_point=self.supply_chain.reorder_point
        self.pendingDelivery =self.supply_chain.pendingDelivery
        self.daysToDelivery = self.supply_chain.daysToDelivery
        self.stockouts = self.supply_chain.stockouts
        self.processingTimes = self.supply_chain.processingTimes
        
        self.action  = self.supply_chain.initial_action().to_array()
        self.serviceTime = self.supply_chain.serviceTime
        self.orderToSupplier = self.supply_chain.orderToSupplier
        self.reorderPoint = self.supply_chain.reorderPoint
        
        
        
        
        low_act = [list(self.serviceTime),list(self.orderToSupplier),list(self.reorderPoint)]
        low_act = [item for sublist in low_act for item in sublist]
        low_act = np.array(low_act)
        print(low_act) 

   

        high_act = [list(self.serviceTime),list(self.orderToSupplier+10),list(self.reorderPoint)]
        high_act = [item for sublist in high_act for item in sublist]
        high_act[2]=10
        high_act = np.array(high_act)
        print(high_act) 


   
        self.action_space = spaces.Box(low=low_act,
                                         high=high_act,
                                         dtype=np.int32)
        
        
        low_obs = [list(self.inventory),list(self.reorder_point),list(self.pendingDelivery),
           list(self.daysToDelivery),list(self.stockouts),list(self.processingTimes)]
        low_obs = [item for sublist in low_obs for item in sublist]
        

        low_obs = np.array(low_obs)
        print(low_obs)
        low_obs = np.reshape(low_obs, (6,3))
        
        high_obs = [list(self.inventory),list(self.reorder_point),list(self.pendingDelivery),
           list(self.daysToDelivery),list(self.stockouts),list(self.processingTimes)]
        high_obs = [item for sublist in high_obs for item in sublist]
        high_obs[:len(self.inventory)]=[30,30,30]
        high_obs[3:6] = [20,20,20]
        high_obs[6:9] = [20,20,20]
        high_obs[9:12] = [1,3,0]
        high_obs[12:15] = [300,300,300]
        high_obs[15:18] = [1,3,0]
        high_obs = np.array(high_obs)
        print(high_obs) 
        high_obs = np.reshape(high_obs, (6,3))

        self.observation_space = spaces.Box(low=low_obs,
                                        high=high_obs,
                                        dtype=np.int32)
        
        # low_act = low_obs = np.zeros(
        #     (len(self.supply_chain.initial_action().to_array()),),
        #     dtype=np.int32)
        # # high values for action space
        # high_act = np.zeros(
        #     (len(self.supply_chain.initial_action().to_array()),),
        #     dtype=np.int32)
        # high_act[1] = 10*self.action[1]
        # # high values for action space (factory)
        # high_act[0:2] = 20
        # high values for action space (distribution warehouses, according to
        # action space
     
        
        

    # # # # low values for observation space
    #     low_obs = np.zeros(
    #         (len(self.supply_chain.initial_state().to_array()),),
    #         dtype=np.int32)
    #     # low values for observation space (factory, worst case scenario in
    #     # case of non-production and maximum demand)
    #     low_obs[5,1] =1
    #     # low values for observation space (distribution warehouses, worst case
    #     # scenario in case of non-shipments and maximum demand)
    #     low_obs[5,2]=3
    #     high_obs = np.zeros(
    #         (len(self.supply_chain.initial_state().to_array()),),
    #         dtype=np.int32)
    #     # high values for observation space (factory and distribution
    #     # warehouses, according to storage capacities)
    #     high_obs[6] = self.supply_chain.max
    #     # high values for observation space (demand, according to the maximum
    #     # demand value)
        
      
            

  
    

        # # self.observation_space = spaces.Box(low=np.full(3 * self.retailerOrder + 1, -np.inf),
        # #                                 high=np.full(3* self.retailerOrder*[np.inf, np.inf],2* self.retailerOrder * [np.inf, np.inf]), dtype=np.int32)

        # self.observation_space = spaces.Box(high=self.obs_high, low=self.obs_low)
        # # self.observation_space=spaces.Box(low=np.array([0,0], high=np.array([np.inf,np.inf])), shape=(24,),
        # #                                            dtype=np.float32) 
        # self.observation_space = spaces.Box(low=np.zeros((7,3)), high=np.zeros((7,3)),
        #                                 shape=(7,3),dtype=np.float32)

    
    
    
    def reset(self):
        self.supply_chain = Environment()
        self.state=self.supply_chain.initial_state().to_array()
        # self.action = self.supply_chain.initial_action()
        return self.state
    

    
    
    
    def step(self, action):
        new_a = Action(self.serviceTime, self.orderToSupplier, 
                       self.reorderPoint)
        action_obj = new_a.to_array()
        action_obj[1] = action[1]
        #pendingDelivery = ordered items
        self.state[2] = action_obj[1]
        action_obj[2] = action[2]
        #set next reorder point
        self.state[1,2] = action_obj[2,2]
        
        #daysToDelivery = serviceTimes + processingTimes
        
        retailerState = self.state[:, 2]
        s1State = self.state[:, 1]
        s1Action = action_obj[:, 1]
        s0State = self.state[:, 0]
        s0Action = action_obj[:, 0]

      

        # daysToDelivery = serviceTimes + processingTimes
        s0State[3] = 0 + s0State[5]  # s0 has 0 supplier serviceTime
        s1State[3] = s0Action[0] + s1State[5]
        retailerState[3] = s1Action[0]

        
        
        # reset stockout
        self.state[4] = 0
        # done = False
       
        self.state,reward, done = self.supply_chain.step(
            self.state, action_obj)
        # self.demand = self.supply_chain.demand()
        # self.demand = np.zeros(self.demand)
        self.demand  = self.supply_chain.demand()
        # info = "Demand was: ", self.demand
       
        info={}
        return self.state, reward, done, info
        
    

        
        
       
    
    
    

    
    

        
    
    
    
    