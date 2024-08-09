 
 
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
The observation space for each agent is the [inventory,backlog, stockout] and the action space is the number of [orders,price].
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
from ray.tune.registry import register_env
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.rl import RLTrainer, RLCheckpoint, RLPredictor
import gymnasium as gym 
 
from gym.spaces import Discrete,MultiDiscrete 
 
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
import ray



class MultiAgent1W1F(MultiAgentEnv): 
 
  
 
    def __init__(self,  *args, **kwargs):#max_inventory=20, factory_capacity=100, k=5, c=2, h=2, p=3, lam=8 
        self.factory_capacity = 100 
        self.retailer_capacity = 10 
        self.price_capacity = 10
        factory_capacity = self.factory_capacity 
        retailer_capacity = self.retailer_capacity 
        price_capacity = self.price_capacity 
        self.action_space = MultiDiscrete([retailer_capacity + 1,price_capacity+1])  #[order , price]
        self.observation_space = MultiDiscrete([factory_capacity*2+1,factory_capacity*2+1,factory_capacity*2+1]) # [inventory,backlog,stockout]
        self.periods = 30 
        self.state =  {1:np.array([10,0,0]), 2: np.array([10,0,0])}  
        self.r = {1:0, 2:0} 
        self.info = {1: {'obs': np.array([20,0,0])}, 2: {'obs': np.array([50,0,0])}} 
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
        # print('-------X------',x)
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

       
        x = {1: np.array([I[0],BL[0],SO[0]]), 2: np.array([I[1],BL[1],SO[1]])} 
        # print('-------np.array(x)22222------',x)

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
        # print('--------actions---------',actions)# {1: array([12,  8]), 2: array([8, 6])}
        obs_dict = self.state.copy() 
    
 
        demand = self.demand() 
 
        observations_dict = self.transition(obs_dict, actions[1][0], actions[2][0], demand) 
 
        
 
        self.state = observations_dict 

        self.info = {1: {'obs': self.state[1]}, 2: {'obs': self.state[2]}} 
 
         
 
        rewards = self.reward(obs_dict, actions[1][0], actions[2][0], actions[1][1], actions[2][1], demand,observations_dict) 
        processed_rewards = self.get_processed_reward(rewards)
 
     
 
        done = {i: self.is_done(i) for i in agent_ids} 
 
 
 
 
        done["__all__"] = all(done.values()) 

        

 
        # print('observations_dict',observations_dict) 
 
        # print('rewards',rewards) 
 
        # print('done',done) 
 
        # print('self.info',self.info) 
 
  
 
        return observations_dict, processed_rewards, done, self.info 
    

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
  
 
 
 
  
 
    env = MultiAgent1W1F() 
    
    
    # search_space = {
    #     'weight': tune.uniform(0,1),
    #     'discount_factory':tune.uniform(0, 1)}
    
    
  
    # a= {1: env.action_space.sample(), 2: env.action_space.sample()} 
    # print('a----------------',a) # {1: array([4, 5], dtype=int32), 2: array([16,  1], dtype=int32)} ,order and price
 
    # OBS= {1: env.observation_space.sample(), 2: env.observation_space.sample()} 
 
    # print('OBS----------------',OBS) 
 
    # obs, rew, done, info = env.step(a) 
 
    # print('obs------',obs) 
 
    # print('rew------',rew) 
 
     
 
    # # while True: 
    
    # episides = 60
    # for i in range(episides): 
    #     obs = env.reset()
    #     while True:
    #         print('+++++++++++++++++++++++++',i,'++++++++++++++++++++++++++')

    #         obs, rew, done, info = env.step( 

    #             {1: env.action_space.sample(), 2: env.action_space.sample()} 

    #         ) 
    #         if done :
    #             break
    # assert done

    # print('obs',obs) 

    # print('rew',rew) 

    # print('done',done) 

    # print('info',info) 


    # print('rew------') 
    
    
    
 
    config = (PPOConfig()
              .environment(MultiAgent1W1F)
              .rollouts(num_rollout_workers=2,
                        create_env_on_local_worker=True)
                )
    
    # pretty_print(config.to_dict())
    
    algo = config.build()
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
    # checkpoint_path= "envs/save_models/"
    # rl_checkpoint = RLCheckpoint(local_path = checkpoint_path)
    # predictor =  RLPredictor.from_checkpoint(checkpoint=rl_checkpoint)
    # print(checkpoint_path)
    
    # save_dir = "envs/save_models"
    # # Set to True if you want to load a checkpoint
    # USE_CHECKPOINT = False #@param {type:"boolean"}
    # def sc_env_creator(env_config):
    #     return MultiAgent1W1F(env_config)
    # register_env("MultiAgent1W1F", sc_env_creator)

    # if ray.is_initialized():
    #     ray.shutdown()
    #     ray.init(include_dashboard=True, ignore_reinit_error=True, logging_level="info", 
    #             log_to_driver=False, object_store_memory=0.2*10**9) 
    #     #object_store_memory=5*10**9  local_mode=True, num_gpus=1
    # ray.available_resources()

    # checkpoint_path = "envs/save_models/AIRPPO_2024-05-13_21-32-27/AIRPPO_e9f20_00000_0_2024-05-13_21-32-27/checkpoint_000001"
    # rl_checkpoint = None
    # if USE_CHECKPOINT:
    #     rl_checkpoint = RLCheckpoint(local_path = checkpoint_path)
    # #rl_checkpoint = ray.put(rl_checkpoint)

    # config_cf = {
        
    #     #
    #     # Maximize performance according to hardware
    #     # https://docs.ray.io/en/latest/rllib/rllib-training.html
    #     #MultiAgent1W1F

    #     # Depending on what you want to train, you might want to reduce the numbers
    #     # for train_batch_size, sgd_minibatch_size, etc. with the increment of the
    #     # observation_shape, as it will consume a lot of VRAM and RAM, and decrease
    #     # the num_envs_per_worker. As it is configured right now, it should work for
    #     # the experiemnt 6 variant, 10x10(x3) observation space.

    #     # Can handle multiple workers if you split CPU
    #     # E.g. Colab, num_workers 8 + 2 evaluation_num_workers = 10
    #     # Split 2.0 cores in 0.2 in run config
    #     # I use only 2 with multiple num_envs_per_worker and disabled eval
    #     "num_workers": 2,
    #     #"simple_optimizer": True,

    #     #
    #     # Number of parallel workers to use for evaluation. 
    #     #

    #     # Experiment 1 / 2 / 3 / 4
    #     # "evaluation_num_workers": 1, 
    #     # Experiment 5 / 6
    #     "evaluation_num_workers": 0, 

    #     #
    #     # Number of environments to evaluate vector-wise per worker. 
    #     # This enables model inference batching, which can improve performance for inference bottlenecked workloads.
    #     #
    #     # A standard Colab instance would have 2 CPUs and 1 GPU, so we split that to "num_workers" + evaluation_num_workers
    #     #

    #     #  MultiAgent1W1F
    #     # "num_envs_per_worker": 10,
    #     # Experiment 1 / 2 / 3 / 4
    #     # "num_envs_per_worker": 1,
    #     # Experiment 5 / 6 / 7
    #     "num_envs_per_worker": 20,
    #     "num_cpus_for_driver": 0,
    #     # Defined also in scaling_config, so commented here
    #     #"num_cpus_per_worker": 0.25, 
    #     #"num_gpus_per_worker": 0.125, 

    #     #
    #     # Ray will use the MultiAgent1W1F environment we defined.
    #     #

    #     "env": "MultiAgent1W1F",

    #     #
    #     # The default learning rate.
    #     #

    #     # Experiment 1
    #     # "lr": 0.001, 
    #     # Experiment 2 / 3
    #     # "lr": 0.003,
    #     # Experiment 4 / 5 / 6
    #     "lr": 0.0003,

    #     #
    #     # Float specifying the discount factor of the Markov Decision process.
    #     #

    #     # Experiment 1 / 2 / 3
    #     # "gamma": 0.97, 
    #     # Experiment 4 / 5 / 6
    #     "gamma": 0.99,

    #     #
    #     # Initial coefficient for KL divergence.
    #     #

    #     # Experiment 1 / 2 / 3
    #     # "kl_coeff": 0.2, 
    #     # Experiment 4 / 5 / 6
    #     "kl_coeff": 1.0,

    #     #
    #     # The GAE (lambda) parameter.
    #     #
        
    #     # Experiment 1 / 2 / 3
    #     # "lambda": 0.99,
    #     # Experiment 4 / 5 / 6
    #     "lambda": 0.95,

    #     #
    #     #  Target value for KL divergence.
    #     #

    #     "kl_target": 0.03,

    #     #
    #     # Coefficient of the entropy regularizer.
    #     #

    #     # Experiment 1
    #     # "entropy_coeff": 0.03,
    #     # Experiment 2 / 3
    #     # "entropy_coeff": 0.1,
    #     # Experiment 4
    #     # Default 0.0
    #     # Experiment 5 / 6
    #     "entropy_coeff": 0.1,

    #     #
    #     # Learning rate schedule. 
    #     # In the format of [[timestep, lr-value], [timestep, lr-value], …] 
    #     # Intermediary timesteps will be assigned to interpolated learning rate values. 
    #     # A schedule should normally start from timestep 0.
    #     #
        
    #     # Experiment 1
    #     # "lr_schedule": [[50000, 0.0005], [100000, 0.0001]],
    #     # Experiment 2
    #     # "lr_schedule": [[20000, 0.001], [50000, 0.0005], [100000, 0.0001], [150000, 0.00005], [200000, 0.00001]],
    #     # Experiment 3
    #     # "lr_schedule": [[25000, 0.001], [50000, 0.0008], [75000, 0.0006], [100000, 0.0004], [125000, 0.0002],[150000, 0.0001],[175000, 0.00005],[200000,0.00001]],
    #     # Experiment 4 
    #     # Disable schedule
    #     # Experiment 5 / 6
    #     "lr_schedule": [[500000, 0.0001]],

    #     #
    #     # Decay schedule for the entropy regularizer.
    #     #

    #     # Experiment 1
    #     # "entropy_coeff_schedule": [[50000, 0.03], [75000, 0.01]],
    #     # Experiment 2
    #     # "entropy_coeff_schedule": [[40000, 0.05], [80000, 0.03], [120000, 0.01]],
    #     # Experiment 3
    #     # "entropy_coeff_schedule": [[40000, 0.08], [80000, 0.06], [120000, 0.04], [160000, 0.02], [200000, 0.01]],
    #     # Experiment 4
    #     # Disable schedule
    #     # Experiment 5 / 6
    #     "entropy_coeff_schedule": [[200000, 0.01], [500000, 0.001], [800000, 0.0001], [1100000, 0.00001]],

    #     #
    #     # The max. number of `step()`s for any episode (per agent) before
    #     # Horizon might limit how much score the MultiAgent1W1F can make after a certain threshold.
    #     # If you find horizon too low, you should increase it. Read the RLlib docs.
    #     #

    #     # MultiAgent1W1F
    #     # "horizon":  2000,
    #     # Experiment 1 / 2 / 3
    #     # "horizon":  800,
    #     # Experiment 4 / 5
    #     # "horizon":  1200,
    #     # Experiment 6
    #     "horizon":  16000,

    #     #
    #     # Using PyTorch as ML framework
    #     #

    #     "framework": "torch", 

    #     # MultiAgent1W1F
    #     # "train_batch_size":  2000,
    #     # Experiment 1 / 2 / 3 / 4 / 5
    #     # "train_batch_size": 800,
    #     # Experiment 6
    #     "train_batch_size": 16000,

    #     #
    #     # Total SGD batch size across all devices for SGD. This defines the minibatch size within each epoch.
    #     #

    #     # Experiment 1 / 2 / 3 / 4 / 5
    #     # "sgd_minibatch_size": 256,
    #     # Experiment 6
    #     "sgd_minibatch_size": 200,

    #     #
    #     # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
    #     #

    #     # Experiment 1 / 2 / 3 / 4
    #     # "num_sgd_iter": 30,
    #     # Experiment 5 / 6
    #     "num_sgd_iter": 20,

    #     "vf_share_layers": True,

    #     #
    #     # Coefficient of the value function loss. 
    #     # IMPORTANT: you must tune this if you set vf_share_layers=True inside your model’s config.
    #     #

    #     # Experiment 1 / 2 / 3
    #     # "vf_loss_coeff": 1.0,
    #     # Experiment 4 / 5 / 6
    #     "vf_loss_coeff": 0.5,
    #     "clip_rewards": True,

    #     #
    #     # PPO clip parameter.
    #     #

    #     # Experiment 1 / 2 / 3
    #     # "clip_param": 0.3,
    #     # Experiment 4 / 5 / 6
    #     "clip_param": 0.2,

    #     #
    #     # Clip param for the value function. 
    #     # Note that this is sensitive to the scale of the rewards. If your expected V is large, increase this. 
    #     #

    #     "vf_clip_param": 10.0,

    #     #
    #     # This will affect how the rollout will be handled. "auto"
    #     # https://docs.ray.io/en/latest/rllib/rllib-sample-collection.html
    #     #

    #     "rollout_fragment_length": "auto",
    #     "batch_mode": "truncate_episodes",


    #     # 
    #     # Here's the CNN and lstm model definition. You might want to adjust this if you want experimenting.
    #     # Make sure it's compatible with your output resolution from the MultiAgent1W1F game class above.
    #     # Ray will get you an error if the dimension mismatch between layers.
    #     # https://docs.ray.io/en/latest/rllib/rllib-models.html#default-model-config-settings
    #     #

    #     #  MultiAgent1W1F
    #     # Depending on the observation_shape, example models:
    #     # "model": {"dim": 224, "conv_filters": [[16, [3, 3], 2], [32, [3, 3], 3], [64, [5, 5], 5], [128, [5, 5], 5]], "fcnet_hiddens": [1024, 512, 512], "fcnet_activation": "than", "use_lstm": False},
    #     # "model": {"dim": 168, "conv_filters": [[32, [4, 4], 4], [128, [4, 4], 3], [256, [4, 4], 3], [512, [4, 4], 4]], "fcnet_hiddens": [1024, 512, 512], "fcnet_activation": "than", "use_lstm": True,  "lstm_cell_size": 32, "max_seq_len": 16},
    #     # Experiment 1
    #     # "model": {"dim": 84, "use_lstm": True, "lstm_cell_size": 64, "max_seq_len": 20},
    #     # Experiment 2 / 3 / 4
    #     # "model": {"dim": 84, "use_lstm": False},
    #     # Experiment 5
    #     # "model": {"dim": 42, "fcnet_hiddens": [512 ,256], "use_lstm": False},
    #     # Experiment 6
    #     "model": {"dim": 10, "fcnet_hiddens": [512 ,512], "use_lstm": False},

    #     #
    #     # This will evaluate the model at defined interval * batch_size steps; e.g. every N training_iteration
    #     # Pick a larger evaluation_interval, as it is slow and requires a lot of resources for standard colab
    #     #

    #     # Experiment 1 / 2 / 3 / 4
    #     # "evaluation_interval": 50, 
    #     # "evaluation_config": {'explore': False, "input": "sampler"}, "log_level": "ERROR",
    #     # "evaluation_parallel_to_training": True,
    #     # Experiment 5 / 6  - > 5 * 16.000
    #     "evaluation_interval": 10,
    #     # Fixed - Disable evaluation because it gets stuck in num_envs_per_worker > 1 and not parallel
        

    #     #
    #     # Exploration config is quite important, and you might want to tune this.
    #     #

    #     "explore": True,

    #     #
    #     # EpsilonGreedy strategy chooses a random action with probability epsilon,
    #     # otherwise chooses the action with the highest predicted value according to the learned policy. 
    #     # The value of epsilon starts at 1.0 and is linearly annealed to 0.1 over 10000 timesteps.
    #     #

    #     #"exploration_config": {"type": "EpsilonGreedy", "initial_epsilon": 0.3, "final_epsilon": 0.01, "epsilon_timesteps": 25000},

    #     #
    #     # StochasticSampling randomly samples actions from the action distribution produced by the policy network, 
    #     # rather than selecting the action with the highest predicted value.
    #     #

    #     # Experiment 1 / 2 / 3
    #     # "exploration_config": {"type": "StochasticSampling", "random_timesteps": 20000},
    #     # Experiment 4 / 5 / 6
    #     "exploration_config": {"type": "StochasticSampling", "random_timesteps": 5000},

    #     #
    #     # Curiosity does not support paralelism
    #     #

    #     # "exploration_config": {"type": "Curiosity", "eta": 0.2, "lr": 0.001, "feature_dim": 128, 
    #     #                       "feature_net_config": {"fcnet_hiddens": [],"fcnet_activation": "relu"},
    #     #                       "sub_exploration": {"type": "StochasticSampling"}},
        
    # }

    # #    
    # # Train for 100,000 timesteps.
    # # Should be sufficient for a fast trained (under 30 mins) demo.
    # # Use above 500,000 if you need a proper working model, e.g. 2-5M.
    # #

    # stop_cf = {"timesteps_total":  100}#10000000
            
            
        
    # # checkpoint_path = "saved_models/AIRPPO_2025-04-20_17-11-24/AIRPPO_62dcf_00000_0_2023-04-20_17-11-27/checkpoint_000150"
        
    # run_config = RunConfig(
    #     local_dir=save_dir,
    #     stop=stop_cf,
    #     # Save a maximum X checkpoints at every N training_iteration
    #     checkpoint_config=CheckpointConfig(
    #         checkpoint_score_attribute = "episode_reward_mean",
    #         checkpoint_score_order = "max",
    #         checkpoint_frequency=10,
    #         checkpoint_at_end=True,
    #         num_to_keep = 3
    #     ),

    #     verbose=1, #3
    #         # progress_reporter=ray.tune.JupyterNotebookReporter(
    #         #     overwrite=True,  
    #         #     parameter_columns=["entropy"],
    #         #     print_intermediate_tables=True,
    #         #     metric_columns=["training_iteration", "episode_reward_min", "episode_reward_mean", "episode_reward_max"], 
    #         #     #metric="episode_reward_mean", mode="max"
    #         #     ),
    #     )

    # rl_checkpoint = None
    #     # if USE_CHECKPOINT:
    #     #     rl_checkpoint = RLCheckpoint(local_path = checkpoint_path)
    #     #rl_checkpoint = ray.put(rl_checkpoint)

    # trainer = RLTrainer(
    #     run_config=run_config,
    #     scaling_config=ScalingConfig(
    #         num_workers=2, use_gpu=False,
    #         trainer_resources={"CPU": 0.0}, 
    #         resources_per_worker={"CPU": 1.0}),
    #     algorithm="PPO",
    #     config=config_cf,
    #     resume_from_checkpoint=rl_checkpoint,
    # )
        

    # # Check configuration and parallelispi pm 
    # print(trainer.run_config)
    # print(trainer.scaling_config)
    # trainable = trainer.as_trainable()
    # print(trainable.default_resource_request(config_cf))
            
    # tune = ray.tune.Tuner(trainer)


    # tune.fit()   
        # checkpoint =  algo.evaluate()
        
    
        
    
    algo = PPOConfig().environment(env=MultiAgent1W1F).multi_agent( 
            policies={ 

                "policy_1": ( 

                    None, env.observation_space, env.action_space, {"gamma": 0.80} 

                ), 

                "policy_2": ( 

                    None, env.observation_space, env.action_space, {"gamma": 0.95} 

                ), 

            }, 

            policy_mapping_fn = lambda agent_id:
        f"policy_{agent_id}", 
        ).build() 

    

    print(algo.train()) 

    print('rew------333') 
    