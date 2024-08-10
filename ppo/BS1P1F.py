 
 
 ####finish same max days
 
# Run successful on MAC computer 
 
# -*- coding: utf-8 -*- 
 
""" 
 
Created on Tue Apr  9 14:11:52 2024 
 
@author: wangw  
 
gymnasium                       0.28.1 
gymnasium                     0.29.1
gym                         0.23.1 
gym                           0.26.2
ray                             2.2.0 
keras                         2.15.0
tensorboard                   2.15.2
tensorboardX                  2.6.2.2
tensorflow                    2.15.0
tensorflow-probability        0.23.0

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
from ray.rllib.policy.policy import PolicySpec
from typing import Set
from ray.rllib.algorithms.dqn import DQNConfig 
from ray.rllib.algorithms.ppo import PPOConfig 
from ray.tune.registry import register_env
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
#from ray.train.rl import RLTrainer, RLCheckpoint, RLPredictor #2.2.0
import gymnasium as gym 
from ray.tune.logger import TBXLoggerCallback,TBXLogger #2.2.0
# from ray import air
from ray.rllib.utils.typing import AgentID
from ray.tune.logger import pretty_print
# from ray.rllib.algorithms.algorithm import Algorithm
# from ray.rllib.env.policy_server_input import PolicyServerInput
# from ray import tune
import os 
from ray import train, tune
from ray.tune import CLIReporter
# import ray
import argparse
from datetime import datetime
import torch
import numpy as np
import time
from ray.rllib.algorithms.ppo import PPO
# import datetime
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import (CombinedStopper,
                              MaximumIterationStopper,
                              ExperimentPlateauStopper)
#from tqdm import trange
from ray.rllib.utils import try_import_torch
from supplyEnv import POMultiAgent1W1F, POMultiAgent1W1F_V1T1,POMultiAgent1W1F_V1T2,POMultiAgent1W1F_V1T3,POMultiAgent1W1F_V2T1,POMultiAgent1W1F_V2T2,POMultiAgent1W1F_V2T3

NUM_EPISODES = 250
# number of episodes for RLib agents
num_episodes_ray = 75000
# stop trials at least from this number of episodes
grace_period_ray = num_episodes_ray / 10
# number of episodes to consider
std_episodes_ray = 5.0
# number of epochs to wait for a change in the episodes
top_episodes_ray = NUM_EPISODES
# name of the experiment (e.g., '2P2W' stands for two product types and two
# distribution warehouses)
now = datetime.now()
now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
env = POMultiAgent1W1F()
save_dir = f"{env.v_f}P{env.task}W_{now_str}"
# dir to save plots
plots_dir = 'plots'
# creating necessary dirs
if not os.path.exists(f"{save_dir}"):
    os.makedirs(f"{save_dir}")
if not os.path.exists(f"{save_dir+'/'+plots_dir}"):
    os.makedirs(f"{save_dir+'/'+plots_dir}")
# dir for saving Ray results
ray_dir = 'ray_results'
# creating necessary dir
if not os.path.exists(f"{save_dir+'/'+ray_dir}"):
    os.makedirs(f"{save_dir+'/'+ray_dir}")

 
# def parse_args():
#     parser = argparse.ArgumentParser(description="POMultiAgent1W1F Environment Configuration")
#     parser.add_argument('--v_f', type=int, help='Visibility of factory stock: 1 or 2')
#     parser.add_argument('--task', type=int,help='Task type: 1, 2, or 3')
#     return parser.parse_args()
def train(algorithm, config, verbose,
          num_episodes_ray=num_episodes_ray, grace_period_ray=grace_period_ray,
          std_episodes_ray=std_episodes_ray, top_episodes_ray=top_episodes_ray,
          local_dir=save_dir, ray_dir=ray_dir):
    """
    Train a RLib Agent.
    """
    # initializing Ray
    ray.shutdown()
    ray.init(log_to_driver=False)

    # logger.debug(f"\n-- train --"
    #              f"\nalgorithm is "
    #              f"{algorithm}"
    #              f"\nconfig is "
    #              f"{config}")

    # https://docs.ray.io/en/latest/tune/api_docs/execution.html
    # https://docs.ray.io/en/master/tune/api_docs/schedulers.html
    # https://docs.ray.io/en/latest/tune/api_docs/stoppers.html
    # https://docs.ray.io/en/master/tune/api_docs/analysis.html
    analysis = tune.run(algorithm,
                        config=config,
                        metric='episode_reward_mean',
                        mode='max',
                        scheduler=ASHAScheduler(
                            time_attr='episodes_total',
                            max_t=num_episodes_ray,
                            grace_period=grace_period_ray,
                            reduction_factor=5),
                        stop=CombinedStopper(
                            ExperimentPlateauStopper(
                                metric='episode_reward_mean',
                                std=std_episodes_ray,
                                top=top_episodes_ray,
                                mode='max',
                                patience=5),
                            MaximumIterationStopper(
                                max_iter=num_episodes_ray)
                        ),
                        checkpoint_freq=1,
                        keep_checkpoints_num=1,
                        checkpoint_score_attr='episode_reward_mean',
                        progress_reporter=tune.JupyterNotebookReporter(
                            overwrite=True),
                        max_failures=5,
                        verbose=verbose,
                        local_dir=os.getcwd()+'/'+local_dir+'/'+ray_dir)

    trial_dataframes = analysis.trial_dataframes
    best_result_df = analysis.best_result_df
    best_config = analysis.best_config
    best_checkpoint = analysis.best_checkpoint._local_path
    print(f"\ncheckpoint saved at {best_checkpoint}")

    # stopping Ray
    ray.shutdown()

    return trial_dataframes, best_result_df, best_config, best_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="POMultiAgent1W1F Environment Configuration")
    parser.add_argument('--env1', action='store_true', help='Use POMultiAgent1W1F_V1T1 environment')
    parser.add_argument('--env2', action='store_true', help='Use POMultiAgent1W1F_V1T2 environment')
    parser.add_argument('--env3', action='store_true', help='Use POMultiAgent1W1F_V1T3 environment')
    parser.add_argument('--env4', action='store_true', help='Use POMultiAgent1W1F_V2T1 environment')
    parser.add_argument('--env5', action='store_true', help='Use POMultiAgent1W1F_V2T2 environment')
    parser.add_argument('--env6', action='store_true', help='Use POMultiAgent1W1F_V2T3 environment')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.env1:
        envs = POMultiAgent1W1F_V1T1
    elif args.env2:
        envs = POMultiAgent1W1F_V1T2
    elif args.env3:
        envs = POMultiAgent1W1F_V1T3
    elif args.env4:
        envs = POMultiAgent1W1F_V2T1
    elif args.env5:
        envs = POMultiAgent1W1F_V2T2
    elif args.env6:
        envs = POMultiAgent1W1F_V2T3
    else:
        raise ValueError("Please specify one of --env1, ..., --env6")


    

    # # training a PPO agent
    # (results_PPO, best_result_PPO,
    # best_config_PPO, checkpoint_PPO) = train(algorithms['PPO'],
    #                                         config_PPO,
    #                                         VERBOSE)
     
    args = parse_args()
    args.v_f = 2
    args.task =3 
    env_config = {"v_f": args.v_f, "task": args.task}
    env = envs()

    config = PPOConfig().environment(env=envs).multi_agent( 
          #.training(train_batch_size=128).
            policies={ 
                # "policy_2": PolicySpec(config=PPOConfig.overrides(lambda_=0.95)),
                "policy_1": ( 
                    None, env.observation_space, env.action_space, {"gamma": 0.80} 
                ), 
                "policy_2": ( 
                    None, env.observation_space, env.action_space, {"gamma": 0.95} 

                ), 

            }, 

            policy_mapping_fn = lambda agent_id:
        f"policy_{agent_id}", 
        )
    

    # save_dir = './envs'
    stop_cf = {"timesteps_total":  400000}#10000000

    # run_config=tune.RunConfig(
    #         stop=stopper,
    #         checkpoint_config=tune.CheckpointConfig(
    #             checkpoint_frequency=1,
    #             num_to_keep=1,
    #             checkpoint_score_attribute='episode_reward_mean'
    #         ),
    #         progress_reporter=reporter,
    #         local_dir=os.path.join(os.getcwd(), local_dir, ray_dir),
    #         max_failures=5,
    #         verbose=verbose
    #     )

    
    


    # algorithms = {
    #     'PPO': ppo.PPO,
    # }

    
   
    # from ray import tune
    # from ray.tune.schedulers import ASHAScheduler
    # from ray.tune.stopper import CombinedStopper, ExperimentPlateauStopper, MaximumIterationStopper
    # from ray.tune import JupyterNotebookReporter

    # 配置调度器和停止条件
    scheduler = ASHAScheduler(
        time_attr='episodes_total',
        max_t=num_episodes_ray,
        grace_period=grace_period_ray,
        reduction_factor=5
    )

    stopper = CombinedStopper(
        ExperimentPlateauStopper(
            metric='episode_reward_mean',
            std=std_episodes_ray,
            top=top_episodes_ray,
            mode='max',
            patience=5
        ),
        MaximumIterationStopper(max_iter=num_episodes_ray)
    )

    # reporter = JupyterNotebookReporter(overwrite=True)
    reporter = CLIReporter(
    metric_columns=["episode_reward_mean", "episodes_total", "training_iteration"])
    
    run_config = RunConfig(
        # local_dir=save_dir,
        stop=stop_cf,
        name="my_train_run",
  
        checkpoint_config=CheckpointConfig(
            checkpoint_score_attribute = "episode_reward_mean",
            checkpoint_score_order = "max",
            checkpoint_frequency=2,
            checkpoint_at_end=True,
            num_to_keep = 10
        ),
        progress_reporter=reporter,
        local_dir=os.path.join(os.getcwd(), save_dir, ray_dir),
        callbacks=[
            TBXLoggerCallback(),
        ],
        verbose=3, #3
        
        log_to_file = True,
        )
    
    
    # tuner = tune.Tuner(
    #         "PPO", run_config=run_config, param_space=config,
    #     )
    
    # results = tuner.fit()
    
    # 定义Tuner
    start = time.time()
    #iterator = trange(1)
    #for epoch in iterator:
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        tune_config=tune.TuneConfig(
            metric='episode_reward_mean',
            mode='max',
            scheduler=scheduler,
            num_samples=1,  # 调优的样本数量，可以根据需要调整
        ),
        # progress_reporter=reporter,
        run_config=run_config,
        # max_failures=5,
        # verbose=3
    )

    # 运行调优
    results = tuner.fit()

    total_time_taken = time.time()-start
    print(f"Total number of models: {len(results)}")
    print(f"Total time taken: {total_time_taken/60:.2f} minutes")
    
    best_result = results.get_best_result()
    best_result_df = best_result.metrics_dataframe
    best_config = best_result.config
    best_checkpoint = best_result.checkpoint
   

    print(f"\nbest_result saved at {best_result}")
    print(f"\nbest_result_df saved at {best_result_df}")
    print(f"\nbest_config saved at {best_config}")
    print(f"\nbest_checkpoint saved at {best_checkpoint}")

    # 停止 Ray
    ray.shutdown()

    


    


    
