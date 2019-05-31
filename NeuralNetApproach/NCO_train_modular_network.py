#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:14:42 2019

@author: hajime
"""
import torch
import torch.utils.data
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable

import time
import datetime
import pytz
import pickle

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from sklearn.model_selection import ParameterGrid
from NCO_functions import *
from NCO_main import NCO_main1

from sklearn.model_selection import ParameterGrid


'''
num_environment = 6
num_agent = 11
num_manager = 10
num_layer = 3
agent_per_layer = [4,4,2]#manager only
num_partition_environment = 2
num_partition_agent = [2,2,2] #manager only

dunbar_number = 2
rho = .05
'''


Description = 'Modular-Network-Small-Oracle-Org'
plt.ioff()

exec_date = datetime.datetime.now(pytz.timezone('US/Mountain')).strftime('%B%d_%H%M')

dirname ='./result_'+exec_date +'_' + Description

createFolder(dirname)

n_rep = 100

param0 = {'num_environment' : 16,
          'agent_per_layer' : [8,4,2],
          'num_partition_environment': 4,
          'num_partition_agent' : [4,2,2],
          'dunbar_number_oracle': 2,
          'dunbar_number_org': 2,
          'rho_oracle': .05,
          'rho_org': .05
          }
param1 = {'num_environment' : 16,
          'agent_per_layer' : [8,4,2],
          'num_partition_environment': 4,
          'num_partition_agent' : [2,2,2],
          'dunbar_number_oracle': 2,
          'dunbar_number_org': 2,
          'rho_oracle': .01,
          'rho_org': .01
          }
param2 = {'num_environment' : 16,
          'agent_per_layer' : [8,4,2],
          'num_partition_environment': 4,
          'num_partition_agent' : [2,2,2],
          'dunbar_number_oracle': 4,
          'dunbar_number_org': 2,
          'rho_oracle': .05,
          'rho_org': .05
          }
param3 = {'num_environment' : 16,
          'agent_per_layer' : [8,4,2],
          'num_partition_environment': 4,
          'num_partition_agent' : [2,2,2],
          'dunbar_number_oracle': 4,
          'dunbar_number_org': 2,
          'rho_oracle': .01,
          'rho_org': .01
          }

params = [param0, param1, param2, param3]
batchsize = 500
pickle.dump(params, open(dirname+"/params.pickle","wb"))



for param_i in range(len(params)):
    param = params[param_i]
    
    final_error = []
    finished_it = []
    
    param['num_layer'] = len(param['agent_per_layer'])
    param['num_manager'] = np.sum(param['agent_per_layer'])
    param['num_agent'] = param['num_manager']+1    
    for key,val in param.items():
        exec(key + '=val')

    environment_partition_id, agent_layer_id, agent_partition_id = gen_layer_and_partition(num_environment,num_agent,num_manager,num_layer,agent_per_layer,num_partition_environment,num_partition_agent)
        
    for rep in range(n_rep):
        nodes_network_for_env = gen_layered_modular_network(num_environment,num_agent,num_layer,dunbar_number_oracle, #Number of input nodes, number of layers other than input layer
                                        agent_layer_id, #if it's seq, it should be a vecctor of length num_layer.
                                        agent_partition_id, #If it's seq, it should be a vector of length num_layer.
                                        environment_partition_id, # partition of environment.
                                        partitions_network, #Network of partitions. (num_partition_environment+sum(num_partition_layer_seq), )
                                        rho_oracle#Probability that link is formed without partition consideration.
                                        )
        nodes_network_for_org = gen_layered_modular_network(num_environment,num_agent,num_layer,dunbar_number_org, #Number of input nodes, number of layers other than input layer
                                        agent_layer_id, #if it's seq, it should be a vecctor of length num_layer.
                                        agent_partition_id, #If it's seq, it should be a vector of length num_layer.
                                        environment_partition_id, # partition of environment.
                                        partitions_network, #Network of partitions. (num_partition_environment+sum(num_partition_layer_seq), )
                                        rho_org#Probability that link is formed without partition consideration.
                                        )




        env = Environment(batchsize=batchsize,num_environment=num_environment,env_network=nodes_network_for_env,num_agents=num_agent,num_manager=num_manager,num_actor=1,
                         env_weight=None,env_bias=None,env_type='gen_from_network',env_n_region=None,input_type='random',
                         flag_normalize=False)
        
        env.create_env()
        env_input = env.environment
        env_output = env.env_pattern
        
        
        org1 = NCO_main1(num_agent = num_agent, num_manager = num_manager, num_environment = num_environment, num_actor = 1, dunbar_number = dunbar_number_org,
                         lr = .01, L1_coeff = 0., n_it = 20000,
                         message_unit = torch.sigmoid, action_unit = torch.sigmoid,
                         flag_DeepR = False, DeepR_freq = None, DeepR_T = None, DeepR_layered=False,
                         flag_pruning = False, type_pruning= None, pruning_freq = None,
                         flag_DiscreteChoice = False, flag_DiscreteChoice_Darts = False, DiscreteChoice_freq = None, DiscreteChoice_lr = None, DiscreteChoice_L1_coeff = None,
                         flag_ResNet = False,
                         flag_AgentPruning = False, AgentPruning_freq = None,
                         flag_slimming = False, slimming_freq = None, slimming_L1_coeff = None,slimming_threshold=None,
                         flag_minibatch = False, minibatch_size = None,
                         type_initial_network = 'specified', flag_BatchNorm = False, initial_network_depth=None,initial_network = nodes_network_for_org,
                         batchsize = None, env_type = 'specified',env_input_type='all_comb',env_n_region=None,width_seq=None, env_input = env_input, env_output = env_output
                         )
                
        org1.func_Train()
        
        final_error.append(org1.error_rate.numpy())
        finished_it.append( org1.current_it )
        
        
    pickle.dump(final_error, open(dirname+"/final_error_param%i.pickle"%param_i,"wb"))
    pickle.dump(finished_it, open(dirname+"/finished_it_param%i.pickle"%param_i,"wb"))
    
