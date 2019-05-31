#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:20:45 2019

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

from sklearn.model_selection import ParameterGrid

from NCO_functions import createFolder,Environment,gen_full_network,gen_constrained_network,draw_network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float
'''
device = torch.device("cpu")
dtype_float = torch.float
'''

if device != 'cpu':
    pass
else:
    pass

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print('---Device:'+str(device)+'---')


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.tensor(np.random.randn(*size).astype(np.float32) * xavier_stddev, requires_grad=True,device=device)


class NCO_main1(nn.Module):
    def __init__(self, num_agent = 10, num_manager = 9, num_environment = 6, num_actor = 1, dunbar_number = 4,
                 lr = .01, L1_coeff = 0., n_it = 200000,
                 message_unit = torch.sigmoid, action_unit = torch.sigmoid,
                 flag_DeepR = False, DeepR_freq = 2000, DeepR_T = 0.00001,DeepR_layered=False,
                 flag_pruning = False, type_pruning= None, pruning_freq = 100,
                 flag_DiscreteChoice = False, flag_DiscreteChoice_Darts = False, DiscreteChoice_freq = 10, DiscreteChoice_lr = 0.,DiscreteChoice_L1_coeff = 0.001,
                 flag_ResNet = False,
                 flag_AgentPruning = False, AgentPruning_freq = None,
                 flag_slimming = False, slimming_freq = None, slimming_L1_coeff = None,slimming_threshold=None,
                 flag_minibatch = False, minibatch_size = None,
                 type_initial_network = 'ConstrainedRandom', flag_BatchNorm = True,initial_network_depth=None,initial_network = None,
                 batchsize = None, env_type = 'match_mod2',env_input_type='all_comb',env_n_region=None,width_seq=None, env_input = None, env_output=None
                 ):
        super(NCO_main1, self).__init__()
        #Basic parameters
        self.num_agent = num_agent #total number of nodes
        self.num_manager = num_manager #number of nodes that do not take action (output)
        self.num_environment = num_environment # number of input nodes
        self.num_actor = num_actor #number of output nodes, currently assumed to be 1.
        self.dunbar_number = dunbar_number #max fan-in of each node
        self.message_unit = message_unit #function that each manager node performs to compute its output, sigmoid, ReLU, etc
        self.action_unit = action_unit #function that actor node performs to compute final output

        self.dim_input_max = num_environment+num_manager

        #Learning parameters
        self.lr = lr#.1#1e-2#1e-3  #Learning rate
        self.L1_coeff = L1_coeff#0.0001#.5  #coefficient on L1 norm
        self.n_it = n_it#1000000  #number of iteration of SDG

        #DeepR
        self.flag_DeepR = flag_DeepR  # True to perform "DeepR"
        self.DeepR_freq = DeepR_freq  #Every * iterations, perform DeepR 
        self.DeepR_T = DeepR_T  # "temperture" of DeepR
        self.DeepR_layered = DeepR_layered  

        #Pruning
        self.flag_pruning = flag_pruning #True to perform pruning
        self.type_pruning = type_pruning #Type of pruning
        self.pruning_freq = pruning_freq #Every * iterations, perform pruning 

        #Slimming
        #Not implemented yet
        self.flag_slimming=flag_slimming #Not implemented yet
        self.slimming_freq = slimming_freq #Not implemented yet
        self.slimming_L1_coeff =slimming_L1_coeff #Not implemented yet
        self.slimming_threshold = slimming_threshold #Not implemented yet

        #Discrete choice and Darts
        self.flag_DiscreteChoice = flag_DiscreteChoice #True to perform DARTS-ish connection reduction.
        self.flag_DiscreteChoice_Darts = flag_DiscreteChoice_Darts
        self.DiscreteChoice_freq = DiscreteChoice_freq
        self.DiscreteChoice_lr = DiscreteChoice_lr
        self.DiscreteChoice_L1_coeff = DiscreteChoice_L1_coeff

        #ResNet
        #Not implemented yet
        self.flag_ResNet = flag_ResNet

        #Agent Pruning
        #Not implemented yet
        self.flag_AgentPruning = flag_AgentPruning
        self.AgentPruning_freq = AgentPruning_freq


        #Initial values
        self.type_initial_network = type_initial_network #'FullyConnected', 'ConstrainedRandom'

        #Batch normalization
        self.flag_BatchNorm = flag_BatchNorm #True to perform batch normalization.

        #Type of environment generation
        self.env_type = env_type
        self.env_input_type = env_input_type
        self.env_n_region = env_n_region


        #Generate network and environment
        self.network_const_env = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number)
        self.network_full_np = gen_full_network(num_environment,num_manager,num_agent)
        self.fanin_max_list = np.sum(self.network_full_np,axis=0).astype(np.int)

        if type_initial_network is 'ConstrainedRandom':
            self.network_const_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number)
            self.network = torch.tensor(np.abs(self.network_const_np),device=device)
        elif type_initial_network is 'FullyConnected':
            self.network = torch.tensor(np.abs(self.network_full_np).astype(np.float32),device=device)
        elif type_initial_network is 'FullyConnected_NoDirectEnv':
            self.network = torch.tensor(np.abs(self.network_full_np).astype(np.float32),device=device)
            self.network[:num_environment, num_manager:]=0
        elif type_initial_network is 'layered_random':
            self.network_full_layered_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number, type_network='layered_full',width_seq=width_seq)
            self.network_full_layered = torch.Tensor(self.network_full_layered_np.astype(np.float32))
            self.network_const_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number, type_network='layered_random',width_seq=width_seq)
            self.network = torch.tensor(np.abs(self.network_const_np).astype(np.float32),device=device )
            self.num_layer = len(width_seq)
            self.width_seq = width_seq
        elif type_initial_network is 'layered_full':
            self.network_full_layered_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number, type_network='layered_full',width_seq=width_seq)
            self.network_full_layered = torch.tensor(self.network_full_layered_np.astype(np.float32), device=device)
            self.network = torch.tensor(np.abs(self.network_full_layered_np).astype(np.float32),device=device)
            self.num_layer = len(width_seq)
            self.width_seq = width_seq
        elif type_initial_network is 'specified':
            self.network = torch.tensor(initial_network.astype(np.float32), device=device)


        #Number of pruning per time
        if flag_pruning:
            self.n_need_to_prune = torch.sum(torch.abs(self.network) ) - self.dunbar_number*self.num_agent
            self.prune_per_time= int(self.n_need_to_prune/int((self.n_it-3000)/self.pruning_freq) )


        #Create "environment", i.e. input and output samples
        if env_type is not 'specified':
            env_class = Environment(batchsize=batchsize,num_environment=self.num_environment,num_agents=num_agent,num_manager=num_manager,num_actor=num_actor,env_type=env_type,env_n_region=env_n_region,input_type=env_input_type,flag_normalize=False,env_network=self.network_const_env)
            env_class.create_env()
    
            self.env_input_np = env_class.environment.astype(np.float32)
            self.env_output_np = env_class.env_pattern.astype(np.float32)
            self.env_input = torch.tensor(self.env_input_np,device=device)
            self.env_output = torch.tensor(self.env_output_np,device=device)
        elif env_type is 'specified':
            self.env_input_np = env_input.astype(np.float32)
            self.env_output_np = env_output.astype(np.float32)
            self.env_input = torch.tensor(self.env_input_np,device=device)
            self.env_output = torch.tensor(self.env_output_np,device=device)
            

        #Batchsize
        self.batchsize = self.env_input.shape[0]#64
        #Minibatch learning
        self.flag_minibatch = flag_minibatch
        if flag_minibatch:
            self.minibatch_size = minibatch_size
        else:
            self.minibatch_size = self.batchsize


        m_in_max = np.sum(self.network_full_np,axis=0)-num_environment
        m_in_max[0] += .0000001


        #Create learnable parameters
        ## Weights and Biases
        self.Wmm_init_np = np.random.normal( 0, 1/np.sqrt(m_in_max[:num_manager]), size = [num_manager,num_manager] )
        self.Wma_init_np = np.random.normal( 0, 1/np.sqrt(m_in_max[num_manager:]), size = [num_manager,num_actor] )
        #Weights. shape[0] is the dimension of inputs, shape[1] is the number of agents.
        self.W_env_to_message = xavier_init([num_environment,num_manager])#Variable(torch.randn([num_environment,num_manager]), requires_grad=True)
        self.W_env_to_action = xavier_init([num_environment,num_actor])#Variable(torch.randn([num_environment,num_actor]), requires_grad=True)
        self.W_message_to_message = torch.tensor(np.random.randn(num_manager,num_manager).astype(np.float32), requires_grad=True,device=device)
        self.W_message_to_action = torch.tensor(np.random.randn(num_manager,num_actor).astype(np.float32), requires_grad=True,device=device)
        self.b_message = torch.tensor(np.zeros([num_manager]).astype(np.float32), requires_grad=True,device=device)
        self.b_action = torch.tensor(np.random.randn(num_actor).astype(np.float32), requires_grad=True,device=device)
        ##Parameters for batch normalization
        if flag_BatchNorm:
            self.BatchNorm_gamma_message_to_message = torch.tensor(np.random.randn(num_manager,num_manager).astype(np.float32), requires_grad=True ,device=device)
            self.BatchNorm_gamma_message_to_action = torch.tensor(np.random.randn(num_manager,num_actor).astype(np.float32), requires_grad=True,device=device)
            self.BatchNorm_gamma_env_to_message = torch.tensor(np.random.randn(num_environment,num_manager).astype(np.float32), requires_grad=True,device=device)
            self.BatchNorm_gamma_env_to_action = torch.tensor(np.random.randn(num_environment,num_actor).astype(np.float32), requires_grad=True,device=device)

            self.BatchNorm_beta_message_to_message = torch.tensor(np.random.randn(num_manager,num_manager).astype(np.float32), requires_grad=True,device=device)
            self.BatchNorm_beta_message_to_action = torch.tensor(np.random.randn(num_manager,num_actor).astype(np.float32), requires_grad=True,device=device)
            self.BatchNorm_beta_env_to_message = torch.tensor(np.random.randn(num_environment,num_manager).astype(np.float32), requires_grad=True,device=device)
            self.BatchNorm_beta_env_to_action = torch.tensor(np.random.randn(num_environment,num_actor).astype(np.float32), requires_grad=True,device=device)

            self.BatchNorm_eps = 1e-5

        #Parameters for discrete chocie method (inactive)
        if flag_DiscreteChoice or flag_DiscreteChoice_Darts:
            self.DiscreteChoice_alpha = torch.tensor(np.zeros_like(self.network).astype(np.float32), requires_grad=True,device=device)#Variable(torch.randn_like(network), requires_grad=True)


        self.params_to_optimize = [self.W_env_to_message,self.W_env_to_action,self.W_message_to_message,self.W_message_to_action,self.b_message,self.b_action]

        if flag_DiscreteChoice or flag_DiscreteChoice_Darts:
            self.params_to_optimize.append(self.DiscreteChoice_alpha)
        if flag_BatchNorm:
            self.params_to_optimize.extend([self.BatchNorm_gamma_message_to_message,self.BatchNorm_gamma_message_to_action,self.BatchNorm_gamma_env_to_message,self.BatchNorm_gamma_env_to_action,self.BatchNorm_beta_message_to_message,self.BatchNorm_beta_message_to_action,self.BatchNorm_beta_env_to_message,self.BatchNorm_beta_env_to_action])


        # Define loss function
        #solver = optim.Adam(params_to_optimize, lr=lr)
        #loss = nn.BCELoss(reduction='sum')
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')


        #Recording the information
        self.W_env_to_message_list = []
        self.W_env_to_action_list = []
        self.W_message_to_message_list = []
        self.W_message_to_action_list = []
        self.b_message_list = []
        self.b_action_list = []

        self.action_loss_list=[]
        self.total_loss_list=[]
        self.error_rate_list=[]

        self.network_list = []
        self.message_list = []

        if flag_DiscreteChoice or flag_DiscreteChoice_Darts:
            self.DiscreteChoice_alpha_list=[]


    def func_BatchNorm(self,X, gamma, beta, eps):
        '''
        function to add batch-normalization
        '''
        X_BachNorm =  gamma * ( (X-torch.mean(X,dim=0) )/ (torch.std(X,dim=0)+eps ) ) +beta

        return X_BachNorm


    def func_Train(self):
        '''
        function to run training
        '''
        for it in range(self.n_it):
            self.current_it = it
            '''
            if it>5000:
                lr = 1.
            if it>20000:
                lr = .01
            '''

            #Initialize message and action
            message = torch.tensor(np.zeros([self.minibatch_size,self.num_manager]).astype(np.float32),device=device)
            self.action_state = torch.tensor(np.zeros([self.minibatch_size,self.num_actor]).astype(np.float32),device=device)
            self.action = torch.tensor(np.zeros([self.minibatch_size,self.num_actor]).astype(np.float32),device=device)
            self.action_loss = torch.tensor(np.zeros([self.minibatch_size,self.num_actor]).astype(np.float32),device=device)

            if self.flag_minibatch:
                minibatch_idx = torch.randint( 0, self.batchsize, (self.minibatch_size,) )
                env_input_it = self.env_input[minibatch_idx]
                env_output_it = self.env_output[minibatch_idx]
            else:
                env_input_it = self.env_input
                env_output_it = self.env_output




            #Create messages sequentially
            for i in range(self.num_manager):
                temp = self.b_message[i].repeat([self.minibatch_size, 1])

                if not self.flag_BatchNorm:
                    env_input_i = env_input_it
                    message_in_i = message.clone()
                elif self.flag_BatchNorm:
                    env_input_i = self.func_BatchNorm(env_input_it, self.BatchNorm_gamma_env_to_message[:,i],self.BatchNorm_beta_env_to_message[:,i], self.BatchNorm_eps)
                    message_in_i = self.func_BatchNorm(message.clone(), self.BatchNorm_gamma_message_to_message[:,i],self.BatchNorm_beta_message_to_message[:,i], self.BatchNorm_eps)

                if not (self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts):
                    message[:,i] = ( self.message_unit( temp + env_input_i @ (self.W_env_to_message[:,i] * self.network[:self.num_environment,i]).reshape([-1,1]) + message_in_i @ (self.W_message_to_message[:,i] * self.network[self.num_environment:,i]).reshape([-1,1])   ) ).flatten()
                elif self.flag_DiscreteChoice:
                    choice_prob_m = nn.functional.softmax(self.DiscreteChoice_alpha[:,i] + (self.network[:,i]-1.)*1000000  )
                    message[:,i] = ( self.message_unit( temp + env_input_i @ (self.W_env_to_message[:,i] * self.network[:self.num_environment,i] * choice_prob_m[:self.num_environment]).reshape([-1,1]) + message_in_i @ (self.W_message_to_message[:,i] * self.network[self.num_environment:,i] * choice_prob_m[self.num_environment:]).reshape([-1,1])   ) ).flatten()
                elif self.flag_DiscreteChoice_Darts:
                    choice_prob_m = 1./(torch.exp(self.DiscreteChoice_alpha[:,i])+1.)
                    message[:,i] = ( self.message_unit( temp + env_input_i @ (self.W_env_to_message[:,i] * self.network[:self.num_environment,i] * choice_prob_m[:self.num_environment]).reshape([-1,1]) + message_in_i @ (self.W_message_to_message[:,i] * self.network[self.num_environment:,i] * choice_prob_m[self.num_environment:]).reshape([-1,1])   ) ).flatten()

            #Create action
            for j in range(self.num_actor):
                if not self.flag_BatchNorm:
                    env_input_j = env_input_it
                    message_in_j = message.clone()
                elif self.flag_BatchNorm:
                    env_input_j = self.func_BatchNorm(env_input_it, self.BatchNorm_gamma_env_to_action[:,j],self.BatchNorm_beta_env_to_action[:,j], self.BatchNorm_eps)
                    message_in_j = self.func_BatchNorm(message.clone(), self.BatchNorm_gamma_message_to_action[:,j], self.BatchNorm_beta_message_to_action[:,j], self.BatchNorm_eps)


                if not (self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts):
                    self.action_state[:,j] = (self.b_action[j].repeat([self.minibatch_size, 1]) + env_input_j @ (self.W_env_to_action[:,j] * self.network[:self.num_environment,self.num_manager+j]).reshape([-1,1]) + message_in_j @ (self.W_message_to_action[:,j] * self.network[self.num_environment:,self.num_manager+j]).reshape([-1,1]) ).flatten()
                elif self.flag_DiscreteChoice:
                    choice_prob_a = nn.functional.softmax(self.DiscreteChoice_alpha[:,self.num_manager+j] + (self.network[:,self.num_manager+j]-1.)*1000000  )
                    self.action_state[:,j] = (self.b_action[j].repeat([self.minibatch_size, 1]) + env_input_j @ (self.W_env_to_action[:,j] * self.network[:self.num_environment,self.num_manager+j] * choice_prob_a[:self.num_environment]).reshape([-1,1]) + message_in_j @ (self.W_message_to_action[:,j] * self.network[self.num_environment:,self.num_manager+j] * choice_prob_a[self.num_environment:]).reshape([-1,1]) ).flatten()
                elif self.flag_DiscreteChoice_Darts:
                    choice_prob_a = 1./(torch.exp(self.DiscreteChoice_alpha[:,self.num_manager+j]) +1.)
                    self.action_state[:,j] = (self.b_action[j].repeat([self.minibatch_size, 1]) + env_input_j @ (self.W_env_to_action[:,j] * self.network[:self.num_environment,self.num_manager+j] * choice_prob_a[:self.num_environment]).reshape([-1,1]) + message_in_j @ (self.W_message_to_action[:,j] * self.network[self.num_environment:,self.num_manager+j] * choice_prob_a[self.num_environment:]).reshape([-1,1]) ).flatten()

                self.action[:,j] = self.action_unit(self.action_state[:,j])

            #Calculate loss and backprop
            self.action_loss = self.loss(self.action_state, env_output_it)
            self.L1_loss = torch.sum( torch.abs(self.W_env_to_message)) + torch.sum(torch.abs(self.W_env_to_action) )+torch.sum(torch.abs(self.W_message_to_message) )+ torch.sum(torch.abs(self.W_message_to_action) )#+ torch.sum(torch.abs(b_message)+torch.abs(b_action) )

            self.total_loss = self.action_loss+self.L1_loss*self.L1_coeff
            if self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts:
                self.L1_alpha = torch.sum(torch.abs(self.DiscreteChoice_alpha) )
                self.total_loss = self.total_loss+self.L1_alpha*self.DiscreteChoice_L1_coeff

            #"Slimming" method
            if self.type_pruning=='Slimming':
                self.L1_on_slimming = torch.sum(torch.abs(self.BatchNorm_gamma_message_to_message))
                self.total_loss = self.total_loss+self.slimming_L1_coeff*self.L1_on_slimming

            #Compute gradient
            self.total_loss.backward()
            #Error statistic
            self.error_rate = torch.mean(torch.abs((self.action.data.cpu()>.5).float() - env_output_it.data.cpu() ) )


            #Gradient Descent Update
            with torch.no_grad():

                #print(W_env_to_message.grad )
                self.update_max = 1.
                if True:#it>20000:#
                    self.W_env_to_message = self.W_env_to_message - torch.max( torch.min(self.lr * self.W_env_to_message.grad, torch.ones_like(self.W_env_to_message)*self.update_max)  ,   -torch.ones_like(self.W_env_to_message)*self.update_max)
                    self.W_message_to_message = self.W_message_to_message - torch.max( torch.min(self.lr * self.W_message_to_message.grad , torch.ones_like(self.W_message_to_message)*self.update_max)       , -torch.ones_like(self.W_message_to_message)*self.update_max)
                    self.b_message = self.b_message - torch.max( torch.min(self.lr * self.b_message.grad  , torch.ones_like(self.b_message)*self.update_max)        ,    -torch.ones_like(self.b_message)*self.update_max)
                if True:#it<=20000:#
                    self.W_env_to_action = self.W_env_to_action - torch.max( torch.min(self.lr * self.W_env_to_action.grad    , torch.ones_like(self.W_env_to_action)*self.update_max), -torch.ones_like(self.W_env_to_action)*self.update_max)
                    self.W_message_to_action = self.W_message_to_action - torch.max( torch.min(self.lr * self.W_message_to_action.grad , torch.ones_like(self.W_message_to_action)*self.update_max)   , -torch.ones_like(self.W_message_to_action)*self.update_max)
                    self.b_action = self.b_action - torch.max( torch.min(self.lr * self.b_action.grad, torch.ones_like(self.b_action)*self.update_max), -torch.ones_like(self.b_action)*self.update_max)

                    #For printing
                    self.W_message_to_action_grad = self.W_message_to_action.grad

                #DeepR algorithm (implemented, inactive)
                if self.flag_DeepR:
                    self.W_env_to_message = self.W_env_to_message +torch.randn_like(self.W_env_to_message) * np.sqrt(2.*self.lr*self.DeepR_T)
                    self.W_env_to_action = self.W_env_to_action +torch.randn_like(self.W_env_to_action) * np.sqrt(2.*self.lr*self.DeepR_T)
                    self.W_message_to_message = self.W_message_to_message + torch.randn_like(self.W_message_to_message) * np.sqrt(2.*self.lr*self.DeepR_T)
                    self.W_message_to_action = self.W_message_to_action + torch.randn_like(self.W_message_to_action) * np.sqrt(2.*self.lr*self.DeepR_T)

                    self.W_env_to_message = torch.where(self.W_env_to_message>0,self.W_env_to_message,torch.zeros_like(self.W_env_to_message))
                    self.W_message_to_message = torch.where(self.W_message_to_message>0,self.W_message_to_message,torch.zeros_like(self.W_message_to_message))
                    self.W_env_to_action = torch.where(self.W_env_to_action>0,self.W_env_to_action,torch.zeros_like(self.W_env_to_action))
                    self.W_message_to_action =torch.where(self.W_message_to_action>0,self.W_message_to_action,torch.zeros_like(self.W_message_to_action))
                
                #Updating batch normalization parameters, active.
                if self.flag_BatchNorm:
                    self.BatchNorm_gamma_message_to_message = self.BatchNorm_gamma_message_to_message - self.lr * self.BatchNorm_gamma_message_to_message.grad
                    self.BatchNorm_gamma_message_to_action = self.BatchNorm_gamma_message_to_action - self.lr * self.BatchNorm_gamma_message_to_action.grad
                    self.BatchNorm_gamma_env_to_message = self.BatchNorm_gamma_env_to_message - self.lr * self.BatchNorm_gamma_env_to_message.grad
                    self.BatchNorm_gamma_env_to_action = self.BatchNorm_gamma_env_to_action - self.lr * self.BatchNorm_gamma_env_to_action.grad

                    self.BatchNorm_beta_message_to_message = self.BatchNorm_beta_message_to_message - self.lr * self.BatchNorm_beta_message_to_message.grad
                    self.BatchNorm_beta_message_to_action = self.BatchNorm_beta_message_to_action - self.lr * self.BatchNorm_beta_message_to_action.grad
                    self.BatchNorm_beta_env_to_message = self.BatchNorm_beta_env_to_message - self.lr * self.BatchNorm_beta_env_to_message.grad
                    self.BatchNorm_beta_env_to_action = self.BatchNorm_beta_env_to_action - self.lr * self.BatchNorm_beta_env_to_action.grad

                    self.BatchNorm_gamma_message_to_message.requires_grad = True
                    self.BatchNorm_gamma_message_to_action.requires_grad = True
                    self.BatchNorm_gamma_env_to_message.requires_grad = True
                    self.BatchNorm_gamma_env_to_action.requires_grad = True

                    self.BatchNorm_beta_message_to_message.requires_grad = True
                    self.BatchNorm_beta_message_to_action.requires_grad = True
                    self.BatchNorm_beta_env_to_message.requires_grad = True
                    self.BatchNorm_beta_env_to_action.requires_grad = True

                #
                self.W_env_to_message.requires_grad = True
                self.W_env_to_action.requires_grad = True
                self.W_message_to_message.requires_grad = True
                self.W_message_to_action.requires_grad = True
                self.b_message.requires_grad = True
                self.b_action.requires_grad = True

                if self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts:
                    if it%self.DiscreteChoice_freq==0:
                        #print('****************Updating alpha*********************')
                        with torch.no_grad():
                            self.DiscreteChoice_alpha = self.DiscreteChoice_alpha -self.DiscreteChoice_lr * self.DiscreteChoice_alpha.grad
                            self.DiscreteChoice_alpha.requires_grad = True
                        #print('alpha mean:'+str( torch.mean(DiscreteChoice_alpha.data.cpu()) ))
                        #print('alpha var:'+str( torch.var(DiscreteChoice_alpha.data.cpu()) ))

            # Housekeeping
            for p in self.params_to_optimize:
                if p.grad is not None:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())


            #DeepR rewiring network
            if self.flag_DeepR:
                if it%self.DeepR_freq==0 and it>0:
                    #print('****************Rewiring Network*********************')
                    negative_m = torch.cat((self.W_env_to_message>0,self.W_message_to_message>0),dim=0).type(dtype_float)
                    negative_a = torch.cat((self.W_env_to_action>0,self.W_message_to_action>0),dim=0).type(dtype_float)
                    negative = torch.cat( (negative_m,negative_a),dim=1 ).type(dtype_float)
                    self.network = negative * self.network
                    for i in range(self.num_agent):
                        network_i = self.network[:,i]
                        fanin_i = torch.sum(torch.abs(network_i) )
                        if fanin_i<self.dunbar_number:

                            n_reactivate = int(self.dunbar_number-fanin_i)
                            if self.DeepR_layered is False:
                                #pos_inactive = np.where(network_i==0)
                                pos_inactive = (network_i==0).nonzero().flatten()
                                #pos_reactivate = np.random.choice(pos_inactive[0][(pos_inactive[0]<self.fanin_max_list[i])],[n_reactivate],replace=False)
                                pos_reactivate = pos_inactive[pos_inactive<self.fanin_max_list[i]][torch.randperm( len(pos_inactive[pos_inactive<self.fanin_max_list[i]]) )[:n_reactivate] ]
                                #network_i[pos_reactivate]=torch.Tensor(np.random.choice( [1.,-1.],len(pos_reactivate) ) )
                                network_i[pos_reactivate]=torch.randint(0,2,(len(pos_reactivate),)).type(dtype_float)*2-1.
                            if self.DeepR_layered is True:
                                #pos_inactive = np.where( (network_i==0) * (self.network_full_layered[:,i]!=0 ) )
                                pos_inactive = ( (network_i==0) * (self.network_full_layered[:,i]!=0 ) ).nonzero().flatten()
                                self.pos_inactive = pos_inactive
                                #if n_reactivate<=len(pos_inactive[0][(pos_inactive[0]<self.fanin_max_list[i])]):
                                if n_reactivate<=len(pos_inactive[(pos_inactive<self.fanin_max_list[i])]):
                                    #pos_reactivate = np.random.choice(pos_inactive[0][(pos_inactive[0]<self.fanin_max_list[i])],[n_reactivate],replace=False)
                                    pos_reactivate = pos_inactive[pos_inactive<self.fanin_max_list[i]][torch.randperm( len(pos_inactive[pos_inactive<self.fanin_max_list[i]]) )[:n_reactivate] ]
                                    #network_i[pos_reactivate]=torch.Tensor(np.random.choice( [1.,-1.],len(pos_reactivate) ) )
                                    network_i[pos_reactivate]=torch.randint(0,2,(len(pos_reactivate),)).type(dtype_float)*2-1.

                            self.network[:,i] = network_i

            #Pruning network
            if self.flag_pruning:
                if it%self.pruning_freq==0 and it>0:

                    total_pruned = 0
                    self.fanin_list = []
                    for i in range(self.num_agent):

                        #Check the nodes that don't speak to anyone.
                        fanout_all = torch.sum( torch.abs(self.network), dim=1 )
                        fanout_all[-1]=2. #the actor is OK not to speak
                        no_fanout = torch.where(fanout_all==0,  torch.tensor([1.],device=device),torch.tensor([0.],  device=device)) #1 if not speaking to other node.
                        one_fanout = torch.where(fanout_all==1,torch.tensor([1.],device=device),torch.tensor([0.],device=device)) #1 if speaking to only one node.
                        multi_fanout =     torch.where(fanout_all>1,torch.tensor([1.],device=device),torch.tensor([0.],device=device))
                        #Check the nodes that speak to only one node. Try not to cut that link

                        self.no_fanout = no_fanout
                        self.one_fanout = one_fanout
                        self.multi_fanout = multi_fanout

                        network_i = torch.abs(self.network[:,i]) #the nodes that i is listening to.
                        fanin_i = torch.sum(torch.abs(network_i) )
                        self.network_i=network_i
                        if fanin_i>self.dunbar_number:
                            n_inactivate = 1
                            if torch.any( (network_i*no_fanout).type('torch.ByteTensor') ):
                                #If speaking to a node not linked to other node, cut it first.
                                pos_active = (network_i*no_fanout).nonzero().flatten()
                            elif  torch.any( (network_i*one_fanout).type('torch.ByteTensor') ):
                                #If speaking to a node that speaks to one node, don't cut it.
                                pos_active = (network_i*multi_fanout).nonzero().flatten()
                            else:
                                pos_active = (network_i).nonzero().flatten()

                            if self.type_pruning is 'Random':
                                pos_inactivate = pos_active[pos_active<self.fanin_max_list[i]][torch.randperm( len(pos_active[pos_active<self.fanin_max_list[i]]) )[:n_inactivate] ]
                            elif self.type_pruning is 'Smallest':
                                if i< self.num_manager:
                                    W_i = torch.cat( ( self.W_env_to_message[:,i].flatten() , self.W_message_to_message[:,i].flatten() ),dim=0  ) * network_i
                                else:
                                    W_i = torch.cat( ( self.W_env_to_action[:,i-self.num_manager].flatten() , self.W_message_to_action[:,i-self.num_manager].flatten() ),dim=0  ) * network_i
                                #_,pos_inactivate = torch.topk( -torch.abs(W_i), 1 )
                                pos_inactivate = (torch.abs(W_i)==-torch.topk(-torch.abs(W_i)[torch.abs(W_i).nonzero()].flatten(),1)[0] )
                            elif self.type_pruning is 'Slimming':
                                if i< self.num_manager:
                                    gamma_i = torch.cat( ( self.BatchNorm_gamma_env_to_message[:,i].flatten() , self.BatchNorm_gamma_message_to_message[:,i].flatten() ),dim=0  ) * network_i
                                else:
                                    gamma_i = torch.cat( ( self.BatchNorm_gamma_env_to_action[:,i-self.num_manager].flatten() , self.BatchNorm_gamma_message_to_action[:,i-self.num_manager].flatten() ),dim=0  ) * network_i
                                pos_inactivate = (torch.abs(gamma_i)==-torch.topk(-torch.abs(gamma_i)[torch.abs(gamma_i).nonzero()].flatten(),1)[0] )


                            #pos_inactivate = np.random.choice(pos_active[0][(pos_active[0]<self.fanin_max_list[i])],[n_inactivate],replace=False)
                            network_i[pos_inactivate]= 0.#torch.zeros(len(pos_inactivate))
                            self.network[:,i] = network_i
                            print('pruning(%i,%i)'%(pos_inactivate.nonzero(), i))
                            self.pos_inactivate=pos_inactivate

                            total_pruned = total_pruned+1
                        if total_pruned>=self.prune_per_time:
                            print('pruned %i links'%self.prune_per_time)
                            break


            #Pruning agent
            if self.flag_AgentPruning:
                if it%self.AgentPruning_freq==0 and it>0:
                    for layer_i in range(self.num_layer):
                        pass

            if self.flag_slimming:
                if it%self.slimming_freq==0 and it>0:
                    pass



            if it%200==0:
                #Printing
                print('Iter %i'%it)
                print('action loss: %.6f, L1 loss: %.6f, Total: %.6f'%(self.action_loss.data.cpu(), self.L1_loss.data.cpu(), self.total_loss.data.cpu()))
                print('error rate: %.4f'%(self.error_rate))
                #Recording the result
                self.W_env_to_message_list.append(self.W_env_to_message.data.cpu())
                self.W_env_to_action_list.append(self.W_env_to_action.data.cpu())
                self.W_message_to_message_list.append(self.W_message_to_message.data.cpu())
                self.W_message_to_action_list.append(self.W_message_to_action.data.cpu())
                self.b_message_list.append(self.b_message.data.cpu())
                self.b_action_list.append(self.b_action.data.cpu())

                self.action_loss_list.append(self.action_loss.data.cpu())
                self.total_loss_list.append(self.total_loss.data.cpu())
                self.error_rate_list.append(self.error_rate.data.cpu())

                self.network_list.append(self.network.data.cpu())
                self.message_list.append(message.data.cpu())

                if self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts:
                    print('alpha L1 loss: %.6f'%self.L1_alpha)
                    self.DiscreteChoice_alpha_list.append(self.DiscreteChoice_alpha)
                    print('alpha:'+str(self.DiscreteChoice_alpha.data.cpu()[:,-1]))
                    print('w_ma:'+str(self.W_message_to_action.data.cpu().flatten()))


                print(self.action[:10].data.cpu())
                #print(W_message_to_action_grad)

                '''

                self.fig_loss = plt.figure()
                plt.plot(np.arange(len(self.total_loss_list) ) ,self.total_loss_list)
                plt.title('Loss'  )
                plt.show()
                #plt.draw()
                #plt.pause(.5)
                plt.close()
                '''


                if self.error_rate<1/self.minibatch_size:
                    print('Function learned')
                    if self.flag_minibatch:
                        print(' (for minibatch)')
                    if not (self.flag_minibatch or self.flag_pruning):
                        break

                if torch.isnan(self.total_loss):
                    print('Loss NaN')
                    break

                if it>10000 and np.all( np.array(self.error_rate_list[-10:])==.5  ) :
                    print('Error rate stuck at .5')
                    break



            if it%1000==0:
                lf = torch.nn.BCELoss()
                l = np.zeros([len(self.message_list),message.data.cpu().shape[1] ])
                '''
                for i in range( len(self.message_list) ):
                    m = self.message_list[i]
                    for j in range( self.message.data.cpu().shape[1] ):
                        l[i,j] = lf(m[:,j], self.env_output.flatten())

                for j in range(self.message.data.cpu().shape[1]):
                    plt.plot(np.arange(len(l[:,j]) ) ,l[:,j])
                    plt.title('%i-th message'%j  )
                    plt.show()
                    plt.close()
                '''
        self.fig_loss = plt.figure()
        plt.plot(np.arange(len(self.total_loss_list) ) ,self.total_loss_list)
        plt.title('Loss'  )
        #plt.show()
        #plt.draw()
        #plt.pause(.5)
        plt.close()





if __name__=="__main__":
    Description = 'A_little_careful_pruning'
    plt.ioff()

    exec_date = datetime.datetime.now(pytz.timezone('US/Mountain')).strftime('%B%d_%H%M')

    dirname ='./result_'+exec_date +'_' + Description

    createFolder(dirname)

    parameters_for_grid = {#'num_agent':[10],
                           'num_manager':[24,36],#15, #9, #24,36 #36,60,90
                           'num_environment':[6,12,36],  #6 12,24 #36,48
                           'num_actor':[1], #Not tested for >2 yet.
                           'dunbar_number':[4],##,6,8
                            'lr':[.0001],
                            'L1_coeff':[.0],#0.,
                            'n_it':[30000],#10000  30000
                            'message_unit':[nn.functional.relu],#[torch.sigmoid],
                            'action_unit':[torch.sigmoid],

                            'flag_DeepR': [False],#
                            'DeepR_layered': [False],
                            'DeepR_freq' : [5],
                            'DeepR_T' : [0.00001],

                            'flag_pruning':[True],
                            'type_pruning':['Slimming'], #'Random','Smallest','Slimming'
                            'pruning_freq':[300],

                            'flag_slimming':[False],# For agent slimming, not implemented yet.
                            'slimming_freq':[200],
                            'slimming_L1_coeff':[.0001],
                            'slimming_threshold':[.1],


                            'flag_AgentPruning':[False],
                            'AgentPruning_freq':[None],

                            'flag_DiscreteChoice': [False],
                            'flag_DiscreteChoice_Darts': [False],
                            'DiscreteChoice_freq': [10],
                            'DiscreteChoice_lr': [0.],
                            'DiscreteChoice_L1_coeff': [0.001],

                            'flag_ResNet':[False],
                            'flag_minibatch':[False],
                            'minibatch_size':[None],
                            'type_initial_network': ['FullyConnected'], #'layered_full''FullyConnected''layered_full'''layered_random'layered_full',,'layered_full' #'ConstrainedRandom',
                            'initial_network_depth':[5],
                            'flag_BatchNorm': [True],
                            'env_type': ['match_mod2_n'],#match_mod2
                            'env_n_region':[2]
                            #'width_seq':[[8,8,8]]
            }


    parameters_temp = list(ParameterGrid(parameters_for_grid))
    n_param_temp = len(parameters_temp)
    parameters_list = []

    for i in range(n_param_temp):
        if parameters_temp[i]['type_initial_network'] in ['layered_random','layered_full']:
            flag_layered = True
        else:
            flag_layered = False
        if parameters_temp[i]['type_initial_network'] in ['FullyConnected','layered_full']:
            flag_full = True
        else:
            flag_full = False

        if parameters_temp[i]['num_environment']>10:
            parameters_temp[i]['env_input_type'] = 'random'
            parameters_temp[i]['batchsize'] = 10000


        if parameters_temp[i]['dunbar_number']>parameters_temp[i]['num_environment']:
            continue
        parameters_temp[i]['num_agent'] = parameters_temp[i]['num_manager'] + parameters_temp[i]['num_actor']
        if not parameters_temp[i]['flag_DeepR']:
            parameters_temp[i]['DeepR_freq'] = None
            parameters_temp[i]['DeepR_T'] = None
            parameters_temp[i]['DeepR_layered'] = None

        if parameters_temp[i]['flag_DeepR']:
            if parameters_temp[i]['L1_coeff']==0.:
                continue
            if parameters_temp[i]['DeepR_layered']:
                if not flag_layered:
                    continue

        if not (parameters_temp[i]['flag_DiscreteChoice'] or parameters_temp[i]['flag_DiscreteChoice_Darts'] ):
            parameters_temp[i]['DiscreteChoice_freq'] = None
            parameters_temp[i]['DiscreteChoice_lr'] = None
            parameters_temp[i]['DiscreteChoice_L1_coeff'] = None



        if parameters_temp[i]['flag_pruning']:
            if not flag_full:
                continue
        if not parameters_temp[i]['flag_pruning']:
            parameters_temp[i]['type_pruning'] = None
            parameters_temp[i]['pruning_freq'] = None

        if not parameters_temp[i]['env_type']=='match_mod2_n':
            parameters_temp[i]['env_n_region'] = None



        if flag_layered:
            depth = parameters_temp[i]['initial_network_depth']
            num_manager = parameters_temp[i]['num_manager']
            dunbar_number = parameters_temp[i]['dunbar_number'] # last layer width is following dunber number
            manager_per_layer =  int((num_manager-dunbar_number)/(depth-1) )
            width_seq = np.append( np.ones(depth-1)*manager_per_layer, dunbar_number).astype(int)
            remaining_manager = int(num_manager - np.sum(width_seq) )
            for layer_to_add in range(remaining_manager):
                width_seq[layer_to_add] = width_seq[layer_to_add]+1
            parameters_temp[i]['width_seq'] = width_seq






            '''
            parameters_temp[i]['width_seq']  =[ int(parameters_temp[i]['num_manager']/3),int(parameters_temp[i]['num_manager']/3),int(parameters_temp[i]['num_manager']/3) ]
            if np.mod(parameters_temp[i]['num_manager'], 3) == 1:
                parameters_temp[i]['width_seq'][1] = parameters_temp[i]['width_seq'][1]+1
            if np.mod(parameters_temp[i]['num_manager'], 3) == 2:
                parameters_temp[i]['width_seq'][0] = parameters_temp[i]['width_seq'][0]+1
                parameters_temp[i]['width_seq'][1] = parameters_temp[i]['width_seq'][1]+1
            '''
        if not flag_layered:
            parameters_temp[i]['width_seq'] = None
            parameters_temp[i]['initial_network_depth'] = None




        if not parameters_temp[i]['flag_minibatch']:
            parameters_temp[i]['minibatch_size'] = None



        dup = False
        for p in parameters_list:
            if parameters_temp[i] is p:
                dup=True
        if dup is False:
            parameters_list.append(parameters_temp[i])



    n_rep = 20
    for param_i in range( len(parameters_list) ):


        org_parameters = parameters_list[param_i]
        print(org_parameters)
        final_action_loss_list = []
        final_network_list = []
        final_error_list = []
        print('********************'+'Setting'+str(param_i)+'********************')
        filename_setting = '/Param%i_final'%(param_i)

        with open(dirname+'/Time.txt','w') as text_file:
            text_file.write('parameter setting %i th out of %i'%(param_i,len(parameters_list))  )


        for rep_i in range(n_rep):

            torch.cuda.empty_cache()

            print('----Setting%i, rep%i-------'%(param_i,rep_i))
            start_time = time.time()

            filename_trial = '/Param%i_rep%i'%(param_i,rep_i)
            '''
            #Learns up to error rate: 0.0156 once in ten times.
            #Other times get stuck at the usual local minima.
            org = NCO_main(num_agent = 10, num_manager = 9, num_environment = 6, num_actor = 1, dunbar_number = 4,
                         lr = .01, L1_coeff = 0., n_it = 200000,
                         message_unit = torch.sigmoid, action_unit = torch.sigmoid,
                         flag_DeepR = False, DeepR_freq = 2000, DeepR_T = 0.00001,
                         flag_DiscreteChoice = False, flag_DiscreteChoice_Darts = False, DiscreteChoice_freq = 10, DiscreteChoice_lr = 0.,DiscreteChoice_L1_coeff = 0.001,
                         type_initial_network = 'layered_with_width', flag_BatchNorm = True, env_type = 'match_mod2',width_seq=[3,3,3]
                         )
            '''
            '''
            num_agent = 25
            num_manager = 24
            num_actor = 1

            org_parameters = {'num_agent':num_agent, 'num_manager':num_manager, 'num_environment':6, 'num_actor':num_actor,'dunbar_number':4,
                              'lr':.01, 'L1_coeff':0., 'n_it':5000,
                              'message_unit':torch.sigmoid, 'action_unit':torch.sigmoid,
                              'flag_DeepR': False, 'DeepR_freq' : 2000, 'DeepR_T' : 0.00001,
                              'flag_DiscreteChoice': False, 'flag_DiscreteChoice_Darts': False, 'DiscreteChoice_freq': 10, 'DiscreteChoice_lr': 0.,'DiscreteChoice_L1_coeff': 0.001,
                              'type_initial_network': 'layered_with_width', 'flag_BatchNorm': True, 'env_type': 'match_mod2','width_seq':[8,8,8]
                    }
            '''



            org = NCO_main(**org_parameters)

            org.to(device)

            org.func_Train()

            draw_network(num_environment=org_parameters['num_environment'], num_manager=org_parameters['num_manager'], num_actor=org_parameters['num_actor'],num_agent=org_parameters['num_agent'], network=org.network.data.cpu(), filename = dirname+filename_trial)



            #ConstrainedRandom

            '''
            pickle.dump(org.W_env_to_message_list, open(dirname+filename_trial+"_W_env_to_message_list.pickle","wb"))
            pickle.dump(org.W_env_to_action_list, open(dirname+filename_trial+"_W_env_to_action_list.pickle","wb"))
            pickle.dump(org.W_message_to_message_list, open(dirname+filename_trial+"_W_message_to_message_list.pickle","wb"))
            pickle.dump(org.W_message_to_action_list, open(dirname+filename_trial+"_W_message_to_action_list.pickle","wb"))
            pickle.dump(org.b_message_list, open(dirname+filename_trial+"_b_message_list.pickle","wb"))
            pickle.dump(org.b_action_list, open(dirname+filename_trial+"_b_action_list.pickle","wb"))

            pickle.dump(org.action_loss_list, open(dirname+filename_trial+"_action_loss_list.pickle","wb"))
            pickle.dump(org.total_loss_list, open(dirname+filename_trial+"_total_loss_list.pickle","wb"))
            pickle.dump(org.error_rate_list, open(dirname+filename_trial+"_error_rate_list.pickle","wb"))

            pickle.dump(org.network_list, open(dirname+filename_trial+"_network_list.pickle","wb"))

            pickle.dump(org.message_list, open(dirname+filename_trial+"_message_list.pickle","wb"))
            '''

            org.fig_loss.savefig(dirname+filename_trial+"_loss_graph.png")

            final_action_loss_list.append(org.action_loss_list[-1])
            final_network_list.append(org.network_list[-1])
            final_error_list.append(org.error_rate_list[-1])

            end_time = time.time()
            time_elapsed = end_time-start_time
            print('time per rep: ',time_elapsed)

            with open(dirname+'/Time.txt','w') as text_file:
                text_file.write('time per rep:%.2f'%time_elapsed  )
                text_file.write('time estimate per parameter:%.2f min'%(time_elapsed*n_rep/60)  )
                text_file.write('time estimate remaining for parameter:%.2f'%(time_elapsed*(n_rep-rep_i-1)/60)  )




        pickle.dump(org_parameters, open(dirname+filename_setting+"_org_parameters.pickle","wb"))
        pickle.dump( final_action_loss_list, open(dirname+filename_setting+"_final_action_loss_list.pickle","wb") )
        pickle.dump( final_network_list, open(dirname+filename_setting+"_final_network_list.pickle","wb") )
        pickle.dump( final_error_list, open(dirname+filename_setting+"_final_error_list.pickle","wb") )

    pickle.dump(parameters_list, open(dirname+"/Parameters_list.pickle","wb"))
