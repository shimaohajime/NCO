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

from sklearn.model_selection import ParameterGrid

from NCO_functions import createFolder,Environment,gen_full_network,gen_constrained_network,draw_network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


class NCO_main():
    def __init__(self, num_agent = 10, num_manager = 9, num_environment = 6, num_actor = 1, dunbar_number = 4, 
                 lr = .01, L1_coeff = 0., n_it = 200000, 
                 message_unit = torch.sigmoid, action_unit = torch.sigmoid, 
                 flag_DeepR = False, DeepR_freq = 2000, DeepR_T = 0.00001,DeepR_layered=False,
                 flag_pruning = False, pruning_freq = 100,
                 flag_DiscreteChoice = False, flag_DiscreteChoice_Darts = False, DiscreteChoice_freq = 10, DiscreteChoice_lr = 0.,DiscreteChoice_L1_coeff = 0.001,
                 type_initial_network = 'ConstrainedRandom', flag_BatchNorm = True, env_type = 'match_mod2',width_seq=None
                 ):
        
        #Basic parameters
        self.num_agent = num_agent
        self.num_manager = num_manager
        self.num_environment = num_environment
        self.num_actor = num_actor
        self.dunbar_number = dunbar_number
        self.message_unit = message_unit
        self.action_unit = action_unit
        
        self.dim_input_max = num_environment+num_manager
        
        #Learning parameters
        self.lr = lr#.1#1e-2#1e-3
        self.L1_coeff = L1_coeff#0.0001#.5
        self.n_it = n_it#1000000
        
        #DeepR
        self.flag_DeepR = flag_DeepR
        self.DeepR_freq = DeepR_freq
        self.DeepR_T = DeepR_T
        self.DeepR_layered = DeepR_layered
        
        #Pruning
        self.flag_pruning = flag_pruning
        self.pruning_freq = pruning_freq
        
        #Discrete choice and Darts
        self.flag_DiscreteChoice = flag_DiscreteChoice
        self.flag_DiscreteChoice_Darts = flag_DiscreteChoice_Darts
        self.DiscreteChoice_freq = DiscreteChoice_freq
        self.DiscreteChoice_lr = DiscreteChoice_lr
        self.DiscreteChoice_L1_coeff = DiscreteChoice_L1_coeff
        
        
        #Initial values
        self.type_initial_network = type_initial_network#'FullyConnected'#'ConstrainedRandom'
        
        #Batch normalization
        self.flag_BatchNorm = flag_BatchNorm
        
        #Type of environment generation
        self.env_type = env_type
        
        
        #Generate network and environment
        self.network_const_env = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number)
        self.network_full_np = gen_full_network(num_environment,num_manager,num_agent)
        self.fanin_max_list = np.sum(self.network_full_np,axis=0).astype(np.int)

        if type_initial_network is 'ConstrainedRandom':
            self.network_const_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number)
            self.network = torch.Tensor(np.abs(self.network_const_np) )
        elif type_initial_network is 'FullyConnected':
            self.network = torch.Tensor(np.abs(self.network_full_np) )
        elif type_initial_network is 'FullyConnected_NoDirectEnv':
            self.network = torch.Tensor(np.abs(self.network_full_np) )
            self.network[:num_environment, num_manager:]=0            
        elif type_initial_network is 'layered_random':
            self.network_full_layered_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number, type_network='layered_full',width_seq=width_seq)            
            self.network_full_layered = torch.Tensor(self.network_full_layered_np)
            self.network_const_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number, type_network='layered_random',width_seq=width_seq)            
            self.network = torch.Tensor(np.abs(self.network_const_np) )
        elif type_initial_network is 'layered_full':
            self.network_full_layered_np = gen_constrained_network(num_environment,num_manager,num_agent,dunbar_number, type_network='layered_full',width_seq=width_seq)            
            self.network_full_layered = torch.Tensor(self.network_full_layered_np)
            self.network = torch.Tensor(np.abs(self.network_full_layered_np) )
            
            
            
        env_class = Environment(batchsize=None,num_environment=self.num_environment,num_agents=num_agent,num_manager=num_manager,num_actor=num_actor,env_type=env_type,input_type='all_comb',flag_normalize=False,env_network=self.network_const_env)
        env_class.create_env()
        
        self.env_input_np = env_class.environment
        self.env_output_np = env_class.env_pattern
        self.env_input = torch.Tensor(self.env_input_np)
        self.env_output = torch.Tensor(self.env_output_np)

        #Batchsize                
        self.batchsize = self.env_input.shape[0]#64


        m_in_max = np.sum(self.network_full_np,axis=0)-num_environment
        m_in_max[0] += .0000001
        
        
        #Create learnable parameters
        ## Weights and Biases
        self.Wmm_init_np = np.random.normal( 0, 1/np.sqrt(m_in_max[:num_manager]), size = [num_manager,num_manager] )
        self.Wma_init_np = np.random.normal( 0, 1/np.sqrt(m_in_max[num_manager:]), size = [num_manager,num_actor] )        
        #Weights. shape[0] is the dimension of inputs, shape[1] is the number of agents.
        self.W_env_to_message = xavier_init([num_environment,num_manager]) #Variable(torch.randn([num_environment,num_manager]), requires_grad=True)
        self.W_env_to_action = xavier_init([num_environment,num_actor])#Variable(torch.randn([num_environment,num_actor]), requires_grad=True)
        self.W_message_to_message = Variable(torch.randn([num_manager,num_manager]), requires_grad=True)
        self.W_message_to_action = Variable(torch.randn([num_manager,num_actor]), requires_grad=True)        
        self.b_message = Variable(torch.zeros([num_manager]), requires_grad=True)
        self.b_action = Variable(torch.randn([num_actor]), requires_grad=True)

        ##Parameters for batch normalization
        if flag_BatchNorm:
            self.BatchNorm_gamma_message_to_message = Variable(torch.randn([num_manager,num_manager]), requires_grad=True)
            self.BatchNorm_gamma_message_to_action = Variable(torch.randn([num_manager,num_actor]), requires_grad=True)
            self.BatchNorm_gamma_env_to_message = Variable(torch.randn([num_environment,num_manager]), requires_grad=True)
            self.BatchNorm_gamma_env_to_action = Variable(torch.randn([num_environment,num_actor]), requires_grad=True)
            
            self.BatchNorm_beta_message_to_message = Variable(torch.randn([num_manager,num_manager]), requires_grad=True)
            self.BatchNorm_beta_message_to_action = Variable(torch.randn([num_manager,num_actor]), requires_grad=True)
            self.BatchNorm_beta_env_to_message = Variable(torch.randn([num_environment,num_manager]), requires_grad=True)
            self.BatchNorm_beta_env_to_action = Variable(torch.randn([num_environment,num_actor]), requires_grad=True)
        
            self.BatchNorm_eps = 1e-5
            
        #Parameters for discrete chocie
        if flag_DiscreteChoice or flag_DiscreteChoice_Darts:
            self.DiscreteChoice_alpha = Variable(torch.zeros_like(self.network), requires_grad=True)#Variable(torch.randn_like(network), requires_grad=True)


        self.params_to_optimize = [self.W_env_to_message,self.W_env_to_action,self.W_message_to_message,self.W_message_to_action,self.b_message,self.b_action]
        
        if flag_DiscreteChoice or flag_DiscreteChoice_Darts:
            self.params_to_optimize.append(self.DiscreteChoice_alpha)
        if flag_BatchNorm:
            self.params_to_optimize.extend([self.BatchNorm_gamma_message_to_message,self.BatchNorm_gamma_message_to_action,self.BatchNorm_gamma_env_to_message,self.BatchNorm_gamma_env_to_action,self.BatchNorm_beta_message_to_message,self.BatchNorm_beta_message_to_action,self.BatchNorm_beta_env_to_message,self.BatchNorm_beta_env_to_action])
            
        
        #
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
        X_BachNorm =  gamma * ( (X-torch.mean(X,dim=0) )/ (torch.std(X,dim=0)+eps ) ) +beta
        
        return X_BachNorm


    def func_Train(self):
        for it in range(self.n_it):
            '''
            if it>5000:
                lr = 1.
            if it>20000:
                lr = .01
            '''
            
            #Initialize message and action
            self.message = torch.Tensor(torch.zeros([self.batchsize,self.num_manager]))
            self.action_state = torch.Tensor(torch.zeros([self.batchsize,self.num_actor]))
            self.action = torch.Tensor(torch.zeros([self.batchsize,self.num_actor]))
            self.action_loss = torch.Tensor(torch.zeros([self.batchsize,self.num_actor]))
            
            #Create messages sequentially
            for i in range(self.num_manager):
                temp = self.b_message[i].repeat([self.batchsize, 1])
                        
                if not self.flag_BatchNorm:
                    self.env_input_i = self.env_input
                    self.message_in_i = self.message.clone()
                elif self.flag_BatchNorm:
                    self.env_input_i = self.func_BatchNorm(self.env_input, self.BatchNorm_gamma_env_to_message[:,i],self.BatchNorm_beta_env_to_message[:,i], self.BatchNorm_eps)
                    self.message_in_i = self.func_BatchNorm(self.message.clone(), self.BatchNorm_gamma_message_to_message[:,i],self.BatchNorm_beta_message_to_message[:,i], self.BatchNorm_eps)
                
                if not (self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts):
                    self.message[:,i] = ( self.message_unit( temp + self.env_input_i @ (self.W_env_to_message[:,i] * self.network[:self.num_environment,i]).reshape([-1,1]) + self.message_in_i @ (self.W_message_to_message[:,i] * self.network[self.num_environment:,i]).reshape([-1,1])   ) ).flatten()
                elif self.flag_DiscreteChoice:
                    choice_prob_m = nn.functional.softmax(self.DiscreteChoice_alpha[:,i] + (self.network[:,i]-1.)*1000000  )
                    self.message[:,i] = ( self.message_unit( temp + self.env_input_i @ (self.W_env_to_message[:,i] * self.network[:self.num_environment,i] * choice_prob_m[:self.num_environment]).reshape([-1,1]) + self.message_in_i @ (self.W_message_to_message[:,i] * self.network[self.num_environment:,i] * choice_prob_m[self.num_environment:]).reshape([-1,1])   ) ).flatten()
                elif self.flag_DiscreteChoice_Darts:
                    choice_prob_m = 1./(torch.exp(self.DiscreteChoice_alpha[:,i])+1.)
                    self.message[:,i] = ( self.message_unit( temp + self.env_input_i @ (self.W_env_to_message[:,i] * self.network[:self.num_environment,i] * choice_prob_m[:self.num_environment]).reshape([-1,1]) + self.message_in_i @ (self.W_message_to_message[:,i] * self.network[self.num_environment:,i] * choice_prob_m[self.num_environment:]).reshape([-1,1])   ) ).flatten()
    
            #Create action
            for j in range(self.num_actor):
                if not self.flag_BatchNorm:
                    self.env_input_j = self.env_input
                    self.message_in_j = self.message.clone()
                elif self.flag_BatchNorm:
                    self.env_input_j = self.func_BatchNorm(self.env_input, self.BatchNorm_gamma_env_to_action[:,j],self.BatchNorm_beta_env_to_action[:,j], self.BatchNorm_eps)
                    self.message_in_j = self.func_BatchNorm(self.message.clone(), self.BatchNorm_gamma_message_to_action[:,j], self.BatchNorm_beta_message_to_action[:,j], self.BatchNorm_eps)
                
                
                if not (self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts):        
                    self.action_state[:,j] = (self.b_action[j].repeat([self.batchsize, 1]) + self.env_input_j @ (self.W_env_to_action[:,j] * self.network[:self.num_environment,self.num_manager+j]).reshape([-1,1]) + self.message_in_j @ (self.W_message_to_action[:,j] * self.network[self.num_environment:,self.num_manager+j]).reshape([-1,1]) ).flatten()  
                elif self.flag_DiscreteChoice:
                    choice_prob_a = nn.functional.softmax(self.DiscreteChoice_alpha[:,self.num_manager+j] + (self.network[:,self.num_manager+j]-1.)*1000000  )
                    self.action_state[:,j] = (self.b_action[j].repeat([self.batchsize, 1]) + self.env_input_j @ (self.W_env_to_action[:,j] * self.network[:self.num_environment,self.num_manager+j] * choice_prob_a[:self.num_environment]).reshape([-1,1]) + self.message_in_j @ (self.W_message_to_action[:,j] * self.network[self.num_environment:,self.num_manager+j] * choice_prob_a[self.num_environment:]).reshape([-1,1]) ).flatten()  
                elif self.flag_DiscreteChoice_Darts:
                    choice_prob_a = 1./(torch.exp(self.DiscreteChoice_alpha[:,self.num_manager+j]) +1.)
                    self.action_state[:,j] = (self.b_action[j].repeat([self.batchsize, 1]) + self.env_input_j @ (self.W_env_to_action[:,j] * self.network[:self.num_environment,self.num_manager+j] * choice_prob_a[:self.num_environment]).reshape([-1,1]) + self.message_in_j @ (self.W_message_to_action[:,j] * self.network[self.num_environment:,self.num_manager+j] * choice_prob_a[self.num_environment:]).reshape([-1,1]) ).flatten()  
                    
                self.action[:,j] = self.action_unit(self.action_state[:,j]) 

            #Calculate loss and backprop
            self.action_loss = self.loss(self.action_state, self.env_output)
            self.L1_loss = torch.sum( torch.abs(self.W_env_to_message)) + torch.sum(torch.abs(self.W_env_to_action) )+torch.sum(torch.abs(self.W_message_to_message) )+ torch.sum(torch.abs(self.W_message_to_action) )#+ torch.sum(torch.abs(b_message)+torch.abs(b_action) ) 
            
            self.total_loss = self.action_loss+self.L1_loss*self.L1_coeff
            if self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts:
                self.L1_alpha = torch.sum(torch.abs(self.DiscreteChoice_alpha) )
                self.total_loss = self.total_loss+self.L1_alpha*self.DiscreteChoice_L1_coeff
                
                
            self.total_loss.backward()
            self.error_rate = torch.mean(torch.abs((self.action.data>.5).float() - self.env_output ) )
            

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
                
                if self.flag_DeepR:
                    self.W_env_to_message = self.W_env_to_message +torch.randn_like(self.W_env_to_message) * np.sqrt(2.*self.lr*self.DeepR_T)
                    self.W_env_to_action = self.W_env_to_action +torch.randn_like(self.W_env_to_action) * np.sqrt(2.*self.lr*self.DeepR_T)
                    self.W_message_to_message = self.W_message_to_message + torch.randn_like(self.W_message_to_message) * np.sqrt(2.*self.lr*self.DeepR_T)
                    self.W_message_to_action = self.W_message_to_action + torch.randn_like(self.W_message_to_action) * np.sqrt(2.*self.lr*self.DeepR_T)
                    
                    self.W_env_to_message = torch.where(self.W_env_to_message>0,self.W_env_to_message,torch.zeros_like(self.W_env_to_message))
                    self.W_message_to_message = torch.where(self.W_message_to_message>0,self.W_message_to_message,torch.zeros_like(self.W_message_to_message))
                    self.W_env_to_action = torch.where(self.W_env_to_action>0,self.W_env_to_action,torch.zeros_like(self.W_env_to_action))
                    self.W_message_to_action =torch.where(self.W_message_to_action>0,self.W_message_to_action,torch.zeros_like(self.W_message_to_action))
                    
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
                        #print('alpha mean:'+str( torch.mean(DiscreteChoice_alpha.data) ))
                        #print('alpha var:'+str( torch.var(DiscreteChoice_alpha.data) ))

            # Housekeeping
            for p in self.params_to_optimize:
                if p.grad is not None:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())    
                    
                    
            #DeepR rewiring network
            if self.flag_DeepR:
                if it%self.DeepR_freq==0 and it>0:
                    #print('****************Rewiring Network*********************')
                    negative_m = torch.cat((self.W_env_to_message>0,self.W_message_to_message>0),dim=0).type(torch.FloatTensor) 
                    negative_a = torch.cat((self.W_env_to_action>0,self.W_message_to_action>0),dim=0).type(torch.FloatTensor) 
                    negative = torch.cat( (negative_m,negative_a),dim=1 )                    
                    self.network = negative * self.network
                    for i in range(self.num_agent):
                        network_i = self.network[:,i]
                        fanin_i = torch.sum(torch.abs(network_i) )
                        if fanin_i<self.dunbar_number:
                            
                            n_reactivate = int(self.dunbar_number-fanin_i)
                            if self.DeepR_layered is False:                            
                                pos_inactive = np.where(network_i==0)
                                pos_reactivate = np.random.choice(pos_inactive[0][(pos_inactive[0]<self.fanin_max_list[i])],[n_reactivate],replace=False)
                                network_i[pos_reactivate]=torch.Tensor(np.random.choice( [1.,-1.],len(pos_reactivate) ) )
                            if self.DeepR_layered is True:
                                pos_inactive = np.where( (network_i==0) * (self.network_full_layered[:,i]==1 ) )
                                self.pos_inactive = pos_inactive
                                if n_reactivate<=len(pos_inactive[0][(pos_inactive[0]<self.fanin_max_list[i])]):
                                    pos_reactivate = np.random.choice(pos_inactive[0][(pos_inactive[0]<self.fanin_max_list[i])],[n_reactivate],replace=False)
                                    network_i[pos_reactivate]=torch.Tensor(np.random.choice( [1.,-1.],len(pos_reactivate) ) )
                                
                            self.network[:,i] = network_i
                            
            #Pruning network
            if self.flag_pruning:
                if it%self.pruning_freq==0 and it>0:
                    for i in range(self.num_agent):
                        network_i = self.network[:,i]
                        fanin_i = torch.sum(torch.abs(network_i) )
                        if fanin_i>self.dunbar_number: 
                            n_inactivate = 1
                            pos_active = np.where(network_i!=0)
                            pos_inactivate = np.random.choice(pos_active[0][(pos_active[0]<self.fanin_max_list[i])],[n_inactivate],replace=False)
                            network_i[pos_inactivate]=torch.zeros(len(pos_inactivate))
                            
            if it%200==0:
                #Printing
                print('Iter %i'%it)
                print('action loss: %.6f, L1 loss: %.6f, Total: %.6f'%(self.action_loss.data, self.L1_loss.data, self.total_loss.data))
                print('error rate: %.4f'%(self.error_rate))
                #Recording the result
                self.W_env_to_message_list.append(self.W_env_to_message.data)
                self.W_env_to_action_list.append(self.W_env_to_action.data)
                self.W_message_to_message_list.append(self.W_message_to_message.data)
                self.W_message_to_action_list.append(self.W_message_to_action.data)
                self.b_message_list.append(self.b_message.data)
                self.b_action_list.append(self.b_action.data)
                
                self.action_loss_list.append(self.action_loss.data)
                self.total_loss_list.append(self.total_loss.data)
                self.error_rate_list.append(self.error_rate.data)
                
                self.network_list.append(self.network.data)
                self.message_list.append(self.message.data)
                
                if self.flag_DiscreteChoice or self.flag_DiscreteChoice_Darts:
                    print('alpha L1 loss: %.6f'%self.L1_alpha)
                    self.DiscreteChoice_alpha_list.append(self.DiscreteChoice_alpha)
                    print('alpha:'+str(self.DiscreteChoice_alpha.data[:,-1]))
                    print('w_ma:'+str(self.W_message_to_action.data.flatten()))
        
        
                print(self.action[:10])
                #print(W_message_to_action_grad)

                self.fig_loss = plt.figure()
                plt.plot(np.arange(len(self.total_loss_list) ) ,self.total_loss_list)
                plt.title('Loss'  )
                plt.show()
                plt.close()


                if self.error_rate<1/self.batchsize:
                    print('Function learned!')
                    break
            
                if torch.isnan(self.total_loss):
                    print('Loss NaN')
                    break
                
                    

            if it%1000==0:
                lf = torch.nn.BCELoss()
                l = np.zeros([len(self.message_list),self.message.data.shape[1] ])
                '''
                for i in range( len(self.message_list) ):
                    m = self.message_list[i]
                    for j in range( self.message.data.shape[1] ):
                        l[i,j] = lf(m[:,j], self.env_output.flatten())
                       
                for j in range(self.message.data.shape[1]):
                    plt.plot(np.arange(len(l[:,j]) ) ,l[:,j])
                    plt.title('%i-th message'%j  )
                    plt.show()
                    plt.close()
                '''
        
        
                
                
if __name__=="__main__":
    Description = 'Iterate_DeepR'

    exec_date = datetime.datetime.now(pytz.timezone('US/Mountain')).strftime('%B%d_%H%M')
    
    dirname ='./result_'+exec_date +'_' + Description
    
    createFolder(dirname)
    
    parameters_for_grid = {#'num_agent':[10], 
                           'num_manager':[9,24],#15, 
                           'num_environment':[6], 
                           'num_actor':[1],
                           'dunbar_number':[4],#2,
                            'lr':[.001], 
                            'L1_coeff':[.01],#0., 
                            'n_it':[10000],#10000
                            'message_unit':[nn.functional.relu],#[torch.sigmoid], 
                            'action_unit':[torch.sigmoid], 
                            'flag_DeepR': [True,False],#
                            'DeepR_layered': [True,False],
                            'DeepR_freq' : [5], 
                            'DeepR_T' : [0.00001],
                            'flag_pruning':[False],
                            'pruning_freq':[100],
                            'flag_DiscreteChoice': [False], 
                            'flag_DiscreteChoice_Darts': [False], 
                            'DiscreteChoice_freq': [10], 
                            'DiscreteChoice_lr': [0.],
                            'DiscreteChoice_L1_coeff': [0.001],
                            'type_initial_network': ['layered_full','layered_random'], #'layered_full' #'ConstrainedRandom',
                            'flag_BatchNorm': [True], 
                            'env_type': ['match_mod2'],
                            #'width_seq':[[8,8,8]]
            }
    
    
    parameters_temp = list(ParameterGrid(parameters_for_grid))
    n_param_temp = len(parameters_temp)
    parameters_list = []
    
    for i in range(n_param_temp):
        if parameters_temp[i]['dunbar_number']>parameters_temp[i]['num_environment']:
            pass
        parameters_temp[i]['num_agent'] = parameters_temp[i]['num_manager'] + parameters_temp[i]['num_actor']
        if not parameters_temp[i]['flag_DeepR']:
            parameters_temp[i]['DeepR_freq'] = None
            parameters_temp[i]['DeepR_T'] = None
        if parameters_temp[i]['flag_DeepR']:
            if parameters_temp[i]['L1_coeff']==0.:
                pass
            if parameters_temp[i]['DeepR_layered']:
                if parameters_temp[i]['type_initial_network'] in ['ConstrainedRandom', 'FullyConnected', 'FullyConnected_NoDirectEnv']:
                    pass
                
        if not (parameters_temp[i]['flag_DiscreteChoice'] or parameters_temp[i]['flag_DiscreteChoice_Darts'] ):
            parameters_temp[i]['DiscreteChoice_freq'] = None
            parameters_temp[i]['DiscreteChoice_lr'] = None
            parameters_temp[i]['DiscreteChoice_L1_coeff'] = None
            
        if parameters_temp[i]['flag_pruning']:
            parameters_temp[i]['type_initial_network'] = 'FullyConnected'
            
            
        if parameters_temp[i]['type_initial_network'] in ['layered_random','layered_full']:
            parameters_temp[i]['width_seq']  =[ int(parameters_temp[i]['num_manager']/3),int(parameters_temp[i]['num_manager']/3),int(parameters_temp[i]['num_manager']/3) ]
            if np.mod(parameters_temp[i]['num_manager'], 3) == 1:
                parameters_temp[i]['width_seq'][1] = parameters_temp[i]['width_seq'][1]+1
            if np.mod(parameters_temp[i]['num_manager'], 3) == 2:
                parameters_temp[i]['width_seq'][0] = parameters_temp[i]['width_seq'][0]+1
                parameters_temp[i]['width_seq'][1] = parameters_temp[i]['width_seq'][1]+1
        
        dup = False
        for p in parameters_list:
            if parameters_temp[i]==p:
                dup=True
        if dup is False:
            parameters_list.append(parameters_temp[i])
                
        

    n_rep = 50
    for param_i in range( len(parameters_list) ):
        org_parameters = parameters_list[param_i]
        final_action_loss_list = []
        final_network_list = []   
        final_error_list = []                 
        print('********************'+'Setting'+str(param_i)+'********************')
        filename_setting = '/Param%i_final'%(param_i)
        
        for rep_i in range(n_rep):
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
            org.func_Train()
        
            draw_network(num_environment=org_parameters['num_environment'], num_manager=org_parameters['num_manager'], num_actor=org_parameters['num_actor'],num_agent=org_parameters['num_agent'], network=org.network, filename = dirname+filename_trial)
        
            
            
            #ConstrainedRandom
            
            
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
        
            
            org.fig_loss.savefig(dirname+filename_trial+"_loss_graph.png")
            
            final_action_loss_list.append(org.action_loss_list[-1])
            final_network_list.append(org.network_list[-1])
            final_error_list.append(org.error_rate_list[-1])
            
            end_time = time.time()
            time_elapsed = end_time-start_time
            print('time per rep: ',time_elapsed)
        pickle.dump(org_parameters, open(dirname+filename_setting+"_org_parameters.pickle","wb"))
        pickle.dump( final_action_loss_list, open(dirname+filename_setting+"_final_action_loss_list.pickle","wb") )
        pickle.dump( final_network_list, open(dirname+filename_setting+"_final_network_list.pickle","wb") )
        pickle.dump( final_error_list, open(dirname+filename_setting+"_final_error_list.pickle","wb") )
        
    pickle.dump(parameters_list, open(dirname+filename_setting+"_parameters_list.pickle","wb"))
