#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 20:20:26 2019

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

from NCO_functions import createFolder,Environment,gen_full_network

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)



num_agent = 10
num_manager = 9
num_environment = 6
num_actor = 1
dunbar_number = 4

dim_input_max = num_environment+num_manager

batchsize = 64#64
lr = 1e-2#1e-3
L1_coeff = 0.0001#.5
n_it = 100000


message_unit = torch.sigmoid#nn.functional.relu
#message_unit = nn.functional.relu
action_unit = torch.sigmoid

flag_DeepR = False
DeepR_freq = 2000
DeepR_T = 0.00001


'''
'''
env_class = Environment(batchsize,num_environment,num_agent,env_type='match_mod2',input_type='all_comb',flag_normalize=False)
env_class.create_env()
env_input = torch.Tensor(env_class.environment)
env_output = torch.Tensor(env_class.env_pattern)


#TEST - majority
##Learns after 500 iter
'''
env_input_np = np.random.choice([0,1],size=[batchsize,num_environment])
env_output_np = (np.sum(env_input_np,axis=1)>3).astype(np.float32).reshape([-1,1])
env_input = torch.Tensor(env_input_np)
env_output = torch.Tensor(env_output_np)
'''

network_full_np = gen_full_network(num_environment,num_manager,num_agent)
fanin_max_list = np.sum(network_full_np,axis=0).astype(np.int)

m_in_max = np.sum(network_full_np,axis=0)-num_environment
m_in_max[0] += .0000001
Wmm_init_np = np.random.normal( 0, 1/np.sqrt(m_in_max[:num_manager]), size = [num_manager,num_manager] )
Wma_init_np = np.random.normal( 0, 1/np.sqrt(m_in_max[num_manager:]), size = [num_manager,num_actor] )

#Weights. shape[0] is the dimension of inputs, shape[1] is the number of agents.
W_env_to_message = xavier_init([num_environment,num_manager])#Variable(torch.randn([num_environment,num_manager]), requires_grad=True)
W_env_to_action = xavier_init([num_environment,num_actor])#Variable(torch.randn([num_environment,num_actor]), requires_grad=True)

W_message_to_message = Variable(torch.randn([num_manager,num_manager]), requires_grad=True)
W_message_to_action = Variable(torch.randn([num_manager,num_actor]), requires_grad=True)

b_message = Variable(torch.randn([num_manager]), requires_grad=True)
b_action = Variable(torch.randn([num_actor]), requires_grad=True)

'''
message = Variable(torch.zeros([batchsize,num_manager]), requires_grad=False)
action_state = Variable(torch.zeros([batchsize,num_actor]), requires_grad=False)
action = Variable(torch.zeros([batchsize,num_actor]), requires_grad=False)
action_loss = Variable(torch.zeros([batchsize,num_actor]), requires_grad=False)

network = Variable(torch.ones([dim_input_max,num_agent]), requires_grad=False)
'''




params_to_optimize = [W_env_to_message,W_env_to_action,W_message_to_message,W_message_to_action,b_message,b_action]

solver = optim.Adam(params_to_optimize, lr=lr)
#loss = nn.BCELoss(reduction='sum')
loss = nn.BCEWithLogitsLoss(reduction='sum')



#Recording the information
W_env_to_message_list = []
W_env_to_action_list = []
W_message_to_message_list = []
W_message_to_action_list = []
b_message_list = []
b_action_list = []

action_loss_list=[]
total_loss_list=[]
error_rate_list=[]


network = torch.Tensor(network_full_np)
for it in range(n_it):
    message = torch.Tensor(torch.zeros([batchsize,num_manager]))
    action_state = torch.Tensor(torch.zeros([batchsize,num_actor]))
    action = torch.Tensor(torch.zeros([batchsize,num_actor]))
    action_loss = torch.Tensor(torch.zeros([batchsize,num_actor]))
    
    #network = torch.Tensor(torch.ones([dim_input_max,num_agent]))
    
    for i in range(num_manager):
        temp = b_message[i].repeat([batchsize, 1])
        #message[:,i] = ( message_unit(b_message[i].repeat([batchsize, 1]) + env_input @ (W_env_to_message[:,i] * network[:num_environment,i]).reshape([-1,1]) + message.clone() @ (W_message_to_message[:,i] * network[num_environment:,i]).reshape([-1,1])   ) ).flatten()
        message[:,i] = ( message_unit( temp + env_input @ (W_env_to_message[:,i] * network[:num_environment,i]).reshape([-1,1]) + message.clone() @ (W_message_to_message[:,i] * network[num_environment:,i]).reshape([-1,1])   ) ).flatten()
        
    #print(message[3,3])
        
    for i in range(num_actor):
        action_state[:,i] = (b_action[i].repeat([batchsize, 1]) + env_input @ (W_env_to_action[:,i] * network[:num_environment,num_manager+i]).reshape([-1,1]) + message.clone() @ (W_message_to_action[:,i] * network[num_environment:,num_manager+i]).reshape([-1,1]) ).flatten()  
        action[:,i] = action_unit(action_state[:,i]) 
        
        
    #action_loss = torch.nn.CrossEntropyLoss(action, env_output, reduction='sum')
    #action_loss = loss(action, env_output)
    action_loss = loss(action_state, env_output)
    L1_loss = torch.sum( torch.abs(W_env_to_message)) + torch.sum(torch.abs(W_env_to_action) )+torch.sum(torch.abs(W_message_to_message) )+ torch.sum(torch.abs(W_message_to_action) )#+ torch.sum(torch.abs(b_message)+torch.abs(b_action) ) 
    
    total_loss = action_loss+L1_loss*L1_coeff

     # Backward
    total_loss.backward()
    
    error_rate = torch.mean(torch.abs((action.data>.5).float() - env_output ) )

    # Update
    #solver.step()
    
    # Try manual GD
    '''
    
    p_i=0
    for p in params_to_optimize:
        with torch.no_grad():
        #if True:
            p = p - lr * p.grad
            p.requires_grad = True
        
        p_i = p_i+1
        #if p_i == 5:
        #    print(p)
    '''

    '''
    '''
    with torch.no_grad():
        W_env_to_message = W_env_to_message - lr * W_env_to_message.grad        
        W_env_to_action = W_env_to_action - lr * W_env_to_action.grad    
        W_message_to_message = W_message_to_message - lr * W_message_to_message.grad        
        W_message_to_action = W_message_to_action - lr * W_message_to_action.grad        
        b_message = b_message - lr * b_message.grad               
        b_action = b_action - lr * b_action.grad
        
        if flag_DeepR:
            W_env_to_message = W_env_to_message +torch.randn_like(W_env_to_message) * np.sqrt(2.*lr*DeepR_T)
            W_env_to_action = W_env_to_action +torch.randn_like(W_env_to_action) * np.sqrt(2.*lr*DeepR_T)
            W_message_to_message = W_message_to_message + torch.randn_like(W_message_to_message) * np.sqrt(2.*lr*DeepR_T)
            W_message_to_action = W_message_to_action + torch.randn_like(W_message_to_action) * np.sqrt(2.*lr*DeepR_T)
            
            W_env_to_message = torch.where(W_env_to_message>0,W_env_to_message,torch.zeros_like(W_env_to_message))
            W_message_to_message = torch.where(W_message_to_message>0,W_message_to_message,torch.zeros_like(W_message_to_message))
            W_env_to_action = torch.where(W_env_to_action>0,W_env_to_action,torch.zeros_like(W_env_to_action))
            W_message_to_action =torch.where(W_message_to_action>0,W_message_to_action,torch.zeros_like(W_message_to_action))


        W_env_to_message.requires_grad = True
        W_env_to_action.requires_grad = True
        W_message_to_message.requires_grad = True
        W_message_to_action.requires_grad = True
        b_message.requires_grad = True
        b_action.requires_grad = True

    # Housekeeping
    for p in params_to_optimize:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())    


    
    #print('b_message[1]:', b_message[1])
    #print('b_message.grad[1]:', b_message.grad[1])
    
    #print('b_action[1]:', b_action[0])
    #print('b_action.grad[1]:', b_message.grad[1])
    
    
    #print('W_env_to_message[0,0]:', W_env_to_message[0,0])
    #print('W_env_to_message.grad[0,0]:', W_env_to_message.grad[0,0])
    
    #DeepR
    if flag_DeepR:
        if it%DeepR_freq==0:
            print('****************Rewiring Network*********************')
            negative_m = torch.cat((W_env_to_message>0,W_message_to_message>0),dim=0).type(torch.FloatTensor) 
            negative_a = torch.cat((W_env_to_action>0,W_message_to_action>0),dim=0).type(torch.FloatTensor) 
            negative = torch.cat( (negative_m,negative_a),dim=1 )
            
            network = negative * network
            

            W_env_to_message = W_env_to_message +torch.randn_like(W_env_to_message) * np.sqrt(2.*lr*DeepR_T)
            W_env_to_action = W_env_to_action +torch.randn_like(W_env_to_action) * np.sqrt(2.*lr*DeepR_T)
            W_message_to_message = W_message_to_message + torch.randn_like(W_message_to_message) * np.sqrt(2.*lr*DeepR_T)
            W_message_to_action = W_message_to_action + torch.randn_like(W_message_to_action) * np.sqrt(2.*lr*DeepR_T)
            

            
            for i in range(num_agent):
                network_i = network[:,i]
                fanin_i = torch.sum(torch.abs(network_i) )
                if fanin_i<dunbar_number:
                    n_reactivate = int(dunbar_number-fanin_i)
                    pos_inactive = np.where(network_i==0)
                    pos_reactivate = np.random.choice(pos_inactive[0][(pos_inactive[0]<fanin_max_list[i])],[n_reactivate],replace=False)
                    network_i[pos_reactivate]=torch.Tensor(np.random.choice( [1.,-1.],len(pos_reactivate) ) )
                    network[:,i] = network_i

                    
            
            
    
    
    #Print
    
    if it%100==0:
        #Printing
        print('Iter %i'%it)
        print('action loss: %.4f, L1 loss: %.4f, Total: %.4f'%(action_loss.data, L1_loss.data, total_loss.data))
        print('error rate: %.4f'%(error_rate))
        #Recording the result
        W_env_to_message_list.append(W_env_to_message.data)
        W_env_to_action_list.append(W_env_to_action.data)
        W_message_to_message_list.append(W_message_to_message.data)
        W_message_to_action_list.append(W_message_to_action.data)
        b_message_list.append(b_message.data)
        b_action_list.append(b_action.data)
        
        action_loss_list.append(action_loss.data)
        total_loss_list.append(total_loss.data)
        error_rate_list.append(error_rate.data)
        
        if error_rate<1/batchsize:
            print('Function learned!')
            break





        