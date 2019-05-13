#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:49:16 2019

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
from torch import tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import time
import datetime
import pytz
import pickle

from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('Agg')

from sklearn.model_selection import ParameterGrid


'''
Reservoir-ish recurrent network model
'''

class ESN(Module):
    def __init__(self,num_agent=10, num_environment_input=6, num_environment_output=1, num_sample=10,
                 alpha=.5, rho=.8,network_aa=None, network_ia=None, network_ao=None,
                 W_x=None, W_in=None, W_out=None,
                 x_init=None,
                 flag_train=False, flag_dynamic=False,n_it=100,loss_function=nn.BCELoss(), lr=.001, num_iter_converge=100,
                 learning_method='gradient',beta=0.,
                 flag_train_W_x = False, flag_train_W_in=False,
                 flag_print_loss=True
                 ):
        super(ESN, self).__init__()
        
        '''
        Data
        '''        
        self.num_sample = num_sample


        '''
        Reservoir structure
        '''
        self.num_agent = num_agent
        self.num_environment_input = num_environment_input
        self.num_environment_output = num_environment_output
        self.alpha = alpha
        self.rho = rho
        if x_init is None:
            self.x = tensor(np.random.randn(self.num_sample,self.num_agent).astype(np.float32))
            self.x_init = self.x.clone()
        else:
            self.x = x_init
            self.x_init = x_init        
        self.y = tensor(np.zeros([self.num_sample,self.num_environment_output]).astype(np.float32))
        

        '''
        Weights
        '''
        #Initialize W_x
        #(i,j) of W: message from j to i
        if W_x is None:
            W_x_temp = tensor( np.random.randn(num_agent, num_agent).astype(np.float32) )
        else:
            W_x_temp = W_x            
        W_x_temp = W_x_temp/torch.abs(torch.eig(W_x_temp)[0][0][0] ) * self.rho
        if not flag_train_W_x:
            self.W_x = nn.Parameter( W_x_temp )
        if flag_train_W_x: 
            self.W_x = tensor( np.random.randn(num_agent, num_agent).astype(np.float32) )
            
        #Initialize W_in
        if W_in is None:
            W_in_temp = tensor( np.random.randn( num_agent, num_environment_input ).astype(np.float32) )
        else:
            W_in_temp = W_in                      
        if not flag_train_W_in:
            self.W_in = W_in_temp
        if flag_train_W_in:
            self.W_in = nn.Parameter( W_in_temp )
                
        if W_out is None:
            self.W_out = nn.Parameter( tensor( np.random.randn(num_environment_output, num_agent).astype(np.float32) ) )
        else:
            self.W_out = nn.Parameter( tensor(W_out) )
        
            
            
        '''
        Network
        '''
        if (type(network_ia) is torch.Tensor) or (network_ia is None):
            self.network_ia = network_ia
        else:
            self.network_ia = torch.tensor(network_ia,dtype=torch.float)            
        if (type(network_aa) is torch.Tensor) or (network_aa is None):
            self.network_aa = network_aa
        else:
            self.network_aa = torch.tensor(network_aa,dtype=torch.float)
        if (type(network_ao) is torch.Tensor) or (network_ao is None):
            self.network_ao = network_ao
        else:
            self.network_ao = torch.tensor(network_ao,dtype=torch.float)
        
        
            
        '''
        Parameters for optimization
        '''
        self.learning_method = learning_method
        self.beta = beta
        self.flag_train = flag_train
        if flag_train:
        
            self.params_to_optimize = []
            self.W_out.requires_grad=True
            self.params_to_optimize.append(self.W_out)
            
            self.n_it = n_it
            self.flag_dynamic=flag_dynamic
            self.loss_function = loss_function
            self.optimizer = optim.Adam(self.params_to_optimize, lr=lr)
        else:
            self.W_out.requires_grad=False
        self.flag_dynamic = flag_dynamic
        if not flag_dynamic:
            self.num_iter_converge = num_iter_converge
            
            
        '''
        '''
        self.flag_print_loss = flag_print_loss
            
    def reset_x(self):
        self.x = self.x_init.clone()
    
    def update(self,e_in,x=None):
        #e_in: environment input. num_sample (or minibatch size) by num_environment_input
        if x is None:
            x=self.x
            
        if self.network_aa is None:
            z = torch.sigmoid( torch.mm(self.W_x, x.t()).t() + torch.mm(self.W_in, e_in.t()).t()  ) 
        else:
            z = torch.sigmoid( torch.mm(self.W_x*self.network_aa, x.t()).t()  + torch.mm(self.W_in*self.network_ia, e_in.t()).t()   )
        x_next = self.alpha*z +(1-self.alpha)*x
        
        if self.network_ao is None:        
            y_next = torch.sigmoid( torch.mm(self.W_out, x_next.t() ).t() )
        else:        
            y_next = torch.sigmoid( torch.mm(self.W_out*self.network_ao, x_next.t() ).t() )
        self.x = x_next
        self.y = y_next
        
        return x_next, y_next
        
    def process_sequence(self,input_sequence):
        #input_seuqnce:  num_sample by num_environment_input by time step
        T = input_sequence.shape[-1]
        minibatch_size = input_sequence[0]
        output_sequence = torch.zeros( [minibatch_size,self.num_environment_output, T] )
        for t in range(T):
            e_in = input_sequence[:,:,t]
            self.x, self.y = self.update(e_in)
            output_sequence[:,:,t] = self.y        
        return output_sequence
    
    def train(self,train_loader ):       
        if self.learning_method is 'gradient':
            for it in range(self.n_it):
                for e_in_seq, e_out_seq in train_loader:
                    self.reset_x()
                    if self.flag_dynamic:
                        out_hat = self.process_sequence(e_in_seq.type('torch.FloatTensor'))
                    else:
                        for t in range(self.num_iter_converge):
                            e_in = e_in_seq.type('torch.FloatTensor')
                            self.x, out_hat = self.update(e_in)
                    self.optimizer.zero_grad()
                    
                    #self.e_out_seq=e_out_seq
                    #self.out_hat=out_hat
                    loss = self.loss_function(out_hat, e_out_seq)                
                    loss.backward()
                    self.optimizer.step()
                    self.loss = loss.data
                if self.flag_print_loss:
                    print('---iter: %i---'%it)
                    print('Loss %.4f'%loss)
        elif self.learning_method is 'ridge':
            for e_in_seq, e_out_seq in train_loader:
                self.reset_x()
                if self.flag_dynamic:
                    out_hat = self.process_sequence(e_in_seq)
                else:
                    for t in range(self.num_iter_converge):
                        e_in = e_in_seq.type('torch.FloatTensor')
                        self.x, out_hat = self.update(e_in)
                        self.e_in = e_in
                if not self.flag_dynamic:
                    e_out = e_out_seq.type('torch.FloatTensor')
                    self.W_out_ridge = ( ( self.x.t().mm(self.x)+self.beta*torch.eye(self.num_agent)  ).inverse().mm(  self.x.t().mm(e_out)  )   ).t()
                    if self.network_ao is None:        
                        out_hat = torch.sigmoid( torch.mm(self.W_out_ridge, self.x.t() ).t() )
                    else:        
                        out_hat = torch.sigmoid( torch.mm(self.W_out_ridge*self.network_ao, self.x.t() ).t() )
                        
                    self.out_hat = out_hat
                    self.e_out_seq=e_out_seq
                        
                    loss = self.loss_function(out_hat, e_out)     
                
                print('Loss %.4f'%loss)
                
            
            
            
                        
            
                    
                    
        
        

class OrgTask(Dataset):

    def __init__(self, num_environment, batchsize):
        super(OrgTask, self).__init__()
        
        self.batchsize=batchsize
        self.num_environment=num_environment        
        self.environment_input = np.random.randint(2,size = [self.batchsize, self.num_environment])
        
        left = self.environment_input[:,:int(self.num_environment/2)]
        right = self.environment_input[:,int(self.num_environment/2):]
        lmod = np.mod(np.sum(left,axis=1),2)
        rmod = np.mod(np.sum(right,axis=1),2)
        self.environment_output = (lmod==rmod).astype(np.float32).reshape([-1,1])

        
    def __len__(self):
        return self.batchsize

    def __getitem__(self, index):
        return  self.environment_input[index,:], self.environment_output[index,:]  



class DeepESN(Module):
    def __init__(self):
        super(DeepESN, self).__init__()
        
        #Parameter for Deep ESN
        self.num_layer = num_layer
        
        self.W_between_layer = tensor( np.zeros([num_agent,num_agent,num_layer-1]) )
        self.W_out = ensor( np.zeros([num_environment_output ,num_agent*num_layer]) )

        self.num_environment_input = num_environment_input
        self.num_environment_output = num_environment_output
        self.num_sample = num_sample
        
        #Parameters for each layer
        self.num_agent = num_agent
        self.alpha = alpha

        self.layer_list = []
        self.x_all = torch.zeros( [num_sample, num_agent*num_layer] )
        for l in range(num_layer):
            if l==0:
                esn = ESN(num_agent=num_agent, num_environment_input=num_environment_input, num_environment_output=num_environment_output, num_sample=num_sample,
                 alpha=alpha,  W_x=None, W_in=None, W_out=None,x_init=None,
                 flag_train=False)
            else:
                esn = ESN(num_agent=num_agent, num_environment_input=num_environment_input+num_agent, num_environment_output=num_environment_output, num_sample=num_sample,
                 alpha=alpha,  W_x=None, W_in=None, W_out=None,x_init=None,
                 flag_train=False)
            self.layer_list.append(esn)
            self.x_all[:, num_agent*l: num_agent*(l+1)] = esn.x
            

    def update(self,e_in):
        for l in range(self.num_layer):
            if l==0:
                e_in_layer = e_in
            else:
                e_in_layer = torch.cat( (e_in,x), axis=1 )
            x,_=self.layer_list[l].update(e_in_layer)
            self.x_all[:, self.num_agent*l: self.num_agent*(l+1)] = self.layer_list[l].x
        y_next = F.simoid( torch.mm(self.W_out, self.x_all.t()).t() )
        return y_next
            
            
    def process_sequence(self, input_sequence):
        #input_seuqnce:  num_sample by num_environment_input by time step
        T = input_sequence.shape[2]
        output_sequence = torch.zeros( [self.num_sample, self.num_environment_output, T] )
        for t in range(T):
            e_in = input_sequence[:,:,t]
            y = self.update(e_in)
            output_sequence[:,:,t] = y
        return output_sequence
    
    def train():
        pass
        
        
    
class ESN_AE(Module):
    pass


class CreateReservoir():
    pass 
