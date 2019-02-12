#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:10:49 2019

@author: hajime
"""
import os
import numpy as np
from itertools import combinations
from sklearn.preprocessing import  normalize
import time

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#Create initial network that satisfies the constraint (for DeepR)
def gen_constrained_network(num_environment,num_managers,num_agents,dunbar_number):
    init_network = np.zeros([num_environment+num_managers, num_agents])
    
    n_in_i = num_environment #number of possible input
    for i in range (num_agents-1):
        flow_in = np.random.choice(np.arange(n_in_i),dunbar_number, replace=False)
        init_network[flow_in, i] = 1
        if n_in_i<=num_environment+num_managers:
            n_in_i = n_in_i+1
    flow_in = np.random.choice(np.arange(num_environment,n_in_i),dunbar_number, replace=False)
    init_network[flow_in, i+1] = np.random.choice([1,-1], len(flow_in))
    
            
    return init_network


def gen_full_network(num_environment,num_manager,num_agent):
    temp = np.zeros([num_environment+num_manager,num_agent])
    temp[np.triu_indices_from(temp,k=-num_environment+1)] = 1.
    init_network = temp
    return init_network


class Environment():
    def __init__(self,batchsize,num_environment,env_network=None,num_agents=None,env_weight=None,env_type='match_mod2',input_type='all_comb',flag_normalize=False):
        self.batchsize = batchsize
        self.num_environment = num_environment
        self.env_type = env_type
        self.input_type = input_type
        self.flag_normalize = flag_normalize
        if env_type == 'gen_from_network':
            if env_network is None:
                self.num_agents = num_agents
                self.env_network = np.random.randint(2,size=[self.num_environment+self.num_agents, self.num_agents])
            else:
                self.env_network = env_network # (num_env+num_agents,num_agents) matrix
                self.num_agents = self.env_network.shape[1]
            if env_weight is None:
                self.env_weight = np.random.randn(1+self.num_environment+self.num_agents,self.num_agents)
            else:
                self.env_weight=env_weight # (1+num_env+num_agents,num_agents) matrix. +1 for bias

    def create_env(self):
        if self.input_type is 'random':
            self.environment = np.random.randint(2,size = [self.batchsize, self.num_environment])
        if self.input_type is 'all_comb':
            self.batchsize = 2**self.num_environment
            self.environment = np.zeros([self.batchsize, self.num_environment])
            for i in range(1,self.num_environment+1):
                self.environment[i,i-1] = 1
            i = self.num_environment+1
            for n1 in range(2,self.num_environment+1):
                for idx in combinations(np.arange(self.num_environment),n1):
                    self.environment[i,idx] = 1
                    i = i+1

        if self.env_type is 'match_mod2':                
            left = self.environment[:,:int(self.num_environment/2)]
            right = self.environment[:,int(self.num_environment/2):]
            lmod = np.mod(np.sum(left,axis=1),2)
            rmod = np.mod(np.sum(right,axis=1),2)
            self.env_pattern = (lmod==rmod).astype(np.float32).reshape([-1,1])            
            
        if self.env_type is 'gen_from_network':

            ones = np.ones([self.batchsize,1])
            zeros = np.zeros([self.batchsize,self.num_agents])
            indata = np.concatenate((ones,self.environment,zeros),axis=1)
                        
            for i in range(self.num_agents):
                network = np.concatenate( ([1.], self.env_network[:,i]))
                w = (self.env_weight[:,i] * network).reshape([-1,1])
                m = np.dot(indata,w)
                indata[:,1+self.num_environment+i] = np.copy(m.flatten())
            self.env_pattern = (m>0).astype(int)
            
        if self.flag_normalize:
            self.environment = normalize(self.environment)
            #self.env_pattern = normalize(self.env_pattern)

