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
from scipy.special import expit
import networkx as nx
from matplotlib import pyplot as plt


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#Create initial network that satisfies the constraint (for DeepR)
def gen_constrained_network(num_environment,num_managers,num_agents,dunbar_number,type_network='random',width_seq=None):
    init_network = np.zeros([num_environment+num_managers, num_agents])
    if type_network is 'random':
        n_in_i = num_environment #number of possible input
        for i in range (num_agents-1):
            flow_in = np.random.choice(np.arange(n_in_i),dunbar_number, replace=False)
            init_network[flow_in, i] = 1
            if n_in_i<=num_environment+num_managers:
                n_in_i = n_in_i+1
        flow_in = np.random.choice(np.arange(num_environment,n_in_i),dunbar_number, replace=False)
        init_network[flow_in, i+1] = np.random.choice([1,-1], len(flow_in))
        
    if type_network is 'layered_with_width':
        
        in_lb=0
        in_ub=num_environment
        idx = 0
        for i in range(len(width_seq)):        
            possible_in = np.arange(in_lb,in_ub)
            for j in range(width_seq[i]):    
                pos_active =   np.random.choice(possible_in, np.min( (dunbar_number, len(possible_in)) ) , replace=False)
                init_network[pos_active, idx+j  ] = 1
            in_lb = np.copy(in_ub)
            in_ub = in_ub+width_seq[i]
            idx = idx+width_seq[i]
        for j in range(idx, num_agents):
            possible_in = np.arange(in_lb,in_ub)
            pos_active =   np.random.choice(possible_in, np.min( (dunbar_number, len(possible_in)) ) , replace=False)
            init_network[pos_active, j  ] = 1
            
        
            
    return init_network


def gen_full_network(num_environment,num_manager,num_agent):
    temp = np.zeros([num_environment+num_manager,num_agent])
    temp[np.triu_indices_from(temp,k=-num_environment+1)] = 1.
    init_network = temp
    return init_network


class Environment():
    def __init__(self,batchsize=None,num_environment=6,env_network=None,num_agents=None,num_manager=None,num_actor=None,env_weight=None,env_bias=None,env_type='match_mod2',input_type='all_comb',flag_normalize=False):
        self.batchsize = batchsize
        self.num_environment = num_environment
        self.env_type = env_type
        self.input_type = input_type
        self.flag_normalize = flag_normalize
        if env_type == 'gen_from_network':
            if env_network is None:
                self.num_agents = num_agents
                self.num_manager = num_manager
                self.num_actor = num_actor
                self.env_network = np.random.randint(2,size=[self.num_environment+self.num_manager, self.num_agents])
                '''
                #This part assumes network matrix to be (num_agent+num_environment, num_agent) for TF version
                #PyTorch version assumes network to be (num_manager+num_environment-1, num_agent), excluding loops within an agent.
                self.num_agents = num_agents
                self.env_network = np.random.randint(2,size=[self.num_environment+self.num_agents, self.num_agents])
                '''
            else:
                self.env_network = env_network # (num_env+num_agents,num_agents) matrix
                self.num_agents = self.env_network.shape[1]
                self.num_manager = num_manager
                self.num_actor = num_actor
                
            if env_weight is None:
                
                #(TF VERSION)self.env_weight = np.random.randn(1+self.num_environment+self.num_agents,self.num_agents)
                #(PyTorch version)
                self.env_weight = np.random.randn(self.num_environment+self.num_manager,self.num_agents)
                self.env_bias = np.random.randn(self.num_agents)
                
            else:
                self.env_weight = env_weight 
                self.env_bias= env_bias

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
        '''
        #This part assumes network matrix to be (num_agent+num_environment, num_agent) for TF version
        #PyTorch version assumes network to be (num_agent+num_environment-1, num_agent), excluding loops within an agent.
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
        '''
        #This is for PyTorch, network matrix is (num_agent+num_environment-1, num_agent).
        if self.env_type is 'gen_from_network':

            ones = np.ones([self.batchsize,1])
            zeros = np.zeros([self.batchsize,self.num_manager])
            indata = np.concatenate((self.environment,zeros),axis=1)
                        
            
            
            for i in range(self.num_manager):
                #network = np.concatenate( ([1.], self.env_network[:,i]))
                network = self.env_network[:,i]
                w = (self.env_weight[:,i] * network).reshape([-1,1])
                m = expit( np.dot(indata,w) +self.env_bias[i] )
                indata[:,self.num_environment+i] = np.copy(m.flatten())
                #indata[:,1+self.num_environment+i] = np.copy(m.flatten())
            network = self.env_network[:,-1]
            w = (self.env_weight[:,-1] * network).reshape([-1,1])
            
            self.env_bias[-1] = - np.mean(np.dot(indata,w))            
            
            a = expit(np.dot(indata,w) +self.env_bias[-1] )
            self.env_pattern = (a>0.5).astype(int)
            self.message = indata
            
        if self.flag_normalize:
            self.environment = normalize(self.environment)
            #self.env_pattern = normalize(self.env_pattern)


def draw_network(num_environment=6, num_manager=9, num_actor=1,num_agent=10, network=None, filename=None):
    pass
    position={}
    color = []        
    G = nx.DiGraph()        
    for i in range(num_environment):
        #G.add_node(i, node_color="b", name="E" + str(i))
        G.add_node(i, node_color="g", name="E")

    for aix in range(num_agent):
        nodenum = num_environment +aix
        if aix<num_manager:
            G.add_node(nodenum, node_color='b', name = "M" + str(aix))
        if aix>=num_manager:
            G.add_node(nodenum, node_color='r', name = "A" + str(aix))
        for eix, val in enumerate(network[:,aix]):
            if abs(val)>.0001:
                G.add_edge(eix, nodenum, width=val)

    hpos = np.zeros(num_environment+num_agent)
    for i in range(num_environment,num_environment+num_manager):
        
        path_to_actors = []
        for j in range(num_environment+num_manager,num_environment+num_agent):
            try:
                path_to_actors.append(nx.shortest_path_length(G,i,j))
            except:
                pass
        try:
            hpos[i] =  np.max(path_to_actors)+1
        except:
            hpos[i]=1#0
    
    hpos[:num_environment] = np.max(hpos) + 2#1
    
    hpos = np.zeros(num_environment+num_agent)
    for i in range(num_environment,num_environment+num_manager):
        
        path_to_actors = []
        for j in range(num_environment+num_manager,num_environment+num_agent):
            try:
                path_to_actors.append(nx.shortest_path_length(G,i,j))
            except:
                pass
        try:
            hpos[i] =  np.max(path_to_actors)+1
        except:
            hpos[i]=1#0
    
    hpos[:num_environment] = np.max(hpos) + 2#1
    
    
    vpos = np.zeros(num_environment+num_agent)
    
    for i in range(num_environment+num_agent):
        for j in range(np.max(hpos).astype(int)+1):
            vpos[np.where(hpos==j)] = np.arange( len( np.where(hpos==j)[0] ) )
            if np.mod(j,2) == 1.:
                vpos[np.where(hpos==j)] = vpos[np.where(hpos==j)] +.5
    
    
    color_list = []
    label_list = []
    for i in range(num_environment+num_agent):
        G.node[i]['pos'] = (hpos[i],vpos[i])
        if i<num_environment:
            color_list.append('g')
            label_list.append('E')
        elif i<num_environment+num_manager:
            color_list.append('b')
            label_list.append('M')
        else:
            color_list.append('r')
            label_list.append('A')
        
    pos=nx.get_node_attributes(G,'pos')
    
    
    fig = plt.figure()
    nx.draw(G,pos=pos,node_color=color_list,with_label=True)
    if filename is not None:
        fig.savefig(filename+"_network.png")
    plt.show()
    plt.clf()



