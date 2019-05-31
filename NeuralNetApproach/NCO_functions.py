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


#Create a tree-structured partitioned groups 
def gen_partition_tree(param_dict):
    for key,val in param_dict.items():
        exec(key + '=val')
                
    G = nx.DiGraph()
    for i in range(num_partition_environment):
        G.add_node("env_partition_%i" % i)
    #1st layer
    env_p = np.arange(num_partition_environment)
    env_p_p = np.array_split(env_p, num_partition_agent[0])
    for i in range(num_partition_agent[0]):
        G.add_node("agent_partition_%i" % i)
        env_covered_by_i = env_p_p[i]
        for j in env_covered_by_i:
            G.add_edge("env_partition_%i" % j, "agent_partition_%i" % i  )
    
    sum_partition_agent_prev_prev = 0
    sum_partition_agent_prev = num_partition_agent[0]
    sum_partition_agent_current = num_partition_agent[0] + num_partition_agent[1]
    
    
    for k in range(1,num_layer):
        prev_p = np.arange(sum_partition_agent_prev_prev, sum_partition_agent_prev)
        prev_p_p = np.array_split(prev_p, num_partition_agent[k])
        for i in range(sum_partition_agent_prev, sum_partition_agent_current):
            G.add_node("agent_partition_%i" % i)
            prev_p_covered_by_i = prev_p_p[i-sum_partition_agent_prev]
            for j in prev_p_covered_by_i:
                G.add_edge("agent_partition_%i" % j, "agent_partition_%i" % i  )
        if k<num_layer-1:
            sum_partition_agent_prev_prev = np.copy(sum_partition_agent_prev)
            sum_partition_agent_prev = np.copy(sum_partition_agent_current)
            sum_partition_agent_current = sum_partition_agent_prev + num_partition_agent[k]
            
    #Last layer
    G.add_node('actor')
    for i in range(sum_partition_agent_prev, sum_partition_agent_current):
        G.add_edge( "agent_partition_%i"%i, "actor" )
    return nx.to_numpy_matrix(G)



#Create initial network that satisfies the constraint (for DeepR)
def gen_constrained_network(num_environment,num_managers,num_agents,dunbar_number,type_network='random',width_seq=None):
    init_network = np.zeros([num_environment+num_managers, num_agents])
    if type_network is 'random':
        n_in_i = num_environment #number of possible input
        for i in range (num_agents-1):
            flow_in = np.random.choice(np.arange(n_in_i), np.min( (n_in_i,dunbar_number) ), replace=False)
            init_network[flow_in, i] = 1
            if n_in_i<=num_environment+num_managers:
                n_in_i = n_in_i+1
        flow_in = np.random.choice(np.arange(num_environment,n_in_i), np.min( (n_in_i-num_environment , dunbar_number) ), replace=False)
        init_network[flow_in, i+1] = np.random.choice([1,-1], len(flow_in))

    if type_network in ['layered_random','layered_full'] :

        in_lb=0
        in_ub=num_environment
        idx = 0
        for i in range(len(width_seq)):
            possible_in = np.arange(in_lb,in_ub)
            for j in range(width_seq[i]):
                if type_network is 'layered_random':
                    pos_active =   np.random.choice(possible_in, np.min( (dunbar_number, len(possible_in)) ) , replace=False)
                elif type_network is 'layered_full':
                    pos_active = possible_in

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


def gen_layer_and_partition(num_environment,num_agent,num_manager,
                            num_layer=None,
                            agent_per_layer=None,
                            num_partition_environment=2,
                            num_partition_agent=2):
    environment_partition_id = np.arange(num_partition_environment).repeat( np.ceil(num_environment/num_partition_environment).astype(int)  )
    environment_partition_id = environment_partition_id[:num_environment]

    #Assign layer number to each agent    
    if not hasattr( agent_per_layer, "__len__" ): #Same size for each layer
        manager_layer_id = np.arange(num_layer).repeat( np.ceil(num_manager/num_layer).astype(int) )
        manager_layer_id = manager_layer_id[:num_manager]
    elif hasattr( agent_per_layer, "__len__" ): #agent_per_layer is a sequence to speciy number of nodes in each layer
        manager_layer_id = np.array([])
        for l in range(num_layer):
            manager_layer_id = np.concatenate( (manager_layer_id, np.ones(agent_per_layer[l])*l  ) )
        manager_layer_id=manager_layer_id.astype(int)            
    actor_layer_id = (np.ones(num_agent-num_manager)*(np.max(manager_layer_id)+1) ).astype(int)
    agent_layer_id = np.concatenate(  ( manager_layer_id, actor_layer_id ) )

            
    if not hasattr(num_partition_agent,"__len__"): #num_partition_agent is a scalar value that applies to all the layers.
        manager_partition_id = np.arange(num_layer*num_partition_agent).repeat( np.ceil(num_manager/(num_layer*num_partition_agent) ).astype(int) )
        manager_partition_id = manager_partition_id[:num_manager]
        actor_partition_id = (np.ones( num_agent-num_manager ) * np.max(manager_partition_id) + 1).astype(int)
        agent_partition_id = np.concatenate( (manager_partition_id, actor_partition_id) )
    elif hasattr(num_partition_agent,"__len__"): #num_partition_agent is a sequence of length num_layer.
        manager_partition_id = np.zeros(num_manager)
        partition_id = 0
        for l in range( len(num_partition_agent) ):#For each layer
            manager_in_l = np.where(manager_layer_id==l)[0]
            for p in np.array_split( manager_in_l, num_partition_agent[l] ):
                manager_partition_id[p] = partition_id
                partition_id = partition_id+1
    actor_partition_id = ( np.ones(num_agent-num_manager) * (np.max(manager_partition_id)+1) ).astype(int)
    agent_partition_id = np.concatenate( (manager_partition_id, actor_partition_id) ) 
    
    return environment_partition_id, agent_layer_id, agent_partition_id 
        
        
        
def gen_layered_modular_network(num_environment,num_agent,num_layer,dunbar_number, #Number of input nodes, number of layers other than input layer
                                agent_layer_id=None, #if it's seq, it should be a vecctor of length num_layer.
                                agent_partition_id=None, #If it's seq, it should be a vector of length num_layer.
                                environment_partition_id=None, # partition of environment.
                                partitions_network=None, #Network of partitions. (num_partition_environment+sum(num_partition_layer_seq), )
                                rho=0.05 #Probability that link is formed without partition consideration.
                                ):
    '''
    num_environment=6
    num_agent = 12
    agent_layer_id = np.array([0,0,0,0,1,1,1,1,2,2,2,2]) 
    environment_partition_id = np.array([0,0,0,1,1,1])
    agent_partition_id = np.array([0,0,1,1,2,2,3,3,4,4,5,5] )+2    
    
    partitions_network = np.zeros([2*4, 2*4])
    partitions_network[0,2] = 1
    partitions_network[1,3] = 1
    partitions_network[2,4] = 1
    partitions_network[3,5] = 1
    partitions_network[4,6] = 1
    partitions_network[5,7] = 1
    '''
    
    partition_id = np.concatenate( (environment_partition_id, agent_partition_id+np.max(environment_partition_id)+1 ) )
    layer_id = np.concatenate( (np.zeros(num_environment).astype(int),  agent_layer_id+1) )
    
    nodes_network_extended = np.zeros([num_agent+num_environment, num_agent+num_environment]) 
    
    #Fully connect the connected partitions:
    idx_connected_partitions = np.where( partitions_network==1 )
    for i in range( len(idx_connected_partitions[0]) ):
        partition_from = idx_connected_partitions[0][i]
        partition_to = idx_connected_partitions[1][i]
        node_from = np.where(partition_id==partition_from)[0]
        node_to = np.where(partition_id==partition_to)[0]
        for j in node_from:
            for k in node_to:
                nodes_network_extended[j,k]=1
                
    #Satisfy Dunbar number
    for i in range(nodes_network_extended.shape[1]):
        link_in = np.where(nodes_network_extended[:,i])[0]
        fan_in = len(link_in)
        while fan_in>dunbar_number:
            link_to_rm = np.random.choice(link_in)
            nodes_network_extended[link_to_rm,i] = 0
            link_in = np.where(nodes_network_extended[:,i])[0]
            fan_in = len(link_in)

    #Randomly reconnect link without partition constraint
    for i in range(nodes_network_extended.shape[1]):
        link_in = np.where(nodes_network_extended[:,i])[0]
        fan_in = len(link_in)
        layer_i = layer_id[i]
        prev_layer = layer_i-1
        nodes_in_prev_layer = np.where(layer_id==prev_layer)[0]
        if prev_layer>=0:
            for link in link_in:
                r = np.random.uniform()
                if r<rho:
                    nodes_network_extended[link,i] = 0
                    new_link = np.random.choice( nodes_in_prev_layer )
                    nodes_network_extended[new_link,i] = 1
                    if link!=new_link:
                        print('non-modular link happend')

                    
                
    nodes_network = nodes_network_extended[:-1,num_environment:]
        
    return nodes_network





class Environment():
    def __init__(self,batchsize=None,num_environment=6,env_network=None,num_agents=None,num_manager=None,num_actor=None,
                 env_weight=None,env_bias=None,env_type='match_mod2',env_n_region=None,input_type='all_comb',
                 flag_normalize=False):
        self.batchsize = batchsize
        self.num_environment = num_environment
        self.env_type = env_type
        self.env_n_region = env_n_region
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

        if self.env_type is 'match_mod2_n':
            pattern_region = np.zeros([self.batchsize, self.env_n_region])
            for i in range(self.env_n_region):
                env_i = self.environment[:, int(self.num_environment/self.env_n_region)*i:int(self.num_environment/self.env_n_region)*(i+1)  ]
                pattern_region[:,i] = np.mod( np.sum(env_i,axis=1),2  )
            self.env_pattern = np.mod(  np.sum(pattern_region,axis=1),2    ).reshape([-1,1])



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

    network = np.abs(network)

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
    #plt.show()
    plt.clf()



def check_dunbar(network, dunbar_number):
    dunbar_violate =np.zeros(network.shape[1])
    for i in range(network.shape[1]):
        network_i = np.abs(network[:,i] )
        if np.sum(network_i)>dunbar_number:
            dunbar_violate[i] = 1
    return dunbar_violate

def network_index(network, num_environment,num_agent ):
    pass
    

