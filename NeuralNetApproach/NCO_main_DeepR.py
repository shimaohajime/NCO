#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:44:38 2019

@author: hajime
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:05:56 2018

@author: hajime
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
from scipy.stats import norm
from scipy.special import comb
from itertools import count
from itertools import combinations
from itertools import combinations_with_replacement   
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import os
import sys
import networkx as nx
import re
import pickle
import multiprocessing
from sklearn.model_selection import ParameterGrid
import datetime
import pytz
from tensorflow.python.client import timeline

from sklearn.preprocessing import StandardScaler, normalize, PolynomialFeatures


np.set_printoptions(suppress=True)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

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

        
        
class Organization(object):
    def __init__(self, num_environment=10, num_agents=10, num_managers=9, innoise=0.,
                     outnoise=0., fanout=1,  envnoise=0., envobsnoise=0.,#statedim,
                     batchsize=100, optimizer='Adam',env_input=None,env_pattern_input=None,
                     agent_type = "sigmoid",agent_order='linear',
                     network_prespecified_input=None,network_update_method=None,
                     dropout_rate = 0.0,dropout_type='AllIn',L1_norm=.0,
                     weight_on_cost=0.,weight_update=False,initializer_type='zeros' ,dunbar_number=2 ,dunbar_function='linear_kth' ,
                     flag_DeepR=False, DeepR_T=0.,
                     randomSeed=False, decay=None, start_learning_rate=.01,tensorboard_filename=None, **kwargs):

        self.sess = tf.Session()
        #self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

        #Number of environment nodes
        self.num_environment = num_environment
        #Number of agent nodes
        self.num_agents = num_agents
        #Number of managers (agents who do not take actions)
        if num_managers is "AllButOne":
            self.num_managers = num_agents-1
        else:
            self.num_managers = num_managers
        #Number of actors (agents who take actions and decide the welfare)
        self.num_actors = self.num_agents - self.num_managers
            
        #Polinomial unit. 
        self.agent_order = agent_order
            
        #Batchsize
        self.batchsize = batchsize
        
        #error added between environment node to agents
        self.envobsnoise = envobsnoise
        
        #Prespecify the environment as (env_input, env_pattern_input), otherwise randomly generate here.
        self.env_input = env_input #The environment nodes to be fed toself.environment
        self.env_pattern_input = env_pattern_input # Environment outpt to be predicted, to be fed to the placeholder "env_pattern"
        if env_pattern_input is None: #If no environment specified, generate randomly.
            self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float32)
            zero = tf.convert_to_tensor(0.0, tf.float32)
            greater = tf.greater(self.environment, zero, name="Organization_greater")
            self.environment = tf.where(greater, tf.ones_like(self.environment), tf.zeros_like(self.environment), name="where_env")
        else:
            self.environment = tf.placeholder(tf.float32,shape=[self.batchsize,self.num_environment],name='environment')
            self.env_pattern= tf.placeholder(tf.float32,shape=[self.batchsize,1],name='env_pattern')
            
            
            
        #network_prespecified specifies the network topology
        #num_environment+num_agents times num_agents matrix of binary
        #Includes environment, but not bias
        #network: (num_environment+num_managers, num_agent) matrix representing (i,j)=1 means i->j path exists.
        self.network_prespecified = tf.placeholder(tf.float32, shape=[self.num_environment+self.num_managers,self.num_agents],name='network_prespecified')
        if network_prespecified_input is None: 
            temp = np.zeros([self.num_environment+self.num_managers,self.num_agents])
            temp[np.triu_indices_from(temp,k=-self.num_environment)] = 1.
            self.network_prespecified_input = temp
            #self.network_prespecified_input = np.ones([self.num_environment+self.num_agents,self.num_agents])
        else:            
            self.network_prespecified_input = network_prespecified_input
                
        #Pruning
        self.network_update_method = network_update_method

        #Dropout
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        
        #Coefficient on L1 norm
        self.L1_norm = L1_norm
        #Deep rewiring
        self.flag_DeepR = flag_DeepR
        self.DeepR_T = DeepR_T #Temperature 
        
        
        #Update the balance between the task accuracy and the penalty on the Dunbar constraint
        if weight_update is False:
            self.weight_on_cost = tf.constant(weight_on_cost,dtype=tf.float32) #the weight on the listening cost in loss function
            self.weight_on_cost_val = weight_on_cost
        elif weight_update is True:
            self.weight_on_cost = tf.get_variable(name="weight_on_cost",dtype=tf.float32,initializer=tf.constant(weight_on_cost,dtype=tf.float32),trainable=False  )
            self.weight_on_cost_val = weight_on_cost
            self.assign_weight = tf.assign(self.weight_on_cost, self.weight_on_cost_val)
        self.weight_update = weight_update
        
        #Dunbar number parameter
        self.dunbar_number = dunbar_number 
        self.dunbar_function = dunbar_function
        
        self.initializer_type = initializer_type
        
        self._build_org()
        
        
        #Penalty functions are not moved to new version yet.
        self.objective_task = self._make_loss_task()
        #self.objective_cost = self._make_loss_cost()
        self.objective_L1 = self._make_loss_L1()
        #self.objective = self.weight_on_cost * self.objective_cost + (1-self.weight_on_cost) * self.objective_task + self.objective_L1
        self.objective = self.objective_task +self.objective_L1*self.L1_norm 

        #Parameters for optimization
        self.learning_rate = tf.placeholder(tf.float32)
        #self.optimize =tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective)
        if optimizer is 'Adam':
            self.optimize =tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective)
        if optimizer is 'GradientDescent':
            self.optimize =tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.objective)
            
        self.start_learning_rate = start_learning_rate
        self.decay = decay #None #.01
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
                
    def _build_org(self):
        self._build_agent_params()
        self._build_wave()
        
    def _build_agent_params(self):
        #Array version.
        # out_weights is now (env+manager+1, manager) big matrix that stack the weights of all the agents
        # action weights is (env+manager+1, actor) big matrix
        print('--Build agent parameters--')
        if self.agent_order is 'linear':
            self.out_indim_max = self.num_environment+self.num_managers #no bias
            self.action_indim_max = self.num_environment+self.num_managers #no bias
            
            
        ###01/29 This part is not fixed yet for removing bias from weight matrix.
        elif type(self.agent_order) is int:
            pass
            self.out_indim_max =  1#01/29 should be 0 for no bias
            self.action_indim_max = 1#01/29 should be 0 for no bias
            for i in range(1,self.agent_order+1):
                self.out_indim_max =  self.out_indim_max + comb(self.num_environment+self.num_managers,i,repetition=True)
                self.action_indim_max = self.action_indim_max + comb( self.num_environment+self.num_managers, i, repetition=True)
            
            self.out_indim_max=int(self.out_indim_max)
            self.action_indim_max=int(self.action_indim_max)
            #self.out_indim_max =  int( np.sum( comb( self.num_environment+self.num_managers, np.arange(1,self.agent_order+1)  ) ) +1 )
            #self.action_indim_max = int(  np.sum( comb( self.num_environment+self.num_managers, np.arange(1,self.agent_order+1)  ) ) +1 )
            
            self.out_indim_list = []
            self.action_indim_list =[]
            for i in range(self.num_managers):
                dim = 1
                for j in range(1,self.agent_order+1):
                    dim =dim+comb(self.num_environment+i,j,repetition=True)
                dim = int(dim)
                #dim = int( np.sum( comb( self.num_environment+i, np.arange(1,self.agent_order+1)  ) ) +1 )
                self.out_indim_list.append(dim)
            for i in range(self.num_actors):
                dim = 1
                for j in range(1,self.agent_order+1):
                    dim = dim+comb(self.num_environment+self.num_managers, j, repetition=True)
                dim = int(dim)
                #dim = int(  np.sum( comb( self.num_environment+self.num_managers, np.arange(1,self.agent_order+1)  ) ) +1 )
                self.action_indim_list.append(dim)
        ###01/29 This part is not fixed yet for removing bias from weight matrix.
            
                
                
        #Initialize values
        if self.initializer_type is "zeros":
            init_out_weights = tf.constant(0.0,shape=[self.out_indim_max,self.num_managers],dtype=tf.float32)
            init_action_weights = tf.constant(0.0,shape=[self.action_indim_max,self.num_actors],dtype=tf.float32)
            #01/29 seprated bias
            init_out_bias = tf.constant(0.0,shape=[1,self.num_managers],dtype=tf.float32)
            init_action_bias = tf.constant(0.0,shape=[1,self.num_actors],dtype=tf.float32)
            
        if self.initializer_type is "normal":
            print('self.out_indim_max')
            print(self.out_indim_max)
            print('self.num_managers')
            print(self.num_managers)
            init_out_np = np.random.randn(self.out_indim_max,self.num_managers)
            init_out_weights = tf.constant(value=init_out_np,dtype=tf.float32)
            init_action_weights = tf.constant( np.random.randn(self.action_indim_max,self.num_actors), dtype=tf.float32 )            

            #01/29 seprated bias
            init_out_bias = tf.constant(np.random.randn(self.num_managers),dtype=tf.float32)
            init_action_bias = tf.constant(np.random.randn(self.num_actors),dtype=tf.float32)


        if self.initializer_type is "xavier":
            pass
        if self.flag_DeepR:
            init_out_weights = tf.maximum(init_out_weights, 0.)
            init_action_weights = tf.maximum(init_action_weights, 0.)
            
        self.out_weights = tf.get_variable(dtype=tf.float32,  name='out_weights', initializer=init_out_weights)
        self.action_weights = tf.get_variable(dtype=tf.float32, name='action_weights', initializer=init_action_weights)
        self.out_bias = tf.get_variable(dtype=tf.float32,  name='out_bias', initializer=init_out_bias)
        self.action_bias = tf.get_variable(dtype=tf.float32, name='action_bias', initializer=init_action_bias)
        
        self.action_state_list=[]
        
        print('--Finished agent parameters--')
        
    def _make_loss_L1(self):
        L1_cost_out = tf.reduce_sum(tf.abs(self.out_weights)) 
        L1_cost_action = tf.reduce_sum(tf.abs(self.action_weights)) 
        return (L1_cost_out+L1_cost_action)
    def _make_loss_task(self):
        cross_entropy_list = []
        for i in range(self.num_actors):
            a_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.env_pattern,logits=self.action_state_list[i])
            cross_entropy_list.append(a_loss)
        cross_entropy_mean = tf.reduce_mean(cross_entropy_list)
        return cross_entropy_mean

    ####NEED CHECK###
    #01/29 this part is not fixed for separating bias
    def high_order_indata(self,indata, agent_i, agent_order):
        indata_to_extend = indata[:,:self.num_environment+agent_i]#No bias, input for agent i
        dim = self.num_environment+agent_i
        #indata_out = tf.ones(shape=[self.batchsize,1],dtype=tf.float32)
        indata_out = indata[:,0:1+self.num_environment+agent_i]
        for p in range(2,agent_order+1):
            for j in combinations_with_replacement(np.arange(dim),p):
                #Need to take care of j being (0,) case.
                #data = indata_to_extend[:,np.array(j,dtype=int)]
                #new = tf.reshape( tf.reduce_prod(indata_to_extend[:,j],axis=1), [-1,1]) #Error
                new = tf.ones(shape=[self.batchsize])
                for k in j:
                    new = new * indata_to_extend[:,k]
                new = tf.reshape(new,[-1,1])
                indata_out = tf.concat([indata_out,new],axis=1)
        return indata_out
    #01/29 this part is not fixed for separating bias
    
    
    def _build_wave(self):        
        #For-loop version
        #environment:shape=[self.batchsize,self.num_environment]        
        #indata= tf.concat([tf.ones(shape=[self.batchsize,1],dtype=tf.float32), tf.concat([self.environment, tf.zeros(shape=[self.batchsize, self.num_managers],dtype=tf.float32)],axis=1  )],axis=1)
        #indata: shape = [batchsize, 1+num_environment+num_manager]
        #shape_scatter_manager = tf.constant([self.batchsize,self.num_environment+self.num_managers+1])
        
        #01/29 separated bias
        indata= tf.concat([self.environment, tf.zeros(shape=[self.batchsize, self.num_managers],dtype=tf.float32)],axis=1  )
        #now indata: shape = [batchsize, num_environment+num_manager]
        shape_scatter_manager = tf.constant([self.batchsize,self.num_environment+self.num_managers])
        for i in range(self.num_managers):
            if self.agent_order is 'linear':
                #Added network. check.
                #indata_i = tf.concat([tf.ones(shape=[self.batchsize,1],dtype=tf.float32),   indata[:,1:] * tf.reshape(self.network_prespecified[:,i],[1,-1])  ] ,axis=1)
                #output = tf.sigmoid(tf.matmul(indata, tf.reshape(self.out_weights[:,i],shape=[-1,1]) ) )   
                #01/29 separated bias out
                indata_i =  indata * tf.reshape(self.network_prespecified[:,i],[1,-1])  
                output = self.out_bias[i] + tf.sigmoid( tf.matmul(indata_i, tf.reshape(self.out_weights[:,i],shape=[-1,1]) ) )   



            ####NEED CHECK####
            #01/29 this part is not changed to separate bias
            elif type(self.agent_order) is int:
                indata_higher_order = self.high_order_indata(indata,i,self.agent_order)
                w_i = tf.reshape(self.out_weights[:self.out_indim_list[i],i],shape=[-1,1])
                output = tf.sigmoid(tf.matmul(indata_higher_order, w_i ) )    
            #01/29 this part is not changed to separate bias



            
            #indices = tf.concat([tf.reshape(tf.range(self.batchsize),shape=[-1,1]),tf.constant(1+self.num_environment+i,shape=[self.batchsize,1],dtype=tf.int32)], axis=1)
            indices = tf.concat([tf.reshape(tf.range(self.batchsize),shape=[-1,1]),tf.constant(self.num_environment+i,shape=[self.batchsize,1],dtype=tf.int32)], axis=1)
            updates = tf.reshape(output, shape=[-1] )#tf.reshape(output, shape=[-1,1] )
            scatter = tf.scatter_nd(indices, updates, shape_scatter_manager)
            indata = indata + scatter
        acion_state_list = []
        prediction_list = []
        for i in range(self.num_actors):
            if self.agent_order is 'linear':
                #01/29 separated bias out
                #indata_i = tf.concat([tf.ones(shape=[self.batchsize,1],dtype=tf.float32),   indata[:,1:] * tf.reshape(self.network_prespecified[:,self.num_managers+i],[1,-1])  ] ,axis=1)
                #action_state = tf.matmul(indata_i, tf.reshape( self.action_weights[:,i]  ,shape=[-1,1])    )
                indata_i =    indata * tf.reshape(self.network_prespecified[:,self.num_managers+i],[1,-1]) 
                action_state = self.action_bias[i] + tf.matmul(indata_i, tf.reshape( self.action_weights[:,i]  ,shape=[-1,1])    )

            ####NEED CHECK####
            #01/29 this part is not changed to separate bias
            elif type(self.agent_order) is int:
                indata_higher_order = self.high_order_indata(indata,self.num_managers+i,self.agent_order)
                action_state = tf.sigmoid(tf.matmul(indata_higher_order, tf.reshape(self.out_weights[:,i],shape=[-1,1]) ) )   
        
            prediction = tf.greater(tf.sigmoid(action_state),.5)
            acion_state_list.append(action_state)
            prediction_list.append(prediction)
                
        self.action_state_list = acion_state_list
        self.prediction_list = prediction_list


    def network_rewire(self,network_old, out_params, action_params):
        #1. dormant negative valued link
        out_params_new = np.copy(out_params)
        action_params_new = np.copy(action_params)
        if not network_old.shape[0] == out_params.shape[0]:
            out_params = out_params[1:,:] #need to remove bias
            out_params_new[1:,:] = out_params*(out_params>0)
        else:
            out_params_new = out_params*(out_params>0)
        if not network_old.shape[0] == action_params.shape[0]:
            action_params = action_params[1:,:] #need to remove bias
            action_params_new[1:,:] = action_params*(action_params>0)
            
        network_new = np.copy(network_old)

        network_new[:,:self.num_managers] = network_old[:,:self.num_managers]*(out_params>0)
        network_new[:,self.num_managers:] = network_old[:,self.num_managers:]*(action_params>0)
        
        #2. Activate random dormant links
        fan_in_max_i = self.num_environment
        fan_in_min_i=0
        for i in range(self.num_agents):
            if i == self.num_agents-1:
                fan_in_min_i=self.num_environment
            else:
                fan_in_min_i=0    
            flow_in_i = network_new[:,i]
            fan_in_i = np.sum(np.abs(flow_in_i) )
            if fan_in_i<self.dunbar_number:
                n_reactivate = int(self.dunbar_number-fan_in_i)
                pos_inactive = np.where(flow_in_i==0)
                pos_reactivate = np.random.choice(pos_inactive[0][(pos_inactive[0]<fan_in_max_i)*(pos_inactive[0]>=fan_in_min_i)],[n_reactivate],replace=False)
                flow_in_i[pos_reactivate]=np.random.choice( [1,-1],len(pos_reactivate) )
                network_new[:,i] = flow_in_i
            fan_in_max_i = fan_in_max_i+1
        return network_new
                
                
            

    def train(self, niters, lrinit=None, iplot=False, verbose=False):
        if( lrinit == None ):
            lrinit = self.start_learning_rate

        training_res_seq = []
        task_loss_seq = []
        weight_on_cost_seq = []
        lr_seq = []
        network_seq = []
        
        out_w_seq = []
        action_w_seq = []
        
        self.action_state_seq =[]
        self.prediction_seq = []
        
        network_update_n = 0
        lr_decay = 0.
        lr = float(lrinit)
        
        for i  in range(niters):
            lr = float(lrinit) / float(1. + lr_decay) # Learn less over time
            #_,u0,u_t0,u_c0,w = self.sess.run([self.optimize,self.objective,self.objective_task,self.objective_cost,self.weight_on_cost], feed_dict={self.learning_rate:lr,self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input})
            
            '''
            #Profiling TF
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _,u0,u_t0 = self.sess.run([self.optimize,self.objective,self.objective_task], feed_dict={self.learning_rate:lr,self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input},options=options,run_metadata=run_metadata)
            '''
            
            _,u0,u_t0,out_w,action_w,action_state,prediction = self.sess.run([self.optimize,self.objective,self.objective_task,self.out_weights,self.action_weights,self.action_state_list,self.prediction_list], feed_dict={self.learning_rate:lr,self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input})

                
            #Record error rate and network 
            if i%10==0:
                '''
                #Profiling TF
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('new_timeline_step_%d.json' % i, 'w') as f:
                    f.write(chrome_trace)
                '''
                #weight_on_cost_seq.append(w)
                training_res_seq.append(u0)
                task_loss_seq.append(u_t0)
                
                out_w_seq.append(out_w)
                action_w_seq.append(action_w)

                lr_seq.append(lr)
                network_seq.append(self.network_prespecified_input)
                
                self.action_state_seq.append(action_state)
                self.prediction_seq.append(prediction)
                


            #Update network by pruning
            num_pruning = (self.num_agents+self.num_environment) - self.dunbar_number
            pruning_dur = int( niters/(num_pruning+3) )

            if i%pruning_dur==0 and i>0 and network_update_n<num_pruning:
                #Update network
                if self.network_update_method is not None:
                    prune_target = int(self.num_agents-network_update_n-1)
                    if prune_target <= 0:
                        prune_target = 'All'

                    network_new = self.network_update(self.network_prespecified_input, self.network_update_method, out_w,action_w)

                    self.network_prespecified_input = np.copy(network_new)
                network_update_n = network_update_n+1

            #Update network by DeepR
            #DeepR(3)
            if self.flag_DeepR:
                #eta->lr (learning rate)
                #alpha->L1_norm (regularizer term)
                #T is a parameter
                noise_out_weights = tf.random_normal(shape=(out_w.shape))* np.sqrt(2*lr*self.DeepR_T)
                noise_action_weights = tf.random_normal(shape=(action_w.shape))* np.sqrt(2*lr*self.DeepR_T)
                self.out_weights = self.out_weights + noise_out_weights
                self.action_weights = self.action_weights + noise_action_weights
                network_new = self.network_rewire(self.network_prespecified_input, out_w,action_w)
                self.out_weights = tf.maximum(self.out_weights,0.) * np.abs(network_new[:,:self.num_managers])
                self.action_weights = tf.maximum(self.action_weights,0.) * np.abs(network_new[:,self.num_managers:])
                self.network_prespecified_input = np.copy(network_new)



            #Learning Rate Update
            if self.decay != None:
                lr_decay = lr_decay + self.decay

                

            if verbose and (i%10==0):
                print('----')
                print  ("iter"+str(i)+": Loss function=" + str(u0) )
                #print("task loss:"+str(u_t0)+',cost loss:'+str(u_c0) )
                print("task loss:"+str(u_t0) )
                if self.weight_update is True:
                    print("weight on cost:"+str(w))
                    print('learning rate:'+str(lr))
                print('The correct answer ratio:'+str( np.mean(prediction[0]==self.env_pattern_input) ))
                print('----')
                print('current network')
                print(self.network_prespecified_input)
                print(out_w)
                print(action_w)

                
            self.training_res_seq=training_res_seq
            self.task_loss_seq=task_loss_seq
            self.action_w_seq = action_w_seq
            self.out_w_seq = out_w_seq
            self.network_seq = network_seq

            if u_t0<.01:
                break


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
        
        
    
if __name__=="__main__":
    
    #------------------
    Description = 'Test_DeepR'
    #------------------
        
    print ('Current working directory : ', os.getcwd() )

    start_time = time.time()


    parameters_for_grid = [
        {"num_environment" : [6], # Num univariate environment nodes
        "num_agents" : [10], # Number of Agents
        "num_managers" : ["AllButOne"], # Number of Agents that do not contribute
        "innoise" : [0.], # Stddev on incomming messages #Not included yet
        "outnoise" : [0.], # Stddev on outgoing messages #Not included yet
        "fanout" : [1], # Distinct messages an agent can say, 1 is assumed so far
        "envnoise": [0.], # Stddev of environment state
        "envobsnoise" : [0.], # Stddev on observing environment
        "batchsize" : [1000],#200,#, # Training Batch Size. Can be None if environment is constructed to cover all the possibilities.
        'optimizer':['GradientDescent'],
        'env_input': [None],
        'env_pattern_input':[None],
        'agent_type':["sigmoid"],
        'agent_order':['linear'],
        'network_prespecified_input':[None],
        'network_update_method':[None],#'pruning',None
        'dropout_rate':[.0],
        "dropout_type":[None],#,'AllIn'
        "weight_on_cost":[0.0],
        "weight_update":[False],
        "initializer_type":["normal"],
        "dunbar_number":[4,8],
        "dunbar_function":[None],  #"sigmoid_ratio","sigmoid_kth","L4",'linear_kth'
        'decay':[.0],
        "description" : [Description],
        'flag_DeepR':[True], 
        "L1_norm":[.003,.03,.3], # alpha in DeepR
        'start_learning_rate':[.005,.01,.05,.1], # eta in DeepR
        'DeepR_T':[.001]
        } 
    ]

                     
    parameters_temp = list(ParameterGrid(parameters_for_grid))
    n_param = len(parameters_temp)
    parameters = []

    for i in range(n_param):
        if parameters_temp[i]['dunbar_number']>parameters_temp[i]['num_environment']:
            pass
        else:
            if (parameters_temp[i]['num_managers'] is 'AllButOne'):
                parameters_temp[i]['num_managers']= parameters_temp[i]['num_agents']-1
            
            if (parameters_temp[i]['dropout_type'] is None):
                parameters_temp[i]['dropout_rate']=0.0
            if (parameters_temp[i]['dropout_rate'] == 0.0):
                parameters_temp[i]['dropout_type']=None            
            if (parameters_temp[i]['dunbar_function'] is None):
                parameters_temp[i]['weight_update'] = False
                parameters_temp[i]['weight_on_cost'] = 0.0
            if (parameters_temp[i]['weight_update'] is False) and (parameters_temp[i]['weight_on_cost'] == 0.0):
                parameters_temp[i]['dunbar_function'] = None
                
            if (parameters_temp[i]['flag_DeepR'] is True):
                parameters_temp[i]['network_prespecified_input'] = gen_constrained_network(parameters_temp[i]['num_environment'],parameters_temp[i]['num_managers'],parameters_temp[i]['num_agents'],parameters_temp[i]['dunbar_number'])
                
                
            dup = False
            for p in parameters:
                if parameters_temp[i] is p:
                    dup=True
            if dup is False:
                parameters.append(parameters_temp[i])

    iteration_train = 2000
    iteration_restart = 1

    exec_date = datetime.datetime.now(pytz.timezone('US/Mountain')).strftime('%B%d_%H%M')

    dirname ='./result_'+exec_date +'_' + Description

    createFolder(dirname)

    dirname_abs = os.getcwd() + '/result_'+exec_date +'_' + Description

    for i in range(n_param):
        
        start_time_setting = time.time()

        filename = '/Setting'+str(i)
        j = 0
        filename_trial = filename + '_trial' + str(j)


   
        parameter = parameters[i]
        batchsize = parameter['batchsize']
        num_environment = parameter['num_environment']

        #---Create environment---------------------
        #env_class = Environment(batchsize,num_environment,env_type='match_mod2')
        #env_class = Environment(batchsize,num_environment,num_agents=parameter['num_agents'],env_type='match_mod2',input_type='all_comb')
        env_class = Environment(batchsize,num_environment,num_agents=parameter['num_agents'],env_type='match_mod2',input_type='all_comb',flag_normalize=False)
        #gen_from_network, match_mod2, all_comb, random
        env_class.create_env()
        env_input = env_class.environment
        env_pattern_input = env_class.env_pattern
        
        
        parameter['batchsize'] = env_class.batchsize
        parameter['env_input'] = env_input
        parameter['env_pattern_input'] = env_pattern_input
        #------------------------------------------

        tf.reset_default_graph()
        print('********************'+'Setting'+str(i)+'********************')
        filename = '/Setting'+str(i)

        orgA = Organization(tensorboard_filename='board_log_'+filename,**parameter)
        orgA.train(iteration_train, iplot=False, verbose=False)


        #--------------
        pickle.dump(parameter, open(dirname+filename_trial+"_parameters.pickle","wb"))
        pickle.dump(orgA.training_res_seq, open(dirname+filename_trial+"_training_res_seq.pickle","wb"))
        pickle.dump(orgA.task_loss_seq, open(dirname+filename_trial+"_task_loss_seq.pickle","wb"))

        #pickle.dump(orgA.weight_on_cost_seq, open(dirname+filename_trial+"_weight_on_cost_seq.pickle","wb"))
        #pickle.dump(orgA.lr_seq, open(dirname+filename_trial+"_lr_seq.pickle","wb"))

        pickle.dump(orgA.out_w_seq, open(dirname+filename_trial+"_out_w_seq.pickle","wb"))
        pickle.dump(orgA.action_w_seq, open(dirname+filename_trial+"_action_w_seq.pickle","wb"))
        pickle.dump(orgA.network_seq, open(dirname+filename_trial+"_network_seq.pickle","wb"))

        pickle.dump(orgA.prediction_seq, open(dirname+filename_trial+"_prediction_seq.pickle","wb"))
        pickle.dump(orgA.action_state_seq, open(dirname+filename_trial+"_action_state_seq.pickle","wb"))


        end_time_setting = time.time()
        time_elapsed_setting = end_time_setting-start_time_setting
        print('time for setting%i:%f '%(i,time_elapsed_setting) )

        
    end_time = time.time()
    time_elapsed = end_time-start_time
    print('time: ',time_elapsed)
