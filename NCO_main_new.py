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

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


class Environment():
    def __init__(self,batchsize,num_environment,env_type='match_mod2'):
        self.batchsize = batchsize
        self.num_environment = num_environment
        self.env_type = env_type

    def create_env(self):
        if self.env_type is 'match_mod2':
            self.environment = np.random.randint(2,size = [self.batchsize, self.num_environment])
            left = self.environment[:,:int(self.num_environment/2)]
            right = self.environment[:,int(self.num_environment/2):]
            lmod = np.mod(np.sum(left,axis=1),2)
            rmod = np.mod(np.sum(right,axis=1),2)
            self.env_pattern = (lmod==rmod).astype(np.float32).reshape([-1,1])

class Organization(object):
    def __init__(self, num_environment, num_agents, num_managers, innoise,
                     outnoise, fanout,  envnoise, envobsnoise,#statedim,
                     batchsize, optimizer,env_input,env_pattern_input=None,
                     agent_type = "sigmoid",agent_order='linear',
                     network_type=None,network_prespecified_input=None,network_update_method=None,
                     dropout_rate = 0.0,dropout_type='AllIn',L1_norm=.0,
                     weight_on_cost=0.,weight_update=False,initializer_type='zeros' ,dunbar_number=2 ,dunbar_function='linear_kth' ,
                     randomSeed=False, decay=None, tensorboard_filename=None, **kwargs):

        self.sess = tf.Session()


        self.num_environment = num_environment
        self.num_agents = num_agents
        if num_managers is "AllButOne":
            self.num_managers = num_agents-1
        else:
            self.num_managers = num_managers
        self.num_actors = self.num_agents - self.num_managers
            
        self.agent_order = agent_order
            
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        
        self.env_input = env_input
        self.env_pattern_input = env_pattern_input
        if env_pattern_input is None:
            self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float32)
            zero = tf.convert_to_tensor(0.0, tf.float32)
            greater = tf.greater(self.environment, zero, name="Organization_greater")
            self.environment = tf.where(greater, tf.ones_like(self.environment), tf.zeros_like(self.environment), name="where_env")
        else:
            self.environment = tf.placeholder(tf.float32,shape=[self.batchsize,self.num_environment])
            self.env_pattern= tf.placeholder(tf.float32,shape=[self.batchsize,1])
        #num_environment+num_agents times num_agents matrix of binary
        #Includes environment, but not bias
        self.network_prespecified = tf.placeholder(tf.float32, shape=[self.num_environment+self.num_agents,self.num_agents])
        if network_prespecified_input is None: #all the edges are possible
            temp = np.zeros([self.num_environment+self.num_agents,self.num_agents])
            temp[np.triu_indices_from(temp,k=-self.num_environment)] = 1.
            self.network_prespecified_input = temp
            #self.network_prespecified_input = np.ones([self.num_environment+self.num_agents,self.num_agents])
        else:
            self.network_prespecified_input = network_prespecified_input
                
        self.network_update_method = network_update_method

        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        
        self.L1_norm = L1_norm

        if weight_update is False:
            self.weight_on_cost = tf.constant(weight_on_cost,dtype=tf.float32) #the weight on the listening cost in loss function
            self.weight_on_cost_val = weight_on_cost
        elif weight_update is True:
            self.weight_on_cost = tf.get_variable(name="weight_on_cost",dtype=tf.float32,initializer=tf.constant(weight_on_cost,dtype=tf.float32),trainable=False  )
            self.weight_on_cost_val = weight_on_cost
            self.assign_weight = tf.assign(self.weight_on_cost, self.weight_on_cost_val)
        self.weight_update = weight_update
        self.dunbar_number = dunbar_number #Dunbar number
        self.dunbar_function = dunbar_function
        
        self.initializer_type = initializer_type
        
        self.build_org()
        
        self.objective_task = self._make_loss_task()
        #self.objective_cost = self._make_loss_cost()
        #self.objective_L1 = self._make_loss_L1()
        #self.objective = self.weight_on_cost * self.objective_cost + (1-self.weight_on_cost) * self.objective_task + self.objective_L1
        self.objective = self.objective_task 

        self.learning_rate = tf.placeholder(tf.float32)
        #self.optimize =tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective)
        self.optimize =tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective)
        self.start_learning_rate = .1#15.
        self.decay = decay #None #.01

        init = tf.global_variables_initializer()
        self.sess.run(init)

        

    def build_org(self):
        self.build_agent_params()
        self.build_wave()
        
    def build_agent_params(self):
        #Array version.
        # out_weights is now (env+manager+1, manager) big matrix that stack the weights of all the agents
        # action weights is (env+manager+1, actor) big matrix
        if self.agent_order is 'linear':
            self.out_indim_max = self.num_environment+self.num_managers+1 #+1 for bias
            self.action_indim_max = self.num_environment+self.num_managers+1 #+1 for bias
        elif type(self.agent_order) is int:
            out_indim_list = np.zeros(self.num_managers)
            
            for i in enumerate(self.num_agents):
                inedge = self.network_prespecified[:,i]
                num_in = np.sum(inedge)
                indim = np.sum( comb( num_in, np.arange(1,num_in+1)  ) ).astype(int) #excluding bias
                indim+1


        if self.initializer_type is "zeros":
            init_out_weights = tf.constant(0.0,shape=[self.out_indim_max,self.num_managers],dtype=tf.float32)
            init_action_weights = tf.constant(0.0,shape=[self.action_indim_max,self.num_actors],dtype=tf.float32)
        if self.initializer_type is "normal":
            init_out_weights = tf.constant(np.random.randn(self.out_indim_max,self.num_managers),dtype=tf.float32)
            init_action_weights = tf.constant( np.random.randn(self.action_indim_max,self.num_actors), dtype=tf.float32 )            

        self.out_weights = tf.get_variable(dtype=tf.float32,  name='out_weights', initializer=init_out_weights)
        self.action_weights = tf.get_variable(dtype=tf.float32, name='action_weights', initializer=init_action_weights)
        self.action_state_list=[]

    def build_wave(self):        
        #For loop version
        #environment:shape=[self.batchsize,self.num_environment]        
        indata= tf.concat([tf.ones(shape=[self.batchsize,1],dtype=tf.float32), tf.concat([self.environment, tf.zeros(shape=[self.batchsize, self.num_managers],dtype=tf.float32)],axis=1  )],axis=1)
        #indata: shape = [batchsize, 1+num_environment+num_manager]
        shape_scatter_manager = tf.constant([self.batchsize,self.num_environment+self.num_managers+1])
        #shape_scatter_manager = tf.constant([1000,16])
        for i in range(self.num_managers):
            output = tf.sigmoid(tf.matmul(indata, tf.reshape(self.out_weights[:,i],shape=[-1,1]) ) )      
            indices = tf.concat([tf.reshape(tf.range(self.batchsize),shape=[-1,1]),tf.constant(1+self.num_environment+i,shape=[self.batchsize,1],dtype=tf.int32)], axis=1)
            updates = tf.reshape(output, shape=[-1] )#tf.reshape(output, shape=[-1,1] )
            scatter = tf.scatter_nd(indices, updates, shape_scatter_manager)
            indata = indata + scatter
        acion_state_list = []
        for i in range(self.num_actors):
            action_state = tf.matmul(indata, tf.reshape( self.action_weights[:,i]  ,shape=[-1,1])    )
            acion_state_list.append(action_state)
        self.action_state_list = acion_state_list            

    def _make_loss_task(self):
        cross_entropy_list = []
        for i in range(self.num_actors):
            a_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.env_pattern,logits=self.action_state_list[i])
            cross_entropy_list.append(a_loss)
        cross_entropy_mean = tf.reduce_mean(cross_entropy_list)
        return cross_entropy_mean

    def train(self, niters, lrinit=None, iplot=False, verbose=False):
        if( lrinit == None ):
            lrinit = self.start_learning_rate

        training_res_seq = []
        task_loss_seq = []
        weight_on_cost_seq = []
        lr_seq = []
        
        network_update_n = 0
        lr_decay = 0.
        lr = float(lrinit)
        
        for i  in range(niters):
            lr = float(lrinit) / float(1. + lr_decay) # Learn less over time
            #_,u0,u_t0,u_c0,w = self.sess.run([self.optimize,self.objective,self.objective_task,self.objective_cost,self.weight_on_cost], feed_dict={self.learning_rate:lr,self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input})
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _,u0,u_t0 = self.sess.run([self.optimize,self.objective,self.objective_task], feed_dict={self.learning_rate:lr,self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input},options=options,run_metadata=run_metadata)

            #Learning Rate Update
            if self.decay != None:
                lr_decay = lr_decay + self.decay
                
            if i%100==0:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('new_timeline_step_%d.json' % i, 'w') as f:
                    f.write(chrome_trace)

                #weight_on_cost_seq.append(w)
                training_res_seq.append(u0)
                #Get the strategy under hard-dunbar
                #task_loss_seq.append(u_t0)

                lr_seq.append(lr)                

            if verbose and (i%100==0):
                print('----')
                print  ("iter"+str(i)+": Loss function=" + str(u0) )
                #print("task loss:"+str(u_t0)+',cost loss:'+str(u_c0) )
                print("task loss:"+str(u_t0) )
                if self.weight_update is True:
                    print("weight on cost:"+str(w))
                    print('learning rate:'+str(lr))
                print('----')


#-------------------------------------------------------------------------

            
if __name__=="__main__":
    
    #------------------
    Description = 'Test_New'
    #------------------
        
    print ('Current working directory : ', os.getcwd() )

    start_time = time.time()

    parameters_for_grid = [
        {"innoise" : [1.], # Stddev on incomming messages
        "outnoise" : [1.], # Stddev on outgoing messages
        "num_environment" : [6], # Num univariate environment nodes
        "num_agents" : [40], # Number of Agents
        "num_managers" : ["AllButOne"], # Number of Agents that do not contribute
        "fanout" : [1], # Distinct messages an agent can say
        "envnoise": [1], # Stddev of environment state
        "envobsnoise" : [1], # Stddev on observing environment
        "batchsize" : [1000],#200,#, # Training Batch Size
        "weight_on_cost":[0.0],
        "weight_update":[False],
        "dunbar_number":[2],
        "dunbar_function":[None],  #"sigmoid_ratio","sigmoid_kth","L4"
        "initializer_type":["normal"],
        "dropout_type":[None],#,'AllIn'
        'dropout_rate':[.0],
        'decay':[.002],
        "description" : [Description],
        'network_update_method':[None],#'pruning',None
        "L1_norm":[.1]}
    ]

    parameters_temp = list(ParameterGrid(parameters_for_grid))
    n_param = len(parameters_temp)
    parameters = []
    
    for i in range(n_param):
        if parameters_temp[i]['dunbar_number']>parameters_temp[i]['num_environment']:
            pass
        else:
            if (parameters_temp[i]['dropout_type'] is None):
                parameters_temp[i]['dropout_rate']=0.0
            if (parameters_temp[i]['dropout_rate'] == 0.0):
                parameters_temp[i]['dropout_type']=None            
            if (parameters_temp[i]['dunbar_function'] is None):
                parameters_temp[i]['weight_update'] = False
                parameters_temp[i]['weight_on_cost'] = 0.0
            if (parameters_temp[i]['weight_update'] is False) and (parameters_temp[i]['weight_on_cost'] == 0.0):
                parameters_temp[i]['dunbar_function'] = None
                
            dup = False
            for p in parameters:
                if parameters_temp[i]==p:
                    dup=True
            if dup is False:
                parameters.append(parameters_temp[i])


    iteration_train = 400
    iteration_restart = 2

    exec_date = datetime.datetime.now(pytz.timezone('US/Mountain')).strftime('%B%d_%H%M')

    dirname ='./result_'+exec_date +'_' + Description

    createFolder(dirname)

    dirname_abs = os.getcwd() + '/result_'+exec_date +'_' + Description

    for i in range(n_param):
        parameter = parameters[i]
        batchsize = parameter['batchsize']
        num_environment = parameter['num_environment']
        env_class = Environment(batchsize,num_environment,env_type='match_mod2')
        env_class.create_env()
        env_input = env_class.environment
        env_pattern_input = env_class.env_pattern

        tf.reset_default_graph()
      
        
        print('********************'+'Setting'+str(i)+'********************')
        filename = '/Setting'+str(i)
        p = parameters[i]
        #orgA = runIteration(p, iteration_train, iteration_restart,filename,dirname_abs,verbose=False)        
        orgA = Organization(optimizer="None", tensorboard_filename='board_log_'+filename,env_input=env_input,env_pattern_input=env_pattern_input,**parameter)
        orgA.train(iteration_train, iplot=False, verbose=True)

        
    end_time = time.time()
    time_elapsed = end_time-start_time
    print('time: ',time_elapsed)

