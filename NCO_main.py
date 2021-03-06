# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:03:28 2018

@author: Hajime
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

def sigmoid(x,w):
  return  1/(1+np.exp(-np.dot(x,w)))

def tf_print(tensor, transform=None):

    # Insert a custom python operation into the graph that does nothing but print a tensors value
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(x if transform is None else transform(x))
        return x
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res


class Agent(object):
    _ids = count(0)
    def __init__(self, noiseinstd, noiseoutstd, num, fanout,  batchsize, numagents, numenv,dunbar_number ,initializer_type='normal' , **kwargs):#statedim,
        self.id = next(self._ids)
        self.num = num
        #self.statedim = statedim
        self.fanout = fanout
        self.noiseinstd = noiseinstd
        self.noiseoutstd = noiseoutstd
        self.batchsize = batchsize
        self.numagents = numagents
        self.numenv = numenv
        self.received_messages = None
        self.dunbar_number = dunbar_number
        self.initializer_type = initializer_type

    def set_received_messages(self, msgs):
        self.received_messages = msgs

    def create_out_matrix(self, indim):
        self.indim = indim
        if self.initializer_type is "zeros":
            init_out_weights = tf.constant(0.0,shape=[indim,self.fanout],dtype=tf.float32)
            init_action_weights = tf.constant(0.0,shape=[indim,1],dtype=tf.float32)
        if self.initializer_type is "normal":
            init_out_weights = tf.constant(np.random.randn(indim,self.fanout),dtype=tf.float32)
            init_action_weights = tf.constant( np.random.randn(indim,1), dtype=tf.float32 )            

        with tf.name_scope("Agents_Params"):
            self.out_weights = tf.get_variable(dtype=tf.float32, name=str(self.num) + "msg" +str(self.id), initializer=init_out_weights)
            self.action_weights = tf.get_variable(dtype=tf.float32, name=str(self.num) + "act" +str(self.id), initializer=init_action_weights)

            #For setting weak link to be exact zero
            self.out_weights_hard_dunbar = tf.get_variable(dtype=tf.float32, name=str(self.num) + "msg_hardbunbar" +str(self.id), initializer=init_out_weights)
            self.action_weights_hard_dunbar = tf.get_variable(dtype=tf.float32, name=str(self.num) + "act_hardbunbar" +str(self.id), initializer=init_action_weights)

        #Initialize with only Dunbar weights to be nonzero:
        '''
        init_out_weights = np.zeros([indim, self.fanout])
        init_out_weights[0:self.dunbar, :]= np.random.randn(self.dunbar,self.fanout)
        init_out_weights = tf.constant(init_out_weights)
        self.out_weights = tf.get_variable(dtype=tf.float32, name=str(self.num) + "msg" +str(self.id), initializer=init_out_weights) #, shape=[indim, self.fanout]
        self.action_weights = tf.get_variable(dtype=tf.float32, name=str(self.num) + "action" +str(self.id), shape=[indim,1])
        '''

    def set_output(self,output):
        with tf.name_scope("Agents_Output"):
            self.output = output
    def set_action(self,action):
        with tf.name_scope("Agents_Action"):
            self.action = action
    def set_state_action(self,state_action):
        with tf.name_scope("Agents_State"):
            self.state_action = state_action
    def set_biasedin(self,biasedin):
        with tf.name_scope("Agents_Biasedin"):
            self.biasedin = biasedin

    def set_inedge_prespecified(self, inedge):
        self.inedge = inedge #binary vector of input edge from prespecified network. Shape it to be [indim,1]

class Environment():
    def __init__(self,batchsize,num_environment,env_network=None,env_weight=None,env_type='match_mod2'):
        self.batchsize = batchsize
        self.num_environment = num_environment
        self.env_type = env_type
        if env_type == 'gen_from_network':
            self.env_network = env_network # (num_env+num_agents,num_agents) matrix
            self.env_weight=env_weight # (1+num_env+num_agents,num_agents) matrix. +1 for bias
            self.num_agents = self.env_network.shape[1]

    def create_env(self):
        if self.env_type is 'match_mod2':
            self.environment = np.random.randint(2,size = [self.batchsize, self.num_environment])
            left = self.environment[:,:int(self.num_environment/2)]
            right = self.environment[:,int(self.num_environment/2):]
            lmod = np.mod(np.sum(left,axis=1),2)
            rmod = np.mod(np.sum(right,axis=1),2)
            self.env_pattern = (lmod==rmod).astype(np.float32).reshape([-1,1])
            
        if self.env_type is 'gen_from_network':
            self.environment = np.random.randint(2,size = [self.batchsize, self.num_environment])
            if self.env_weight is None:
                self.env_weight = np.random.randn(1+self.num_environment+self.num_agents,self.num_agents)
            ones = np.ones([self.batchsize,1])
            zeros = np.zeros([self.batchsize,self.num_agents])
            indata = np.concatenate((ones,self.environment,zeros),axis=1)
            for i in range(self.num_agents):
                network = np.concatenate( ([1.], self.env_network[:,i]))
                w = (self.env_weight[:,i] * network).reshape([-1,1])
                m = np.dot(indata,w)
                indata[:,1+self.num_environment+i] = np.copy(m.flatten())
            self.env_pattern = m
            
                
            




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

        #For Debug
        self.task_loss_list = []
        self.cost_loss_list = []
        self.total_loss_list = []

        self.num_environment = num_environment
        self.num_agents = num_agents
        if num_managers is "AllButOne":
            self.num_managers = num_agents-1
        else:
            self.num_managers = num_managers
            
        self.agent_order = agent_order
            
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(innoise, outnoise, i, fanout, batchsize, num_agents, num_environment,dunbar_number,initializer_type=initializer_type)) #, statedim

        self.env_input = env_input
        self.env_pattern_input = env_pattern_input
        with tf.name_scope("Environment"):
            if env_pattern_input is None:
                self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float32)
                zero = tf.convert_to_tensor(0.0, tf.float32)
                greater = tf.greater(self.environment, zero, name="Organization_greater")
                self.environment = tf.where(greater, tf.ones_like(self.environment), tf.zeros_like(self.environment), name="where_env")
            else:
                self.environment = tf.placeholder(tf.float32,shape=[self.batchsize,self.num_environment])
                self.env_pattern= tf.placeholder(tf.float32,shape=[self.batchsize,1])
        with tf.name_scope('Network_prespecified'):
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

        self.build_org()

        with tf.name_scope("Objective"):
            self.objective_task = self._make_loss_task()
            self.objective_cost = self._make_loss_cost()
            self.objective_L1 = self._make_loss_L1()
            # self.objective = self.loss()
            self.objective = self.weight_on_cost * self.objective_cost + (1-self.weight_on_cost) * self.objective_task + self.objective_L1

        with tf.name_scope("Optimizer"):
            self.learning_rate = tf.placeholder(tf.float32)
            #self.optimize =tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective)
            self.optimize =tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective)
            #self.optimize =tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.objective)
            self.start_learning_rate = .1#15.
            self.decay = decay #None #.01


        if( tensorboard_filename == None ):
            self.writer = None
        else:
            self.writer = tf.summary.FileWriter(tensorboard_filename, self.sess.graph)
        self.saver = tf.train.Saver()

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def build_org(self):
        self.build_agent_params()
        self.build_wave()

    def build_agent_params(self):
        created = []
        if self.agent_order is 'linear':
            indim = self.num_environment
            for i,a in enumerate(self.agents):
                created.append(a)
                a.create_out_matrix(indim+1) # Plus one for bias
                a.set_inedge_prespecified( tf.reshape(self.network_prespecified[:indim,i],[-1,1]) ) #Shape it to be [indim,1] so that it can be multiplied with weights
                indim = indim+a.fanout #Update indim for the next agent
        elif type(self.agent_order) is int:
            #If the agent is high-order, indim is (sum_i orderCi)*(num of inedge in prespecified network)+1
            #Do not take full-DAG anymore to reduce parameters
            for i,a in enumerate(self.agents):
                inedge = self.network_prespecified[:,i]
                num_in = np.sum(inedge)
                indim = np.sum( comb( num_in, np.arange(1,num_in+1)  ) ).astype(int) #excluding bias
                a.create_out_matrix(indim+1) # Plus one for bias
                a.set_inedge_prespecified(self.network_prespecified[:indim,i].reshape([-1,1])) #Shape it to be [indim,1] so that it can be multiplied with weights
                                        
            
    def dropout_onlydunbar(self,weights,indata):        
        indata_dim = indata.get_shape()[1].value #dimension before adding bias
        top_k = tf.transpose(tf.nn.top_k(tf.transpose(weights), k=self.dunbar_number+1,sorted=True).values)
        top = top_k[0]
        bottom = top_k[self.dunbar_number]
        #print('bottom_dim'+str(bottom.get_shape() ))
        one_above_bottom = top_k[self.dunbar_number-1]
        #Under construction from here
        r = tf.random_uniform(shape=[self.batchsize,indata_dim])
        zeros = tf.constant(0.0,shape=[self.batchsize,indata_dim], dtype=tf.float32)
        bottom_tile = tf.tile([bottom[0]], [self.batchsize*indata_dim]  )
        bottom_mat = tf.reshape(bottom_tile , [self.batchsize,indata_dim]  )
        weights_tile = tf.tile( tf.reshape(weights,[-1]), [self.batchsize])
        weights_mat =  tf.reshape( weights_tile, [self.batchsize,indata_dim] )

        indata_new = tf.where( tf.logical_and(tf.less_equal(weights_mat, bottom_mat), tf.less(r, self.dropout_rate)  ),  zeros, indata  )
        return indata_new

                
    
    
    def create_poly(self,indata, order):
        indata_poly = indata
        indim_raw = indata.get_shape()[1]
        for i in range(2,order+1):
            for j in combinations(np.range(indim_raw), i  ):
                data = indata[:,j]
                data_poly = tf.reduce_prod(data,axis=1)
                indata_poly = tf.concat([indata_poly, data_poly], 1)
        return indata_poly
        

    def build_wave(self):
        self.outputs = []
        self.action_states = []

        for i,a in enumerate(self.agents):
            with tf.name_scope("Env_Noise"):
                envnoise = tf.random_normal([self.batchsize, self.num_environment], stddev=self.envobsnoise, dtype=tf.float32)

            with tf.name_scope("Indata"):
                inenv = self.environment
                indata=inenv
                for msg in self.outputs:
                    indata = tf.concat([indata, msg], 1)

                if self.dropout_type is 'AllIn':
                    indata = tf.nn.dropout(indata, keep_prob = 1.-self.dropout_rate)
                elif self.dropout_type is 'OnlyDunbar':
                    weights = a.out_weights[1:] + a.action_weights[1:]
                    indata = self.dropout_onlydunbar(weights,indata)
                    '''
                    indata_dim = indata.get_shape()[1].value #dimension before adding bias
                    top_k = tf.transpose(tf.nn.top_k(tf.transpose(weights), k=self.dunbar_number+1,sorted=True).values)
                    top = top_k[0]
                    bottom = top_k[self.dunbar_number]
                    #print('bottom_dim'+str(bottom.get_shape() ))
                    one_above_bottom = top_k[self.dunbar_number-1]
                    #Under construction from here
                    r = tf.random_uniform(shape=[self.batchsize,indata_dim])
                    zeros = tf.constant(0.0,shape=[self.batchsize,indata_dim], dtype=tf.float32)
                    bottom_tile = tf.tile([bottom[0]], [self.batchsize*indata_dim]  )
                    bottom_mat = tf.reshape(bottom_tile , [self.batchsize,indata_dim]  )
                    weights_tile = tf.tile( tf.reshape(weights,[-1]), [self.batchsize])
                    weights_mat =  tf.reshape( weights_tile, [self.batchsize,indata_dim] )

                    indata = tf.where( tf.logical_and(tf.less_equal(weights_mat, bottom_mat), tf.less(r, self.dropout_rate)  ),  zeros, indata  )
                    '''
                if self.agent_order is 'linear':
                    biasedin = tf.concat([tf.constant(1.0, dtype=tf.float32, shape=[self.batchsize, 1]), indata], 1) #dim:(batchsize,dim_indata+1)
                elif type(self.agent_order) is int:
                    indata_poly = self.create_poly(indata, self.agent_order)
                    biasedin = tf.concat([tf.constant(1.0, dtype=tf.float32, shape=[self.batchsize, 1]), indata_poly], 1) #dim:(batchsize,dim_indata+1)
                    
                    
            a.set_received_messages(biasedin)

            with tf.name_scope("Output"):
                out_weights_inedge = a.out_weights * tf.concat([tf.constant(1.0,dtype=tf.float32,shape=[1,1]), a.inedge],axis=0)
                output = tf.sigmoid(tf.matmul(biasedin, out_weights_inedge))
                self.dim_build_wave_output = self.sess.run(tf.shape(output))
            with tf.name_scope("Action"):
                #For convergence, we do not calculate the loss from the action.
                #Instead we use continuous loss function from sigmoid
                action_weights_inedge = a.action_weights * tf.concat([tf.constant(1.0,dtype=tf.float32,shape=[1,1]), a.inedge],axis=0)
                state_action = tf.matmul(biasedin, action_weights_inedge)
            a.set_output(output)
            a.set_state_action(state_action)
            a.set_biasedin(biasedin)

            self.outputs.append(output)
            self.action_states.append(state_action)
            

    def _make_loss_task(self):
        if self.env_pattern_input is None:
            pattern = self.pattern_detected()
        else:
            pattern = self.env_pattern
        cross_entropy_list = []
        for a in self.agents[self.num_managers:]:
            a_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=pattern,logits=a.state_action)
            cross_entropy_list.append(a_loss)
        cross_entropy_mean = tf.reduce_mean(cross_entropy_list)
        return cross_entropy_mean

    def _make_loss_cost(self):
        mean_listening_cost = self.dunbar_listening_cost()
        return mean_listening_cost
    
    def _make_loss_L1(self):
        sum_L1_norm = self.L1_norm_cost()
        return sum_L1_norm
        

    def pattern_detected(self):
        patterns = []
        midpoint = int(self.num_environment/2)

        left = tf.slice(self.environment, [0, 0], [self.batchsize, midpoint])
        right = tf.slice(self.environment, [0, midpoint], [self.batchsize, self.num_environment - midpoint])
        leftsum = tf.reshape( tf.reduce_sum(left, 1), shape=[self.batchsize,1] )
        rightsum = tf.reshape( tf.reduce_sum(right, 1), shape=[self.batchsize,1] )
        lmod = tf.mod(leftsum, 2)
        rmod = tf.mod(rightsum, 2)
        pattern = tf.cast(tf.equal(lmod, rmod), tf.float32)

        #test with easiest task
        #pattern = tf.constant(1., dtype=tf.float32,shape=[self.batchsize,1])

        return pattern


    def L1_norm_cost(self):
        penalties = []
        for x in self.agents:
            #weights_msg = tf.abs(x.out_weights[1:]) #bias doesn't count
            if x.id>=self.num_managers:
                weights_msg = 0.0
            else:
                weights_msg = tf.abs(x.out_weights[1:])
            if x.id<self.num_managers:
                weights_action = 0.0
            else:
                weights_action = tf.abs(x.action_weights[1:])
            weights = (weights_msg + weights_action) * x.inedge #CHECK DIMENSION
            cost = tf.reduce_sum( tf.abs(weights) )*self.L1_norm
            penalties.append( [cost] )
        penalty = tf.stack(penalties)
        return tf.reduce_sum(penalty)



    # Implemented Wolpert's model for Dunbars number
    # This involves looking at the biggest value, and the (dunbar+1) biggest value,
    # and basing punishment on the ratio between the two numbers, such that there
    # is an incentive the make the (dunbar+1)th largest value very small.
    def dunbar_listening_cost(self, cost_violate=1000., cutoff=.01):
        penalties = []
        print("Dunbar Number: " + str(self.dunbar_number))
        print('Dunbar Function:'+str(self.dunbar_function))
        if self.weight_update is True:
            print('Weight on cost:update')
        else:
            print('Weight on cost:fixed at '+str(self.weight_on_cost_val))
        for x in self.agents:
            #weights_msg = tf.abs(x.out_weights[1:]) #bias doesn't count
            if x.id>=self.num_managers:
                weights_msg = 0.0
            else:
                weights_msg = tf.abs(x.out_weights[1:])
            if x.id<self.num_managers:
                weights_action = 0.0
            else:
                weights_action = tf.abs(x.action_weights[1:])

            weights = (weights_msg + weights_action) * x.inedge #CHECK DIMENSION
            top_k = tf.transpose(tf.nn.top_k(tf.transpose(weights), k=self.dunbar_number+1,sorted=True).values)
            top = top_k[0]
            bottom = top_k[self.dunbar_number]
            one_above_bottom = top_k[self.dunbar_number-1]
            below_bottom = top_k[self.dunbar_number:]

            top = tf.cond(tf.reshape(tf.equal(top,0.0),[]),lambda:top+.0000001,lambda:top  ) #To make the denom non-zero
            one_above_bottom = tf.cond(tf.reshape(tf.equal(one_above_bottom,0.0),[]),lambda:one_above_bottom+.0000001,lambda:one_above_bottom  ) #To make the denom non-zero

            if self.dunbar_function is "sigmoid_ratio":
                #cost = tf.sigmoid( tf.divide(bottom, top) ) -.5# -.5 so that the min is zero.
                cost = tf.sigmoid( tf.divide(bottom, one_above_bottom) ) -.5# Denom is k-1th weight instead of top
                #cost = tf.sigmoid( tf.divide(bottom, one_above_bottom) ) - tf.sigmoid( tf.divide( one_above_bottom,bottom) ) -.5# Added -w_{k-1}/w_k for improvement
                
            elif self.dunbar_function is "sigmoid_kth":
                cost = tf.sigmoid( bottom ) -.5

            elif self.dunbar_function is "linear_ratio":
                cost =  tf.divide(bottom, top)

            elif self.dunbar_function is "linear_kth":
                cost =  bottom

            elif self.dunbar_function is "hard":
                cost = tf.cast( tf.greater( bottom, cutoff ), dtype=tf.float32) * cost_violate

            elif self.dunbar_function is "quad_ratio":
                #cost= tf.square( tf.divide(bottom,top) )
                cost= tf.square( tf.divide(bottom,one_above_bottom) )
                #cost = tf.square( tf.divide(bottom, one_above_bottom) ) - tf.square( tf.divide( one_above_bottom,bottom) ) -.5# Added -w_{k-1}/w_k for improvement

            elif self.dunbar_function is "quad_kth":
                cost = tf.square(bottom)

            elif self.dunbar_function is "L4":
                cost = tf.pow(tf.educe_sum(tf.pow(below_bottom/bottom, 4)),1/4)
                
            elif self.dunbar_function is None:
                cost = tf.constant(0., dtype=tf.float32)

            else:
                print("Dunbar cost function type not specified")
                return

            if x.id>=self.num_managers:
                cost = cost #penalize actor's weight severer
                
            penalties.append( [cost] )
        penalty = tf.stack(penalties)
        #return tf.sigmoid(tf.reduce_sum(penalty))
        #return tf.reduce_mean(penalty)
        return tf.pow( tf.reduce_sum( tf.pow(penalty, 4)  ), 1/4  )

        

    # This is the code responsible for running the optimizer and returning results
    def train(self, niters, lrinit=None, iplot=False, verbose=False):
        if( lrinit == None ):
            lrinit = self.start_learning_rate

        training_res_seq = []
        task_loss_seq = []
        #task_loss_hd_seq = []
        weight_on_cost_seq = []
        lr_seq = []
        
        network_update_n = 0
        lr_decay = 0.
        lr = float(lrinit)
        
        # For each iteration
        for i  in range(niters):
            # Run training, and adjust learning rate if it's an Optimizer that
            # works with decaying learning rates (some don't)
            lr = float(lrinit) / float(1. + lr_decay) # Learn less over time
            
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            
            _,u0,u_t0,u_c0,w = self.sess.run([self.optimize,self.objective,self.objective_task,self.objective_cost,self.weight_on_cost], feed_dict={self.learning_rate:lr,self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input},options=options,run_metadata=run_metadata)
            
            with open('timeline_02_step_%d.json' % i, 'w') as f:
                f.write(chrome_trace)
            #Learning Rate Update
            if self.decay != None:
                lr_decay = lr_decay + self.decay

            if i%100==0:
                weight_on_cost_seq.append(w)
                training_res_seq.append(u0)
                #Get the strategy under hard-dunbar
                
                task_loss_seq.append(u_t0)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_02_step_%d.json' % i, 'w') as f:
                    f.write(chrome_trace)
                
                
                #Weight on cost update
                if (i>niters/4) and (self.weight_update is True) and (self.weight_on_cost_val < .8):
                    self.weight_on_cost_val = self.weight_on_cost_val+.01
                    self.weight_on_cost.load(self.weight_on_cost_val,self.sess)
                    #_=self.sess.run(self.assign_weight)


                lr_seq.append(lr)                
                #if (i>niters/3) and ( self.decay != None ):
                    #pass
                    #lr = lr/(1.+self.decay)


            #Check performance with hard-dunbar (slow)
            if i%10000==0:
                pass
                #self.calc_performance_hard_dunbar()
                #task_loss_hd_seq.append(self.welfare_hard_dunbar)

            #Update network if pruning
            num_pruning = (self.num_agents+self.num_environment) - self.dunbar_number
            pruning_dur = int( niters/(num_pruning+3) )

            if i%pruning_dur==0 and i>0 and network_update_n<num_pruning:
                #Update network
                if self.network_update_method is not None:
                    out_params=[]
                    action_params=[]
                    for a in self.agents:
                        out_params.append( self.sess.run( a.out_weights ) )
                        action_params.append( self.sess.run(a.action_weights) )
                        
                    prune_target = int(self.num_agents-network_update_n-1)
                    if prune_target <= 0:
                        prune_target = 'All'

                    network_new = self.network_update(self.network_prespecified_input, self.network_update_method, out_params,action_params)

                    self.network_prespecified_input = np.copy(network_new)
                network_update_n = network_update_n+1
                
                lr_decay = 0.
                    

            if verbose and (i%100==0):
                print('----')
                print  ("iter"+str(i)+": Loss function=" + str(u0) )
                print("task loss:"+str(u_t0)+',cost loss:'+str(u_c0) )
                if self.weight_update is True:
                    print("weight on cost:"+str(w))
                    print('learning rate:'+str(lr))
                if i>2000:
                    print("task loss under hard dunbar:"+str(self.welfare_hard_dunbar))
                #print('Actual Loss function2:'+str(loss_actual0))
                print('----')
                #print('network:')
                #print(self.network_prespecified_input)
 
        # Get the strategy from all agents, which is the "network configuration" at the end
        #out_params = self.sess.run([a.out_weights for a in self.agents])
        out_params=[]
        action_params=[]
        out_params_hd = []
        action_params_hd = []
        for a in self.agents:
            out_params.append( self.sess.run( a.out_weights ) )
            action_params.append( self.sess.run(a.action_weights) )

            #out_params_hd.append( self.sess.run( a.out_weights_hard_dunbar,feed_dict={self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input} ) )
            #action_params_hd.append(self.sess.run(a.action_weights_hard_dunbar ,feed_dict={self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input} ))


        #welfare = self.sess.run(self.objective,feed_dict={self.environment:self.env_input,self.env_pattern:self.env_pattern_input})
        welfare = u_t0
        #welfare_hard_dunbar = self.welfare_hard_dunbar
        if( self.writer != None ):
            self.writer.close()

        #Sequence of the welfare
        self.training_res_seq = training_res_seq
        self.task_loss_seq = task_loss_seq
        #self.task_loss_hd_seq = task_loss_hd_seq
        self.weight_on_cost_seq = weight_on_cost_seq
        self.lr_seq = lr_seq

        self.out_params_final = out_params
        self.action_params_final = action_params
        self.welfare_final = welfare

        #self.out_params_hd_final = out_params_hd
        #self.action_params_hd_final = action_params_hd
        #self.welfare_hd_final = welfare_hard_dunbar

        return
    
    def find_kth(agent,k=None):
        if k is None:
            k = self.dunbar_number
        weights_msg = tf.abs(agent.out_weights[1:]) #bias doesn't count
        if agent.id>=self.num_managers:
            weights_msg = tf.zeros_like(agent.out_weights[1:])
        else:
            weights_msg = tf.abs(agent.out_weights[1:])
        if agent.id<self.num_managers:
            weights_action = tf.zeros_like(agent.action_weights[1:])
        else:
            weights_action = tf.abs(agent.action_weights[1:])

        weights = (weights_msg + weights_action) *agent.inedge #already bias not included
        top_k = tf.transpose(tf.nn.top_k(tf.transpose(weights), k=k+1,sorted=True).values)
        top = top_k[0]
        bottom = top_k[k] #One that has to be small already
        one_above_bottom = top_k[-1]#One that is allowed to be large
        
        return top, bottom, one_above_bottom, weights
    
    def network_update(self,network_old, update_method, out_params, action_params,target='All'):
        
        network_new = np.zeros_like(network_old)
        if update_method is "pruning":
            for i,a in enumerate(self.agents):
                if target is'All':
                    pass
                if type(target) is int:
                    if i<target:
                        continue
                out_param_i = out_params[i].flatten()
                action_params_i = action_params[i].flatten()
                len_weight = action_params_i.size
                if a.id>=self.num_managers:
                    out_param_i = np.zeros_like(out_param_i)
                if a.id<self.num_managers:
                    action_params_i = np.zeros_like(action_params_i)
                weights_i = np.abs(out_param_i+action_params_i) * network_old[:len_weight,i]                
                
                weights_numnonzero = np.count_nonzero(weights_i)                
                weights_min = np.min(weights_i[np.nonzero(weights_i)])
                min_pos = np.where(weights_i==weights_min  )        
                
                new_inedge = np.copy(network_old[:,i])
                if weights_numnonzero>self.dunbar_number:                                    
                    new_inedge[min_pos] = 0.                
                network_new[:,i] = new_inedge
        return network_new
        
    def calc_performance_hard_dunbar(self):
        for i,a in enumerate(self.agents):
            weights_msg = tf.abs(a.out_weights[1:]) #bias doesn't count
            if a.id>=self.num_managers:
                weights_msg = tf.zeros_like(a.out_weights[1:])
            else:
                weights_msg = tf.abs(a.out_weights[1:])
            if a.id<self.num_managers:
                weights_action = tf.zeros_like(a.action_weights[1:])
            else:
                weights_action = tf.abs(a.action_weights[1:])

            weights = (weights_msg + weights_action) *a.inedge #already bias not included
            top_k = tf.transpose(tf.nn.top_k(tf.transpose(weights), k=self.dunbar_number+1,sorted=True).values)
            top = top_k[0]
            bottom = top_k[self.dunbar_number] #One that has to be small already
            one_above_bottom = top_k[self.dunbar_number-1]#One that is allowed to be large
            zeros = tf.zeros_like(weights,dtype=tf.float32)

            out_weights_hard_dunbar_nobias = tf.where( tf.greater(weights, bottom), a.out_weights[1:], zeros  )
            action_weights_hard_dunbar_nobias = tf.where( tf.greater(weights, bottom), a.action_weights[1:], zeros  )

            out_weights_hard_dunbar =  tf.concat( [ [a.out_weights[0]], out_weights_hard_dunbar_nobias],axis=0  )
            action_weights_hard_dunbar =  tf.concat( [ [a.action_weights[0]], action_weights_hard_dunbar_nobias],axis=0  )
            
            a.out_weights_hard_dunbar = out_weights_hard_dunbar
            a.action_weights_hard_dunbar = action_weights_hard_dunbar


        #self.debug_out_weights_hd = self.sess.run(out_weights_hard_dunbar,feed_dict={self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input})
        #self.debug_action_weights_hd = self.sess.run(action_weights_hard_dunbar,feed_dict={self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input})

        #Under construction from here.
        cross_entropy_hard_dunbar_list = []
        output_hard_dunbar_list = []
        for i,a in enumerate(self.agents):

            inenv = self.environment
            indata=inenv
            for msg in output_hard_dunbar_list:
                indata = tf.concat([indata, msg], 1)
            biasedin = tf.concat([tf.constant(1.0, dtype=tf.float32, shape=[self.batchsize, 1]), indata], 1)
            
            out_weights_hd_inedge = a.out_weights_hard_dunbar * tf.concat([tf.constant(1.0,dtype=tf.float32,shape=[1,1]), a.inedge],axis=0)
            
            output_hard_dunbar = tf.sigmoid(tf.matmul(biasedin, out_weights_hd_inedge))

            output_hard_dunbar_list.append(output_hard_dunbar)

            if a.id>=self.num_managers:
                action_weights_hd_inedge = a.action_weights_hard_dunbar * tf.concat([tf.constant(1.0,dtype=tf.float32,shape=[1,1]), a.inedge],axis=0)
                
                state_action_hard_dunbar = tf.matmul(biasedin, action_weights_hd_inedge)
                a_loss_hard_dunbar = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.env_pattern,logits=state_action_hard_dunbar)
                cross_entropy_hard_dunbar_list.append(a_loss_hard_dunbar)

                #self.debug_state_action_hd = self.sess.run(state_action_hard_dunbar)

        self.cross_entropy_mean_hard_dunbar = tf.reduce_mean(cross_entropy_hard_dunbar_list)
        self.welfare_hard_dunbar = self.sess.run(self.cross_entropy_mean_hard_dunbar,feed_dict={self.environment:self.env_input,self.env_pattern:self.env_pattern_input,self.network_prespecified:self.network_prespecified_input})


def runIteration(parameter,iter_train,iter_restart,filename,dirname,env_type='match_mod2',verbose=False,save_result=True):
    batchsize = parameter['batchsize']
    num_environment = parameter['num_environment']
    env_class = Environment(batchsize,num_environment,env_type=env_type,)
    env_class.create_env()
    env_input = env_class.environment
    env_pattern_input = env_class.env_pattern

    pickle.dump(env_pattern_input, open(dirname+filename+"_env_pattern_input.pickle","wb"))

    welfare_final_iterations = []

    for i in range(iter_restart):
        
        start_time = time.time()
        
        
        tf.reset_default_graph()
        tf.summary.FileWriterCache.clear()

        filename_trial = filename + '_trial' + str(i)
        orgA = Organization(optimizer="None", tensorboard_filename='board_log_'+filename,env_input=env_input,env_pattern_input=env_pattern_input,**parameter)
        orgA.train(iter_train, iplot=False, verbose=verbose)


        end_time = time.time()
        time_elapsed = end_time-start_time

        parameter['time_elapsed']=time_elapsed

        if save_result:

            pickle.dump(parameter, open(dirname+filename_trial+"_parameters.pickle","wb"))
            pickle.dump(orgA.training_res_seq, open(dirname+filename_trial+"_training_res_seq.pickle","wb"))
            pickle.dump(orgA.task_loss_seq, open(dirname+filename_trial+"_task_loss_seq.pickle","wb"))
            #pickle.dump(orgA.task_loss_hd_seq, open(dirname+filename_trial+"_task_loss_hd_seq.pickle","wb"))
    
            pickle.dump(orgA.weight_on_cost_seq, open(dirname+filename_trial+"_weight_on_cost_seq.pickle","wb"))
            pickle.dump(orgA.lr_seq, open(dirname+filename_trial+"_lr_seq.pickle","wb"))
    
            pickle.dump(orgA.out_params_final, open(dirname+filename_trial+"_out_params_final.pickle","wb"))
            pickle.dump(orgA.action_params_final, open(dirname+filename_trial+"_action_params_final.pickle","wb"))
            #pickle.dump(orgA.out_params_hd_final, open(dirname+filename_trial+"_out_params_hd_final.pickle","wb"))
            #pickle.dump(orgA.action_params_hd_final, open(dirname+filename_trial+"_action_params_hd_final.pickle","wb"))
        


        
        orgA.welfare_final
        welfare_final_iterations.append(orgA.welfare_final)
    if save_result:
        pickle.dump(welfare_final_iterations, open(dirname+filename+"_welfare_final_iterations.pickle","wb"))
    

    return orgA


def getScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


if __name__=="__main__":
    
    #------------------
    Description = 'NoPruning_to_MeasureTime'
    #------------------
    
    
    print ('Current working directory : ', os.getcwd() )
    os.chdir(getScriptPath())
    print ('Changed working directory : ', os.getcwd() )


    start_time = time.time()

    parameters_for_grid = [
        {"innoise" : [1.], # Stddev on incomming messages
        "outnoise" : [1.], # Stddev on outgoing messages
        "num_environment" : [6], # Num univariate environment nodes
        "num_agents" : [20], # Number of Agents
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
    iteration_restart = 1

    exec_date = datetime.datetime.now(pytz.timezone('US/Mountain')).strftime('%B%d_%H%M')

    dirname ='./result_'+exec_date +'_' + Description

    createFolder(dirname)

    dirname_abs = os.getcwd() + '/result_'+exec_date +'_' + Description

    for i in range(n_param):
        print('********************'+'Setting'+str(i)+'********************')
        filename = '/Setting'+str(i)
        p = parameters[i]
        orgA = runIteration(p, iteration_train, iteration_restart,filename,dirname_abs,verbose=True)

        '''
        # NOTE: We run all simulations on background processes.
        # This is because Tensorflow does not release its memory after we
        # finish running a network, so if we run many simulations in one process
        # we'll swell to using 200GB of memory. This way memory is forcibly
        # freed by process termination after each simulation.
        proc = multiprocessing.Process(target=runIteration, args=(p, iteration_train, iteration_restart,filename,))
        proc.start()
        proc.join()
        '''




    end_time = time.time()
    time_elapsed = end_time-start_time
    print('time: ',time_elapsed)
