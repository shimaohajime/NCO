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
from itertools import count
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import os
import networkx as nx
import re
import pickle
import multiprocessing


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
    def __init__(self, noiseinstd, noiseoutstd, num, fanout, statedim, batchsize, numagents, numenv,dunbar , **kwargs):
        self.id = next(self._ids)
        self.num = num
        self.statedim = statedim
        self.fanout = fanout
        self.noiseinstd = noiseinstd
        self.noiseoutstd = noiseoutstd
        self.batchsize = batchsize
        self.numagents = numagents
        self.numenv = numenv
        self.received_messages = None
        self.dunbar = dunbar
    
    def set_received_messages(self, msgs):
        self.received_messages = msgs    
    # Run once at startup in network.py to initialize the out weights and 
    # (optionally) print debugging data
    # Note: Indim is how many Independent Messages each agent can send
    # In other words, if indim=2 then each agent can send two distinct
    # messages based on their observations. This is currently nonsensical
    # in the nonlinear model.
    def create_out_matrix(self, indim):
        #Initialize without any condition
        self.out_weights = np.empty([indim, self.fanout])
        self.action_weights = np.empty([indim,1])
        
        #Initialize with only Dunbar weights to be nonzero:
        '''
        init_out_weights = np.zeros([indim, self.fanout])
        init_out_weights[0:self.dunbar, :]= np.random.randn(self.dunbar,self.fanout)
        init_out_weights = tf.constant(init_out_weights)
        self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "msg" +str(self.id), initializer=init_out_weights) #, shape=[indim, self.fanout]
        self.action_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "action" +str(self.id), shape=[indim,1])
        '''
        
        
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with out_weights: %s" % (self.num, str(sess.run(self.out_weights)))
            
    def set_output(self,output):
        self.output = output
    def set_action(self,action):
        self.action = action


class Organization(object):
    def __init__(self, num_environment, num_agents, num_managers, innoise,
                     outnoise, fanout, statedim, envnoise, envobsnoise,
                     batchsize, optimizer, weight_on_cost=0. ,dunbar=2 ,dunbar_type='soft',randomSeed=False, tensorboard=None, **kwargs):
        #For Debug
        self.task_loss_list = []
        self.cost_loss_list = []
        self.total_loss_list = []

        self.num_environment = num_environment
        self.num_agents = num_agents
        self.num_managers = num_managers
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(innoise, outnoise, i, fanout, statedim, batchsize, num_agents, num_environment,dunbar))
        #self.environment = np.random.randn(batchsize, num_environment)
        #self.environment = (self.environment>0.).astype(int) #Discretize the environments to (0,1)
        self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float64)
        zero = tf.convert_to_tensor(0.0, tf.float64)
        greater = tf.greater(self.environment, zero)
        self.environment = tf.where(greater, tf.ones_like(self.environment), tf.zeros_like(self.environment))        
        
        #self.weight_on_cost = tf.convert_to_tensor(weight_on_cost, dtype=tf.float64) #the weight on the listening cost on loss function 
        # self.weight_on_cost = tf.constant(weight_on_cost, dtype=tf.float64) #the weight on the listening cost on loss function 
        self.weight_on_cost = weight_on_cost #the weight on the listening cost on loss function 
        self.dunbar = dunbar #Dunbar number
        self.dunbar_type = dunbar_type
        
        self.build_org()
        
        self.objective_task = self._make_loss_task()
        self.objective_cost = self._make_loss_cost()
        # self.objective = self.loss()
        self.objective = self.weight_on_cost * self.objective_cost + (1-self.weight_on_cost) * self.objective_task
        
        self.learning_rate = tf.placeholder(tf.float64)


        self.optimize =tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective)
        #self.optimize =tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective)
        self.start_learning_rate = .01#15.
        self.decay = .001 #None


        self.sess = tf.Session()
        if( tensorboard == None ):
            self.writer = None
        else:
            self.writer = tf.summary.FileWriter(tensorboard, self.sess.graph)
        self.saver = tf.train.Saver()
        
        
        
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        

        
    def build_org(self):
        self.build_agent_params()
        self.build_wave()

    def build_agent_params(self):
        created = []
        indim = self.num_environment
        for i,a in enumerate(self.agents):
            created.append(a)
            a.create_out_matrix(indim+1) # Plus one for bias
            indim+=a.fanout

    def build_wave(self):
        self.outputs = []
        self.actions = []
        for i,a in enumerate(self.agents):
            envnoize = np.random.randn(self.batchsize, self.num_environment)*self.envobsnoise        
            inenv = self.environment 
            indata=inenv
            for msg in self.outputs:
                #indata = np.concatenate((indata, msg), axis=1)
                indata = tf.concat([indata, msg], 1)
            #biasedin = np.concatenate( (indata,np.ones([self.batchsize,1])),axis=1 )
            biasedin = tf.concat([tf.constant(1.0, dtype=tf.float64, shape=[self.batchsize, 1]), indata], 1)
            a.set_received_messages(biasedin)
            output = tf.sigmoid(tf.matmul(biasedin, a.out_weights))
            #output = sigmoid(biasedin, a.out_weights)
            #action = (sigmoid(biasedin, a.action_weights).flatten()>.5)
            action_cont = tf.sigmoid(tf.matmul(biasedin, a.action_weights)) #
            zero = tf.convert_to_tensor(0.0, tf.float64)
            greater = tf.greater(action_cont, zero)
            action = tf.where(greater, tf.ones_like(action_cont), tf.zeros_like(action_cont))
            a.set_action(action)
            self.outputs.append(output)
            self.actions.append(action)
    '''
    def loss(self):    
        punishments = []
        sum_wrong_action = tf.Variable(0.0, dtype=tf.float64)
        sum_listening_cost = tf.Variable(0.0, dtype=tf.float64)
        pattern = self.pattern_detected()
        
        zero = tf.convert_to_tensor(0.0, dtype=tf.float64)
        one = tf.convert_to_tensor(1.0, dtype=tf.float64)

        for a in self.agents[self.num_managers:]:
            #wrong_action =  np.sum(a.action!=pattern)
            wrong_action = tf.reduce_mean(  tf.abs(a.action-pattern) )
            sum_wrong_action += wrong_action            
            #tf.cond(tf.less(wrong_action,tf.constant(0., dtype=tf.float64)),lambda:tf.Print(wrong_action,[wrong_action],'Negative loss from task'),lambda:tf.Print(wrong_action,[wrong_action],''))
        if self.dunbar_type=='soft':
            sum_listening_cost = self.dunbar_listening_cost()
        if self.dunbar_type=='hard':
            sum_listening_cost = self.dunbar_listening_cost_hard()
        #tf.cond(tf.less(sum_listening_cost,tf.constant(0., dtype=tf.float64)),lambda:tf.Print(sum_listening_cost,[sum_listening_cost],'Negative loss from cost'),lambda:tf.Print(sum_listening_cost,[sum_listening_cost],''))
                        
        loss = sum_wrong_action*(one-self.weight_on_cost) + sum_listening_cost*self.weight_on_cost        
        #print_loss = tf.Print(loss,[sum_wrong_action,sum_listening_cost,loss],"task,cost,total loss")
                
        return loss
    '''
    # def loss(self):
    #     one = 1.0 # tf.constant(1.0, dtype=tf.float64)
    #     sum_wrong_action = self.loss_task()
    #     return sum_wrong_action
    #     sum_listening_cost = self.loss_cost()
    #     loss_total = sum_wrong_action*(one-self.weight_on_cost) + sum_listening_cost*self.weight_on_cost
    #     #loss_total = sum_wrong_action + sum_listening_cost
    #     return loss_total        
        
    def _make_loss_task(self):
        sum_wrong_action = tf.Variable(0.0, dtype=tf.float64)
        pattern = self.pattern_detected()
        zero = 0.0 # # tf.convert_to_tensor(0.0, dtype=tf.float64)
        one = 1.0 # tf.convert_to_tensor(1.0, dtype=tf.float64)
        for a in self.agents[self.num_managers:]:
            #wrong_action =  np.sum(a.action!=pattern)
            wrong_action = tf.reduce_mean(  tf.abs(a.action-pattern) )
            sum_wrong_action += wrong_action
        return sum_wrong_action

    def _make_loss_cost(self):
        sum_listening_cost = tf.Variable(0.0, dtype=tf.float64)
        if self.dunbar_type=='soft':
            sum_listening_cost = self.dunbar_listening_cost()
        if self.dunbar_type=='hard':
            sum_listening_cost = self.dunbar_listening_cost_hard()
        return sum_listening_cost
            

    
    def agent_punishment(self,pattern,action):
        neg = tf.convert_to_tensor(-1.0, dtype=tf.float64)
        one = tf.convert_to_tensor(1.0, dtype=tf.float64)
        one_minus_pattern = tf.subtract(one, pattern)
        one_minus_action = tf.subtract(one, action)
        
        false_negative = tf.multiply(pattern,one_minus_action)
        false_positive = tf.multiply(one_minus_pattern, action)
        punishment =  tf.reduce_sum( tf.add( false_negative, false_positive) )
        return punishment
        
        
    def pattern_detected(self):
        patterns = []
        midpoint = int(self.num_environment/2)
        
        #left = self.environment[:,0:midpoint]
        #right = self.environment[:,midpoint:]
        #leftsum = np.sum(left,1)
        #rightsum = np.sum(right,1)
        #lmod = np.mod(leftsum,2)
        #rmod = np.mod(rightsum,2)
        #pattern = (lmod==rmod).astype(int)
        
        left = tf.slice(self.environment, [0, 0], [self.batchsize, midpoint])
        right = tf.slice(self.environment, [0, midpoint], [self.batchsize, self.num_environment - midpoint])
        leftsum = tf.reduce_sum(left, 1)
        rightsum = tf.reduce_sum(right, 1)
        lmod = tf.mod(leftsum, 2)
        rmod = tf.mod(rightsum, 2)
        pattern = tf.cast(tf.equal(lmod, rmod), tf.float64)
        return pattern
    
    

    # Implemented Wolpert's model for Dunbars number
    # This involves looking at the biggest value, and the (dunbar+1) biggest value,
    # and basing punishment on the ratio between the two numbers, such that there
    # is an incentive the make the (dunbar+1)th largest value very small.
    def dunbar_listening_cost(self):
        penalties = []
        ten = tf.convert_to_tensor(10.0, dtype=tf.float64)
        five = tf.convert_to_tensor(5.0, dtype=tf.float64)
        print("Dunbar: " + str(self.dunbar))
        for x in self.agents:
            weights = tf.abs(x.out_weights[1:]) #bias doesn't count
            top_k = tf.transpose(tf.nn.top_k(tf.transpose(weights), self.dunbar+1).values)
            top = top_k[0]
            #top = tf.Print(top, [top], message="Top: ")
            bottom = top_k[self.dunbar]
            #bottom = tf.Print(bottom, [bottom], message="Bottom: ")
            #cost = tf.divide(bottom, top)
            cost = tf.divide(tf.sigmoid(bottom), tf.sigmoid(top)) # At Wolpert's suggestion
            penalties += [cost]
        penalty = tf.stack(penalties)
        return tf.sigmoid(tf.reduce_sum(penalty))


    #Hard constraint. If the listening is more than dunbar number, add a big loss.
    def dunbar_listening_cost_hard(self, cost_violate=1000., cutoff=.01):
        penalties = []
        print("Dunbar: " + str(self.dunbar))
        for x in self.agents:
            weights = tf.abs(x.out_weights[1:]) #bias doesn't count
            
            zero = tf.convert_to_tensor(0.0, tf.float64)
            cost_violate = tf.convert_to_tensor(cost_violate,tf.float64)
            greater = tf.greater(weights, zero)
            weights_above_cutoff = tf.where(greater, tf.ones_like(weights), tf.zeros_like(weights))
            n_weights_above_cutoff = tf.reduce_sum( tf.cast(weights_above_cutoff, tf.float64) )
            cost = tf.cast(tf.greater(n_weights_above_cutoff,self.dunbar),tf.float64)*cost_violate
            
            penalties += [cost]
        penalty = tf.stack(penalties)
        return tf.reduce_sum(penalty)



    
    # This is the code responsible for running the optimizer and returning results
    def train(self, niters, lrinit=None, iplot=False, verbose=False):
        if( lrinit == None ):
            lrinit = self.start_learning_rate
            
            
        if iplot:
            fig, ax = plt.subplots()
            ax.plot([1],[1])
            ax.set_xlim(0,niters)
            ax.set_ylim(0,20)
            ax.set_ylabel("Welfare (Log)")
            ax.set_xlabel("Training Epoch")
            #line = ax.lines[0]

            
        training_res = []
        self.pattern_debug = self.sess.run(self.pattern_detected())
        # For each iteration
        for i  in range(niters):
            # Run training, and adjust learning rate if it's an Optimizer that
            # works with decaying learning rates (some don't)
            lr = float(lrinit)
            if( self.decay != None ):
                lr = float(lrinit) / (1 + i*self.decay) # Learn less over time
            _,u0,u_t0,u_c0 = self.sess.run([self.optimize,self.objective,self.objective_task,self.objective_cost], feed_dict={self.learning_rate:lr})
            #u0,u_t0,u_c0 = self.sess.run([self.objective,self.objective_task,self.objective_cost], feed_dict={self.learning_rate:lr})
            #u0,u_t0,u_c0 = self.sess.run([self.objective,self.objective_task,self.objective_cost], feed_dict={self.learning_rate:lr})
            # Evaluates our current progress towards objective
            #u = self.sess.run(self.objective)
            #u2,u_t2,u_c2 = self.sess.run([self.objective,self.objective_task,self.objective_cost])
            one = tf.convert_to_tensor(1.0, dtype=tf.float64)
            weight_c = self.weight_on_cost # self.sess.run(self.weight_on_cost)
            weight_t = 1.0 - self.weight_on_cost # self.sess.run(1.0-self.weight_on_cost)
            loss_actual0 = weight_t*u_t0 + weight_c*u_c0
            #loss_actual0 = u_t0 + u_c0
            #loss_actual2 = weight_t*u_t2 + weight_c*u_c2

            '''
            u_t = self.sess.run(self.objective_task)
            self.task_loss_list.append(u_t)
            u_c = self.sess.run(self.objective_cost)
            self.cost_loss_list.append(u_c)
            self.total_loss_list.append(u)
            training_res.append(u)
            loss_actual = weight_t*u_t + weight_c*u_c
            
            '''
            if verbose:
                if i%10==0:
                    #print  ("iter"+str(i)+": Loss function=" + str(u) )
                    
                    print('----')
                    #print("task loss:"+str(u_t)+',cost loss:'+str(u_c) ) 
                    #print ('weight on task'+str(weight_t)+',weight on cost:'+str(weight_c))
                    #print('Actual Loss function:'+str(loss_actual))
                    print  ("iter"+str(i)+": Loss function0=" + str(u0) )
                    print("task loss2:"+str(u_t0)+',cost loss2:'+str(u_c0) ) 
                    print('Actual Loss function2:'+str(loss_actual0))
                    print('----')
                    
                    #print  ("iter"+str(i)+": Loss function0=" + str(u0) )
                    print('---------------')
                            
        if iplot:
            #line.set_data(np.arange(len(training_res)), np.log(training_res))
            #fig.canvas.draw()
            ax.plot(np.arange(len(training_res)), np.log(training_res),".")


        # Get the strategy from all agents, which is the "network configuration" at the end
        out_params = self.sess.run([a.out_weights for a in self.agents])
        welfare = self.sess.run(self.objective)
        if( self.writer != None ):
            self.writer.close()
            
        self.out_params = out_params
        self.training_res = training_res
        self.welfare = welfare
        return

    
class Results(object):
    def __init__(self, training_res, listen_params, welfare, cost, binary_cutoff=0.0):
        self.training_res = training_res
        self.listen_params = listen_params
        self.welfare = welfare
        self.welfareCost = cost
        self.G = None
        self.CG = None
        self.binary_cutoff=binary_cutoff

    # This is Justin's original graphing code (more or less)
    # It is intended for rendering networks as PNGs directly in NetworkX,
    # which I haven't been doing.
    def generate_graph(self):
        numenv = len(self.listen_params[0].flatten())-1 #minus one for bias
        numnodes = numenv + len(self.listen_params)
        G = nx.DiGraph()
        
        G.clear()
        hpos=0
        for i in range(numenv):
            G.add_node(i, node_color="b", name="E" + str(i),pos=(hpos,0))
            hpos=hpos+1
        for aix, agent in enumerate(self.listen_params):
            nodenum = numenv +aix
            G.add_node(nodenum, node_color='r', name = "A" + str(aix))
            for eix, val in enumerate(agent.flatten()):
                if eix>0: #to avoid bias
                    if abs(val) > self.binary_cutoff:
                        G.add_edge(eix-1, nodenum, width=val)
        self.G = G
        
        #debug
        self.numenv=numenv
        self.nomnodes=numnodes
        
        


if __name__=="__main__":
    start_time = time.time()
    
    tf.reset_default_graph()
    parameters = []
    # Trivial network: 1 agent, no managers, 5 env nodes
    parameters.append(
        {"innoise" : 2, # Stddev on incomming messages
        "outnoise" : 2, # Stddev on outgoing messages
        "num_environment" : 6, # Num univariate environment nodes
        "num_agents" : 10, # Number of Agents
        "num_managers" : 5, # Number of Agents that do not contribute
        "fanout" : 1, # Distinct messages an agent can say
        "statedim" : 1, # Dimension of Agent State
        "envnoise": 25, # Stddev of environment state (NO LONGER USED)
        "envobsnoise" : 1, # Stddev on observing environment
        "batchsize" : 1000,#200,#, # Training Batch Size
        "weight_on_cost":.99,
        "description" : "Baseline"}
    )
    
    iterations=100000
    orgA = Organization(optimizer="None",**parameters[0])
    orgA.train(iterations, iplot=False, verbose=False)
    
    orgA_result = Results(training_res=orgA.training_res, listen_params=orgA.out_params, welfare=orgA.welfare, cost=None,binary_cutoff=.001)
    orgA_result.generate_graph()
    
    #-------------------
    position={}
    color = []    
    for i in range(parameters[0]['num_environment']):
        position[i] = (i,0)
        color.append('b')
    hpos=1
    for i in range(parameters[0]['num_environment'], parameters[0]['num_environment']+ parameters[0]['num_agents']):
        position[i] = (i-parameters[0]['num_environment'],hpos)
        color.append('r')
        hpos=hpos + i    
    nx.draw(orgA_result.G, with_labels=True, font_weight='bold',pos=position,node_color=color)
    plt.savefig("plot1.png")

    #nx.draw_kamada_kawai(orgA_result.G, with_labels=True, font_weight='bold')
    
    #orgA.out_params
    
    fig, ax = plt.subplots()
    ax.plot([1],[1])
    ax.set_xlim(0,iterations)
    ax.set_ylim(0,np.max(orgA.training_res))
    ax.set_ylabel("Loss")
    ax.set_xlabel("Training Epoch")
    #ax.plot(np.arange(len(y)), np.log(y),".")
    #line.set_data(np.arange(len(y)), np.log(y))
    #fig.canvas.draw()
    ax.plot(np.arange(len(orgA.training_res)), orgA.training_res,".")
    plt.savefig("plot2.png")
    
    end_time = time.time()
    time_elapsed = end_time-start_time
    print('time: ',time_elapsed)
    
    filename = "orgA"
    
    pickle.dump(orgA.out_params, open(filename + "_out_params.pickle", "wb"))
    pickle.dump(orgA.training_res, open(filename + "_res.pickle", "wb"))
    pickle.dump(orgA_result.G, open(filename + "_G.pickle", "wb"))


