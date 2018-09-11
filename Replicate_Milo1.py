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
    def __init__(self, noiseinstd, noiseoutstd, num, fanout,  batchsize, numagents, numenv,dunbar ,initializer_type , **kwargs):#statedim,
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
        self.dunbar = dunbar
        self.initializer_type = initializer_type

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
        #self.out_weights = np.empty([indim, self.fanout])
        #self.action_weights = np.empty([indim,1])

        #Initizalize to start from zero.
        if self.initializer_type is "zeros":
            init_out_weights = tf.constant(0.0,shape=[indim,self.fanout],dtype=tf.float64)
            init_action_weights = tf.constant(0.0,shape=[indim,1],dtype=tf.float64)
        if self.initializer_type is "normal":
            init_out_weights = tf.constant(np.random.randn(indim,self.fanout),dtype=tf.float64)
            init_action_weights = tf.constant( np.random.randn(indim,1), dtype=tf.float64 )
            
        with tf.name_scope("Agents_Params"):
            self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "msg" +str(self.id), initializer=init_out_weights)
            self.action_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "act" +str(self.id), initializer=init_action_weights)
            tf.summary.tensor_summary("Agent_out_weights",self.out_weights)
            tf.summary.tensor_summary("Agent_action_weights",self.action_weights)


        #Initialize with only Dunbar weights to be nonzero:
        '''
        init_out_weights = np.zeros([indim, self.fanout])
        init_out_weights[0:self.dunbar, :]= np.random.randn(self.dunbar,self.fanout)
        init_out_weights = tf.constant(init_out_weights)
        self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "msg" +str(self.id), initializer=init_out_weights) #, shape=[indim, self.fanout]
        self.action_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "action" +str(self.id), shape=[indim,1])
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


class Organization(object):
    def __init__(self, num_environment, num_agents, num_managers, innoise,
                     outnoise, fanout,  envnoise, envobsnoise,#statedim,
                     batchsize, optimizer, weight_on_cost=0.,initializer_type='zeros' ,dunbar=2 ,dunbar_function='linear_kth' ,randomSeed=False, tensorboard_filename=None, **kwargs):

        self.sess = tf.Session()

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
            self.agents.append(Agent(innoise, outnoise, i, fanout, batchsize, num_agents, num_environment,dunbar,initializer_type=initializer_type)) #, statedim
        with tf.name_scope("Environment"):
            self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float64)
            zero = tf.convert_to_tensor(0.0, tf.float64)
            greater = tf.greater(self.environment, zero, name="Organization_greater")
            self.environment = tf.where(greater, tf.ones_like(self.environment), tf.zeros_like(self.environment), name="where_env")

        self.weight_on_cost = weight_on_cost #the weight on the listening cost on loss function
        self.dunbar = dunbar #Dunbar number
        self.dunbar_function = dunbar_function

        self.build_org()

        with tf.name_scope("Objective"):
            self.objective_task = self._make_loss_task()
            self.objective_cost = self._make_loss_cost()
            # self.objective = self.loss()
            self.objective = self.weight_on_cost * self.objective_cost + (1-self.weight_on_cost) * self.objective_task
            tf.summary.scalar("Objective",self.objective)
            tf.summary.scalar("Objective_task",self.objective_task)
            tf.summary.scalar("Objective_cost",self.objective_cost)

        with tf.name_scope("Optimizer"):
            self.learning_rate = tf.placeholder(tf.float64)
            #self.optimize =tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective)
            self.optimize =tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective)
            self.start_learning_rate = .1#15.
            self.decay = .01 #None


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
        indim = self.num_environment
        for i,a in enumerate(self.agents):
            created.append(a)
            a.create_out_matrix(indim+1) # Plus one for bias
            indim = indim+a.fanout

    def build_wave(self):
        self.outputs = []
        self.action_states = []

        for i,a in enumerate(self.agents):
            with tf.name_scope("Env_Noise"):
                envnoise = tf.random_normal([self.batchsize, self.num_environment], stddev=self.envobsnoise, dtype=tf.float64)

            with tf.name_scope("Indata"):
                inenv = self.environment
                indata=inenv
                for msg in self.outputs:
                    indata = tf.concat([indata, msg], 1)
                biasedin = tf.concat([tf.constant(1.0, dtype=tf.float64, shape=[self.batchsize, 1]), indata], 1)
            a.set_received_messages(biasedin)

            with tf.name_scope("Output"):
                output = tf.sigmoid(tf.matmul(biasedin, a.out_weights))
                self.dim_build_wave_output = self.sess.run(tf.shape(output))
            with tf.name_scope("Action"):
                #For convergence, we do not calculate the loss from the action.
                #Instead we use continuous loss function from sigmoid                
                state_action = tf.matmul(biasedin, a.action_weights)
            a.set_output(output)
            a.set_state_action(state_action)
            a.set_biasedin(biasedin)
            
            #Debug            
            self.outputs.append(output)
            self.action_states.append(state_action)

    def _make_loss_task(self):
        pattern = self.pattern_detected()
        cross_entropy_list = []
        for a in self.agents[self.num_managers:]:
            a_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=pattern,logits=a.state_action)
            cross_entropy_list.append(a_loss)

        cross_entropy_mean = tf.reduce_mean(cross_entropy_list)
        return cross_entropy_mean

    def _make_loss_cost(self):
        mean_listening_cost = self.dunbar_listening_cost()
        return mean_listening_cost


    def pattern_detected(self):
        patterns = []
        midpoint = int(self.num_environment/2)

        left = tf.slice(self.environment, [0, 0], [self.batchsize, midpoint])
        right = tf.slice(self.environment, [0, midpoint], [self.batchsize, self.num_environment - midpoint])
        leftsum = tf.reshape( tf.reduce_sum(left, 1), shape=[self.batchsize,1] )
        rightsum = tf.reshape( tf.reduce_sum(right, 1), shape=[self.batchsize,1] )
        lmod = tf.mod(leftsum, 2)
        rmod = tf.mod(rightsum, 2)
        pattern = tf.cast(tf.equal(lmod, rmod), tf.float64)
        
        #test with easiest task
        #pattern = tf.constant(1., dtype=tf.float64,shape=[self.batchsize,1])

        return pattern



    # Implemented Wolpert's model for Dunbars number
    # This involves looking at the biggest value, and the (dunbar+1) biggest value,
    # and basing punishment on the ratio between the two numbers, such that there
    # is an incentive the make the (dunbar+1)th largest value very small.
    def dunbar_listening_cost(self, cost_violate=1000., cutoff=.01):
        penalties = []
        print("Dunbar Number: " + str(self.dunbar))
        print('Dunbar Function:'+self.dunbar_function)
        for x in self.agents:
            weights_msg = tf.abs(x.out_weights[1:]) #bias doesn't count
            weights_action = tf.abs(x.action_weights[1:])
            weights = weights_msg + weights_action
            top_k = tf.transpose(tf.nn.top_k(tf.transpose(weights), k=self.dunbar+1,sorted=True).values)
            top = top_k[0]
            bottom = top_k[self.dunbar]
            
            top = tf.cond(tf.reshape(tf.equal(top,0.0),[]),lambda:top+.00001,lambda:top  )
            
            if self.dunbar_function is "sigmoid_ratio":
                cost = tf.sigmoid( tf.divide(bottom, top) ) -.5# -.5 so that the min is zero.
                
            elif self.dunbar_function is "sigmoid_kth":
                cost = tf.sigmoid( bottom ) -.5
                
            elif self.dunbar_function is "linear_ratio":
                cost =  tf.divide(bottom, top)  # At Wolpert's suggestion

            elif self.dunbar_function is "linear_kth":
                cost =  bottom
                
            elif self.dunbar_function is "hard":
                cost = tf.cast( tf.greater( bottom, cutoff ), dtype=tf.float64) * cost_violate
                
            elif self.dunbar_function is "quad_ratio":
                cost= tf.square( tf.divide(bottom,top) )
                
            elif self.dunbar_function is "quad_kth":
                cost = tf.square(bottom)
                
                
            else:
                print("Dunbar cost function type not specified")
                return
                
            penalties.append( [cost] )
        penalty = tf.stack(penalties)
        #return tf.sigmoid(tf.reduce_sum(penalty))
        return tf.reduce_mean(penalty)





    # This is the code responsible for running the optimizer and returning results
    def train(self, niters, lrinit=None, iplot=False, verbose=False):
        if( lrinit == None ):
            lrinit = self.start_learning_rate


        training_res = []
        # For each iteration
        for i  in range(niters):
            # Run training, and adjust learning rate if it's an Optimizer that
            # works with decaying learning rates (some don't)
            lr = float(lrinit)
            if( self.decay != None ):
                lr = float(lrinit) / (1 + i*self.decay) # Learn less over time
            _,u0,u_t0,u_c0 = self.sess.run([self.optimize,self.objective,self.objective_task,self.objective_cost], feed_dict={self.learning_rate:lr})
            weight_c = self.weight_on_cost # self.sess.run(self.weight_on_cost)
            weight_t = 1.0 - self.weight_on_cost # self.sess.run(1.0-self.weight_on_cost)
            loss_actual0 = weight_t*u_t0 + weight_c*u_c0
            if i%100==0:
                training_res.append(u0)
                if verbose:
                    print('----')
                    print  ("iter"+str(i)+": Loss function0=" + str(u0) )
                    print("task loss2:"+str(u_t0)+',cost loss2:'+str(u_c0) )
                    print('Actual Loss function2:'+str(loss_actual0))
                    print('----')



        # Get the strategy from all agents, which is the "network configuration" at the end
        #out_params = self.sess.run([a.out_weights for a in self.agents])
        out_params=[]
        action_params=[]
        for a in self.agents:
            out_params.append( self.sess.run( a.out_weights ) )
            action_params.append( self.sess.run(a.action_weights) )
        welfare = self.sess.run(self.objective)
        
        if( self.writer != None ):
            self.writer.close()

        self.out_params_final = out_params
        self.action_params_final = action_params
        self.training_res_final = training_res
        self.welfare_final = welfare
                
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
    tf.summary.FileWriterCache.clear()
    parameters = []
    # Trivial network: 1 agent, no managers, 5 env nodes
    parameters.append(
        {"innoise" : 1., # Stddev on incomming messages
        "outnoise" : 1., # Stddev on outgoing messages
        "num_environment" : 6, # Num univariate environment nodes
        "num_agents" : 10, # Number of Agents
        "num_managers" : 9, # Number of Agents that do not contribute
        "fanout" : 1, # Distinct messages an agent can say
        #"statedim" : 1, # Dimension of Agent State
        "envnoise": 1, # Stddev of environment state
        "envobsnoise" : 1, # Stddev on observing environment
        "batchsize" : 1000,#200,#, # Training Batch Size
        "weight_on_cost":0.5,
        "dunbar":3,
        "dunbar_function":"quad_ratio",
        "initializer_type":"normal",
        "description" : "Baseline"}
    )

    iterations=50000
    orgA = Organization(optimizer="None", tensorboard_filename='board_log',**parameters[0])
    orgA.train(iterations, iplot=False, verbose=True)


    filename = "orgA"
    pickle.dump(parameters[0], open(filename+"_parameters.pickle","wb"))
    pickle.dump(orgA.training_res_final, open(filename+"_training_res_final.pickle","wb"))
    pickle.dump(orgA.out_params_final, open(filename+"_out_params_final.pickle","wb"))
    pickle.dump(orgA.action_params_final, open(filename+"_action_params_final.pickle","wb"))

    end_time = time.time()
    time_elapsed = end_time-start_time
    print('time: ',time_elapsed)





    '''
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


    if False:
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
    '''


    '''
    filename = "orgA"
    pickle.dump(orgA, open(filename+"_class.pickle","wb"))
    #pickle.dump(orgA.out_params, open(filename + "_out_params.pickle", "wb"))
    #pickle.dump(orgA.training_res, open(filename + "_res.pickle", "wb"))
    #pickle.dump(orgA_result.G, open(filename + "_G.pickle", "wb"))
    '''
