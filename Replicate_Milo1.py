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
    def __init__(self, noiseinstd, noiseoutstd, num, fanout,  batchsize, numagents, numenv,dunbar , **kwargs):#statedim,
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
        with tf.name_scope("Agents_Params"):
            self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "msg" +str(self.id), initializer=tf.constant(0.0,shape=[indim,self.fanout],dtype=tf.float64))
            self.action_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "act" +str(self.id), initializer=tf.constant(0.0,shape=[indim,1],dtype=tf.float64))
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
                     batchsize, optimizer, weight_on_cost=0. ,dunbar=2 ,dunbar_type='soft',dunbar_function='linear_kth' ,randomSeed=False, tensorboard_filename=None, **kwargs):

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
            self.agents.append(Agent(innoise, outnoise, i, fanout, batchsize, num_agents, num_environment,dunbar)) #, statedim
        #self.environment = np.random.randn(batchsize, num_environment)
        #self.environment = (self.environment>0.).astype(int) #Discretize the environments to (0,1)
        with tf.name_scope("Environment"):
            self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float64)
            zero = tf.convert_to_tensor(0.0, tf.float64)
            greater = tf.greater(self.environment, zero, name="Organization_greater")
            self.environment = tf.where(greater, tf.ones_like(self.environment), tf.zeros_like(self.environment), name="where_env")

        #self.weight_on_cost = tf.convert_to_tensor(weight_on_cost, dtype=tf.float64) #the weight on the listening cost on loss function
        # self.weight_on_cost = tf.constant(weight_on_cost, dtype=tf.float64) #the weight on the listening cost on loss function
        self.weight_on_cost = weight_on_cost #the weight on the listening cost on loss function
        self.dunbar = dunbar #Dunbar number
        self.dunbar_type = dunbar_type
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
            self.start_learning_rate = .01#15.
            self.decay = .001 #None


        if( tensorboard_filename == None ):
            self.writer = None
        else:
            self.writer = tf.summary.FileWriter(tensorboard_filename, self.sess.graph)
        self.saver = tf.train.Saver()



        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        ###Check dimensions for # DEBUG:
        #self.dim_environment = self.sess.run(tf.shape(self.environment))
        ## DEBUG:


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
        #Debug
        self.outputs = []
        self.actions = []
        self.greaters = []
        self.action_conts = []

        '''
        self.dim_build_wave_biasedin=[]
        self.dim_build_wave_greater= []
        self.dim_build_wave_greater =[]
        self.dim_build_wave_action_cont=[]
        self.dim_build_wave_action=[]
        '''

        for i,a in enumerate(self.agents):
            #envnoize = np.random.randn(self.batchsize, self.num_environment)*self.envobsnoise
            with tf.name_scope("Env_Noise"):
                envnoise = tf.random_normal([self.batchsize, self.num_environment], stddev=self.envobsnoise, dtype=tf.float64)

            with tf.name_scope("Indata"):
                inenv = self.environment
                indata=inenv
                for msg in self.outputs:
                    #indata = np.concatenate((indata, msg), axis=1)
                    indata = tf.concat([indata, msg], 1)
                #biasedin = np.concatenate( (indata,np.ones([self.batchsize,1])),axis=1 )
                biasedin = tf.concat([tf.constant(1.0, dtype=tf.float64, shape=[self.batchsize, 1]), indata], 1)
            a.set_received_messages(biasedin)

            #self.dim_build_wave_biasedin.append( self.sess.run(tf.shape(biasedin)) )

            with tf.name_scope("Output"):
                output = tf.sigmoid(tf.matmul(biasedin, a.out_weights))
                self.dim_build_wave_output = self.sess.run(tf.shape(output))
            with tf.name_scope("Action"):
                action_cont = tf.sigmoid(tf.matmul(biasedin, a.action_weights)) #
                #zero = tf.convert_to_tensor(0.0, tf.float64)
                zero = tf.zeros_like(action_cont,dtype=tf.float64)
                greater = tf.greater(action_cont, zero)
                #action = tf.where(greater, tf.ones_like(action_cont), tf.zeros_like(action_cont))
                #action = tf.where(greater, tf.ones_like(action_cont), tf.zeros_like(action_cont))
                action = tf.round(action_cont)
                
                #For convergence, we do not calculate the loss from the action.
                #Instead we use continuous loss function from sigmoid
                
                state_action = tf.matmul(biasedin, a.action_weights)
                
                '''
                self.dim_build_wave_greater.append( self.sess.run(tf.shape(greater)) )
                self.dim_build_wave_action_cont.append (self.sess.run(tf.shape(action_cont)) )
                self.dim_build_wave_action.append( self.sess.run(tf.shape(action)) )
                '''
            a.set_output(output)
            a.set_action(action)
            a.set_state_action(state_action)
            a.set_biasedin(biasedin)
            
            #Debug            
            self.outputs.append(output)
            self.actions.append(action)
            self.greaters.append(greater)
            self.action_conts.append(action_cont)
            
            

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
        #sum_wrong_action = tf.Variable(0.0, dtype=tf.float64) #can't set those as variable
        pattern = self.pattern_detected()
        '''
        self.dim_makelosstask_pattern = self.sess.run( tf.shape(pattern) )
        self.dim_makelosstask_a_action = []
        self.dim_makelosstask_diff_abs = []
        self.dim_makelosstask_diff = []
        self.dim_makelosstask_wrong_action = []
        '''
        cross_entropy_list = []
        for a in self.agents[self.num_managers:]:
            a_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=pattern,logits=a.state_action)
            cross_entropy_list.append(a_loss)
            
            #wrong_action =  np.sum(a.action!=pattern)
            '''
            action_pattern_diff = a.action-pattern
            action_pattern_diff_abs = tf.abs(action_pattern_diff)
            wrong_action = tf.reduce_mean( action_pattern_diff_abs  )
            #sum_wrong_action = sum_wrong_action + wrong_action

            wrong_action_list.append(wrong_action)
            '''
            '''
            self.dtype_makelosstask_a_action = a.action.dtype
            self.dtype_makelosstask_pattern =  pattern.dtype
            self.dtype_makelosstask_action_pattern_diff =action_pattern_diff.dtype
            self.dtype_makelosstask_action_pattern_diff_abs = action_pattern_diff_abs.dtype
            self.dtype_makelosstask_wrong_action = wrong_action.dtype

            self.dim_makelosstask_a_action.append( self.sess.run(tf.shape(a.action)) )
            self.dim_makelosstask_diff.append(self.sess.run(tf.shape(action_pattern_diff)))
            self.dim_makelosstask_diff_abs.append(self.sess.run(tf.shape(action_pattern_diff_abs)))
            self.dim_makelosstask_wrong_action.append(self.sess.run(tf.shape(wrong_action)))
            '''
        '''
        sum_wrong_action = tf.add_n(wrong_action_list)

        self.wrong_action_list = wrong_action_list
        
        return sum_wrong_action
        '''
        cross_entropy_mean = tf.reduce_mean(cross_entropy_list)
        return cross_entropy_mean

    def _make_loss_cost(self):
        #sum_listening_cost = tf.Variable(0.0, dtype=tf.float64) #can't set as variable!
        if self.dunbar_type=='soft':
            sum_listening_cost = self.dunbar_listening_cost()
        if self.dunbar_type=='hard':
            sum_listening_cost = self.dunbar_listening_cost_hard()
        return sum_listening_cost


    '''
    def agent_punishment(self,pattern,action):
        neg = tf.convert_to_tensor(-1.0, dtype=tf.float64,name="agent_punishment_neg")
        one = tf.convert_to_tensor(1.0, dtype=tf.float64,name="agent_punishment_one")
        one_minus_pattern = tf.subtract(one, pattern)
        one_minus_action = tf.subtract(one, action)

        false_negative = tf.multiply(pattern,one_minus_action)
        false_positive = tf.multiply(one_minus_pattern, action)
        punishment =  tf.reduce_sum( tf.add( false_negative, false_positive) )
        return punishment
    '''


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
        leftsum = tf.reshape( tf.reduce_sum(left, 1), shape=[self.batchsize,1] )
        rightsum = tf.reshape( tf.reduce_sum(right, 1), shape=[self.batchsize,1] )
        lmod = tf.mod(leftsum, 2)
        rmod = tf.mod(rightsum, 2)
        pattern = tf.cast(tf.equal(lmod, rmod), tf.float64)
        
        #test with easiest task
        #pattern = tf.constant(1., dtype=tf.float64,shape=[self.batchsize,1])

        '''
        self.dim_pattern_detected_left =  self.sess.run(tf.shape(left))
        self.dim_pattern_detected_right =  self.sess.run(tf.shape(right))
        self.dim_pattern_detected_leftsum =  self.sess.run(tf.shape(leftsum))
        self.dim_pattern_detected_rightsum =  self.sess.run(tf.shape(rightsum))
        self.dim_pattern_detected_lmod =  self.sess.run(tf.shape(lmod))
        self.dim_pattern_detected_rmod =  self.sess.run(tf.shape(rmod))
        self.dim_pattern_detected_pattern =  self.sess.run(tf.shape(pattern))
        '''

        return pattern



    # Implemented Wolpert's model for Dunbars number
    # This involves looking at the biggest value, and the (dunbar+1) biggest value,
    # and basing punishment on the ratio between the two numbers, such that there
    # is an incentive the make the (dunbar+1)th largest value very small.
    def dunbar_listening_cost(self):
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
            
            if self.dunbar_function is "sigmoid_ratio":
                cost = tf.sigmoid( tf.divide(bottom, top) ) - .5# -.5 so that the min is zero.
                
            elif self.dunbar_function is "sigmoid_kth":
                cost = tf.sigmoid( bottom ) - .5
                
            elif self.dunbar_function is "linear_ratio":
                cost =  tf.divide(bottom, top) # At Wolpert's suggestion

            elif self.dunbar_function is "linear_kth":
                cost =  bottom
                
            else:
                print("Dunbar cost function type not specified")
                return
                
            penalties.append( [cost] )
        penalty = tf.stack(penalties)
        #return tf.sigmoid(tf.reduce_sum(penalty))
        return tf.reduce_sum(penalty)


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
            #one = tf.convert_to_tensor(1.0, dtype=tf.float64)
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
            loss_actual = weight_t*u_t + weight_c*u_c

            '''
            if i%100==0:
                training_res.append(u0)
                if verbose:
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
        "num_managers" : 5, # Number of Agents that do not contribute
        "fanout" : 1, # Distinct messages an agent can say
        #"statedim" : 1, # Dimension of Agent State
        "envnoise": 1, # Stddev of environment state
        "envobsnoise" : 1, # Stddev on observing environment
        "batchsize" : 1000,#200,#, # Training Batch Size
        "weight_on_cost":0.,
        "dunbar":3,
        "dunbar_function":"sigmoid_kth",
        "description" : "Baseline"}
    )

    iterations=20000
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
    print('****************')
    print('Dimensions')
    print('Environment:'+str(orgA.dim_environment))
    print('****************')
    print('dim_build_wave_action'+str(orgA.dim_build_wave_action))
    print('dim_build_wave_output'+str(orgA.dim_build_wave_output))
    print('dim_build_wave_greater'+str(orgA.dim_build_wave_greater))
    print('dim_build_wave_biasedin'+str(orgA.dim_build_wave_biasedin))
    print('dim_build_wave_action_cont'+str(orgA.dim_build_wave_action_cont))
    print('****************')
    print('dim_makelosstask_pattern'+str(orgA.dim_makelosstask_pattern))
    print('dim_makelosstask_a_action'+str(orgA.dim_makelosstask_a_action))
    print('dim_makelosstask_diff'+str(orgA.dim_makelosstask_diff))
    print('dim_makelosstask_wrong_action'+str(orgA.dim_makelosstask_wrong_action))

    print('wrong_action_list'+str(orgA.wrong_action_list))

    print('---')
    print('dtype_makelosstask_a_action'+str(orgA.dtype_makelosstask_a_action))
    print('dtype_makelosstask_pattern'+str(orgA.dtype_makelosstask_pattern))
    print('dtype_makelosstask_action_pattern_diff'+str(orgA.dtype_makelosstask_action_pattern_diff))
    print('dtype_makelosstask_action_pattern_diff_abs'+str(orgA.dtype_makelosstask_action_pattern_diff_abs))
    print('dtype_makelosstask_wrong_action'+str(orgA.dtype_makelosstask_wrong_action))


    print('****************')
    print('dim_pattern_detected_left'+str(orgA.dim_pattern_detected_left))
    print('dim_pattern_detected_right'+str(orgA.dim_pattern_detected_right))
    print('dim_pattern_detected_leftsum'+str(orgA.dim_pattern_detected_leftsum))
    print('dim_pattern_detected_rightsum'+str(orgA.dim_pattern_detected_rightsum))
    print('dim_pattern_detected_lmod'+str(orgA.dim_pattern_detected_lmod))
    print('dim_pattern_detected_rmod'+str(orgA.dim_pattern_detected_rmod))
    print('dim_pattern_detected_pattern'+str(orgA.dim_pattern_detected_pattern))
    '''





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
