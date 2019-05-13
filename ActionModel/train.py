#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:37:30 2019

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

import time
import datetime
import pytz
import pickle
from model import *
from sklearn.model_selection import ParameterGrid

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        

'''
'''        
Description = 'Test_RC'
plt.ioff()
exec_date = datetime.datetime.now(pytz.timezone('US/Mountain')).strftime('%B%d_%H%M')
dirname ='./result_'+exec_date +'_' + Description
createFolder(dirname)

parameters_for_grid = {'num_environment':[6,24,60],
                       'num_agent':[30,60,300],
                       'dunbar_number':[4,6],
                       'dunbar_env':[4,6],
                       'flag_train_W_x':[False,True]
                       }
parameters_temp = list(ParameterGrid(parameters_for_grid))
n_param_temp = len(parameters_temp)
parameters_list = []
for i in range(n_param_temp):
    param_candidate = parameters_temp[i]
    ##Exclude nonsense parameters
    parameters_list.append(parameters_temp[i])

n_rep=10    
batchsize = 500

final_loss_list = []
network_aa_list = []
network_ia_list = []

for param_i in range( len(parameters_list) ):
    org_parameters = parameters_list[param_i]
    print(org_parameters)
    final_action_loss_list = []
    final_network_list = []
    final_error_list = []
    print('********************'+'Setting'+str(param_i)+'********************')
    filename_setting = '/Param%i'%(param_i)
    
    num_environment = parameters_list[param_i]['num_environment']
    num_agent = parameters_list[param_i]['num_agent']
    dunbar_number = parameters_list[param_i]['dunbar_number']
    dunbar_env = parameters_list[param_i]['dunbar_env']
    flag_train_W_x = parameters_list[param_i]['flag_train_W_x']
    
    final_loss_reps=[]
    network_aa_reps=[]
    network_ia_reps=[]
    for rep_i in range(n_rep):
        torch.cuda.empty_cache()
        print('----Setting%i, rep%i-------'%(param_i,rep_i))
        start_time = time.time()
        filename_trial = '/Param%i_rep%i'%(param_i,rep_i)

        network_aa = np.zeros([num_agent,num_agent])
        for i in range(num_agent):
            listen = np.random.choice(num_agent, dunbar_number)
            network_aa[i,listen] =1
            
        network_ia = np.zeros([num_agent,num_environment])
        for i in range(num_agent):
            listen = np.random.choice(num_environment, dunbar_env)
            network_ia[i,listen] =1


        dataset_mod2 = OrgTask(num_environment=num_environment, batchsize=batchsize)
        train_loader = DataLoader(dataset_mod2,batch_size=batchsize)
        
        esn = ESN(num_agent=num_agent, num_environment_input=num_environment, num_environment_output=1, num_sample=batchsize,
                         alpha=.1, rho=.1,network_aa=network_aa, network_ia=network_ia, network_ao=None,
                         W_x=None, W_in=None, W_out=None,
                         x_init=None,
                         flag_train=True, flag_dynamic=False,n_it=3000,loss_function=nn.BCELoss(), lr=.01, num_iter_converge=50,
                         learning_method='gradient',beta=0.,
                         flag_train_W_x=flag_train_W_x,
                         flag_print_loss=False
                         )
        
        esn.train(train_loader)
        
        final_loss_reps.append( esn.loss )
        network_aa_reps.append(network_aa)
        network_ia_reps.append(network_ia)
        
    final_loss_list.append(final_loss_reps)
    network_aa_list.append(network_aa_reps)
    network_ia_list.append(network_ia_reps)
    

pickle.dump( final_loss_list, open(dirname+filename_setting+"_final_loss_list.pickle","wb") )
pickle.dump( network_aa_list, open(dirname+filename_setting+"_network_aa_list.pickle","wb") )
pickle.dump( network_ia_list, open(dirname+filename_setting+"_network_ia_list.pickle","wb") )
pickle.dump( network_ia_list, open(dirname+filename_setting+"_network_ia_list.pickle","wb") )

pickle.dump( parameters_list, open(dirname+filename_setting+"_parameters_list.pickle","wb") )
