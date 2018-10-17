#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:17:14 2018

@author: hajime
"""
import networkx as nx
import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt



class NCO_result():
    def __init__(self,params,out_param_hd,action_param_hd,training_res_seq,task_loss_seq,task_loss_hd_seq,lr_seq,filename=None):
        self.out_param_hd = out_param_hd
        self.action_param_hd = action_param_hd
        self.params = params    
        
        self.training_res_seq = training_res_seq    
        self.task_loss_seq=task_loss_seq
        self.task_loss_hd_seq=task_loss_hd_seq
        
        self.lr_seq = lr_seq
        
        self.filename = filename
        
    def graph_network(self):
        
        num_agents = self.params['num_agents']
        if self.params["num_managers"] == 'AllButOne':
            num_managers = num_agents-1
        else:
            num_managers = self.params['num_managers']
            
        num_environment = self.params['num_environment']
        
        position={}
        color = []        
        G = nx.DiGraph()        
        for i in range(num_environment):
            #G.add_node(i, node_color="b", name="E" + str(i))
            G.add_node(i, node_color="g", name="E")
        for aix, agent in enumerate(self.out_param_hd): #Managers
            nodenum = num_environment +aix
            if aix<num_managers:
                G.add_node(nodenum, node_color='b', name = "M" + str(aix))
                for eix, val in enumerate(agent.flatten()):
                    if eix>0: #to avoid bias
                        if abs(val)>.0001:
                            G.add_edge(eix-1, nodenum, width=val)
        for aix, agent in enumerate(self.action_param_hd): #Actors
            nodenum = num_environment +aix
            if aix>=num_managers:
                G.add_node(nodenum, node_color='r', name = "A" + str(aix))
                for eix, val in enumerate(agent.flatten()):
                    if eix>0: #to avoid bias
                        if abs(val)>.0001:
                            G.add_edge(eix-1, nodenum, width=val)
                            
        hpos = np.zeros(num_environment+num_agents)
        for i in range(num_environment,num_environment+num_managers):
            
            path_to_actors = []
            for j in range(num_environment+num_managers,num_environment+num_agents):
                try:
                    path_to_actors.append(nx.shortest_path_length(G,i,j))
                except:
                    pass
            try:
                hpos[i] =  np.max(path_to_actors)
            except:
                hpos[i]=0
        
        hpos[:num_environment] = np.max(hpos) + 1
        
        vpos = np.zeros(num_environment+num_agents)
        
        for i in range(num_environment+num_agents):
            for j in range(np.max(hpos).astype(int)+1):
                vpos[np.where(hpos==j)] = np.arange( len( np.where(hpos==j)[0] ) )
                if np.mod(j,2) == 1.:
                    vpos[np.where(hpos==j)] = vpos[np.where(hpos==j)] +.5
        
        
        color_list = []
        label_list = []
        for i in range(num_environment+num_agents):
            G.node[i]['pos'] = (hpos[i],vpos[i])
            if i<num_environment:
                color_list.append('g')
                label_list.append('E')
            elif i<num_environment+num_managers:
                color_list.append('b')
                label_list.append('M')
            else:
                color_list.append('r')
                label_list.append('A')
            
        pos=nx.get_node_attributes(G,'pos')
        fig = plt.figure()
        nx.draw(G,pos=pos,node_color=color_list,with_label=True)
        if self.filename is not None:
            fig.savefig(self.filename+"_network.png")
        plt.clf()
        
        
    def graph_welfare(self):
        
        fig, ax = plt.subplots()
        ax.plot([1],[1])
        ax.set_xlim(0,len(self.training_res_seq))
        ax.set_ylim(0,np.max(self.task_loss_hd_seq))
        ax.set_ylabel("Loss")
        ax.set_xlabel("Training Epoch")
        #ax.plot(np.arange(len(y)), np.log(y),".")
        #line.set_data(np.arange(len(y)), np.log(y))
        #fig.canvas.draw()
        ax.plot(np.arange(len(training_res_seq)), training_res_seq,".")
        ax.plot(np.arange(len(task_loss_seq)), task_loss_seq,".")
        ax.plot(np.arange(len(task_loss_hd_seq)), task_loss_hd_seq,"*")
        if self.filename is not None:
            fig.savefig(self.filename+"_loss.png")
        plt.clf()
        
        
    def graph_across_epoch(self,var,ylabel,save=False):
        
        fig, ax = plt.subplots()
        ax.plot([1],[1])
        ax.set_xlim(0,len(var))
        ax.set_ylim(0.0,np.max(var))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training Epoch")
        #ax.plot(np.arange(len(y)), np.log(y),".")
        #line.set_data(np.arange(len(y)), np.log(y))
        #fig.canvas.draw()
        ax.plot(np.arange(len(var)), var,"-.")
        if (self.filename is not None) and save:
            fig.savefig(self.filename+"_"+ylabel+".png")
        plt.show()
        plt.clf()
        


#nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
        

def calc_Dunbar_list(action_param,out_param, num_agents,num_managers,dunbar_number,**kwarg):
    
    Dunbar_ratio = []
    for a in range(num_agents):
        if num_managers == "AllButOne":
            num_managers = num_agents-1
        if a<num_managers:
            weight_sorted = - np.sort(  -np.abs( out_param[a][1:] ).flatten() )
        elif a>=num_managers:
            weight_sorted = - np.sort(  -np.abs( action_param[a][1:] ).flatten() )
        Dplus1_th = weight_sorted[dunbar_number]
        D_th = weight_sorted[dunbar_number-1]
        Dunbar_ratio.append(Dplus1_th/D_th)
    return Dunbar_ratio
            
    
    
        
if __name__=="__main__":

    iteration_restart = 2
    n_param = 2
    
    #dirname = 'result_October15_1345/'
    dirname = 'result_October16_1501_PruningStartFromBottom/'
    
    i_trial = 0
    i_setting = 1
    
    
    filname_trial = "trial"+str(i_trial)+'_'

    filename_setting = "Setting"+str(i_setting)+'_'

    filename = dirname+filename_setting+filname_trial
            
    params = pickle.load(open(filename+"parameters.pickle","rb") )

    welfare_final_iterations = pickle.load(open(dirname+filename_setting+"welfare_final_iterations.pickle","rb") )
    #welfare_final_iterations = pickle.load(open(dirname+'Setting'+str(i_setting)+'_trial19_'+"welfare_final_iterations.pickle","rb") )
    
    min_loss = np.min(welfare_final_iterations)
    print('Best case loss: '+str(min_loss))
    i_trial_best = np.argmin(welfare_final_iterations)
    max_loss = np.max(welfare_final_iterations)
    print('Worst case loss: '+str(max_loss))
    i_trial_worst = np.argmax(welfare_final_iterations)
    

    out_param_hd = pickle.load(open(filename+"out_params_hd_final.pickle","rb") )
    action_param_hd = pickle.load( open(filename+"action_params_hd_final.pickle","rb")  )
    training_res_seq = pickle.load(open(filename+"training_res_seq.pickle","rb") )
    task_loss_seq = pickle.load(open(filename+"task_loss_seq.pickle","rb") )
    task_loss_hd_seq = pickle.load(open(filename+"task_loss_hd_seq.pickle","rb") )

    out_param = pickle.load(open(filename+"out_params_final.pickle","rb") )
    action_param = pickle.load( open(filename+"action_params_final.pickle","rb")  )
    
    env_pattern = pickle.load( open(dirname+filename_setting+"env_pattern_input.pickle","rb")  )
    
    Dunbar_list = calc_Dunbar_list(action_param,out_param,**params)
    
    
    
    lr_seq = pickle.load(open(filename+"lr_seq.pickle","rb") )
    


    result = NCO_result(params,out_param_hd,action_param_hd,training_res_seq,task_loss_seq,task_loss_hd_seq,lr_seq,filename=filename)

    result.graph_across_epoch(result.task_loss_seq,ylabel='Task Loss')

    result.graph_across_epoch(result.lr_seq,ylabel='Learning Rate')
    
    '''
    for i in range(iteration_restart):
        for j in range(n_param):
            filname_trial = "trial"+str(i)+'_'
    
            filename_setting = "Setting"+str(j)+'_'
    
            filename = dirname+filename_setting+filname_trial
                
            
            params = pickle.load(open(filename+"parameters.pickle","rb") )
            out_param_hd = pickle.load(open(filename+"out_params_hd_final.pickle","rb") )
            action_param_hd = pickle.load( open(filename+"action_params_hd_final.pickle","rb")  )
            training_res_seq = pickle.load(open(filename+"training_res_seq.pickle","rb") )
            task_loss_seq = pickle.load(open(filename+"task_loss_seq.pickle","rb") )
            task_loss_hd_seq = pickle.load(open(filename+"task_loss_hd_seq.pickle","rb") )
        
            result = NCO_result(params,out_param_hd,action_param_hd,training_res_seq,task_loss_seq,task_loss_hd_seq,filename=filename)
        
            result.graph_network()
            result.graph_welfare()
            
    '''
    
