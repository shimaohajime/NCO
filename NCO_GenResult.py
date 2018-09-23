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
    def __init__(self,params,out_param_hd,action_param_hd,training_res_seq,task_loss_seq,task_loss_hd_seq):
        self.out_param_hd = out_param_hd
        self.action_param_hd = action_param_hd
        self.params = params    
        
        self.training_res_seq = training_res_seq    
        self.task_loss_seq=task_loss_seq
        self.task_loss_hd_seq=task_loss_hd_seq
        
    def draw_graph(self):
        
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
        
        nx.draw(G,pos=pos,node_color=color_list,with_label=True)
        
    def graph_welfare(self):
        
        fig, ax = plt.subplots()
        ax.plot([1],[1])
        ax.set_xlim(0,len(self.training_res_seq))
        ax.set_ylim(0,np.max(self.training_res_seq))
        ax.set_ylabel("Loss")
        ax.set_xlabel("Training Epoch")
        #ax.plot(np.arange(len(y)), np.log(y),".")
        #line.set_data(np.arange(len(y)), np.log(y))
        #fig.canvas.draw()
        ax.plot(np.arange(len(training_res_seq)), training_res_seq,".")
        ax.plot(np.arange(len(task_loss_seq)), task_loss_seq,".")
        ax.plot(np.arange(len(task_loss_hd_seq)), task_loss_hd_seq,".")


#nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
        
        
if __name__=="__main__":
    
    dirname = 'result_September23_0440/'

    filename_setting = "Setting"+str(0)+'_'
    filname_trial = "trial"+str(0)+'_'
    
    filename = dirname+filename_setting+filname_trial
    
    
    params = pickle.load(open(filename+"parameters.pickle","rb") )
    out_param_hd = pickle.load(open(filename+"out_params_hd_final.pickle","rb") )
    action_param_hd = pickle.load( open(filename+"action_params_hd_final.pickle","rb")  )
    training_res_seq = pickle.load(open(filename+"training_res_seq.pickle","rb") )
    task_loss_seq = pickle.load(open(filename+"task_loss_seq.pickle","rb") )
    task_loss_hd_seq = pickle.load(open(filename+"task_loss_hd_seq.pickle","rb") )

    result = NCO_result(params,out_param_hd,action_param_hd,training_res_seq,task_loss_seq,task_loss_hd_seq)


    result.graph_welfare()
    
