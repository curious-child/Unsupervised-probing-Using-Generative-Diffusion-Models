import copy
import json
import math
import os
import random

from scipy import stats
import itertools as it
import igraph
import numpy as np
import networkx as nx
import torch_geometric
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data, Batch, HeteroData
import torch
import numpy as np
from igraph import *
import matplotlib.pyplot as plt
from glob import glob
from sklearn.manifold import TSNE
import yaml
from torch_geometric.utils import k_hop_subgraph

from models.models import  diffusion_models


def grid_parameters_supervised_learning(train_params,net_params,loss_params,optimizer_params,**kwargs):
    parameters_list=[]
    for params_values in it.product(*train_params.values()):
        train_param = dict(zip(train_params.keys(), params_values))
        #  print(train_param)

        net_params_copy = net_params.copy()
        net_params_copy.pop("prelayers_gnn_param")
        net_params_copy.pop("enclayers_gnn_param")
        #net_params_copy.pop("mask_model")
        for params_values in it.product(*net_params_copy.values()):
            net_param = dict(zip(net_params_copy.keys(), params_values))

            # print(net_param["prelayers_gnn"])
            # print(net_param["enclayers_gnn"])
            for prelayers_values in it.product(*net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].values()):
                net_param["prelayers_gnn_param"] = dict(
                    zip(net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].keys(), prelayers_values))
                # print(net_param["prelayers_gnn_param"])
                for enclayers_values in it.product(
                        *net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].values()):
                    net_param["enclayers_gnn_param"] = dict(
                        zip(net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].keys(), enclayers_values))
                    # print(net_param["enclayers_gnn_param"])
                    # mask_net_params_copy = net_params["mask_model"].copy()
                    # mask_net_params_copy.pop("prelayers_gnn_param")
                    # for mask_net_values in it.product(*mask_net_params_copy.values()):
                    #     net_param["mask_model"] = dict(zip(mask_net_params_copy.keys(), mask_net_values))
                    #     for mask_prelayers_values in it.product(*net_params["mask_model"]["prelayers_gnn_param"][
                    #         net_param["mask_model"]["prelayers_gnn"]].values()):
                    #         net_param["mask_model"]["prelayers_gnn_param"] = dict(zip(
                    #             net_params["mask_model"]["prelayers_gnn_param"][
                    #                 net_param["mask_model"]["prelayers_gnn"]].keys(), mask_prelayers_values))

                    for params_values in it.product(*loss_params.values()):
                        loss_param = dict(zip(loss_params.keys(), params_values))
                      #  print("policy gradient:", loss_param["sample"])
                        for params_values in it.product(*optimizer_params.values()):
                            optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                            parameters_list.append((train_param,net_param,loss_param,optimizer_param))

    return parameters_list

def record_scores_analysis(configs_record_scores,metrics,train_validation,score_thershold):
    best_scores = float('-inf')
    best_config = None
    score_thershold_config = dict()
    for config_name,record_scores in configs_record_scores.items():
        cmp_best_score = max(record_scores[metrics][train_validation])
        if cmp_best_score > best_scores:
            best_scores = cmp_best_score
            best_config = config_name
        if cmp_best_score >= score_thershold:
            score_thershold_config[config_name] = cmp_best_score
    print("Best config:", best_config)
    print("Best scores:", best_scores)
    for n,config_name in enumerate(score_thershold_config.keys()):

        print("Score thershold config {}:\n".format(n), config_name)
    return best_config, best_scores,score_thershold_config
def grid_parameters_generative_learning(train_params, net_params, loss_params, optimizer_params, **kwargs):
    Hp_grid_analysis_file={"net":net_params,"train":train_params,"loss":loss_params,"optimizer":optimizer_params}
    Hp_grid_file=dict()
    for key,hp_params in Hp_grid_analysis_file.items():
        Hp_grid_file[key]=dict()
        for param_name, param_values in hp_params.items():
            if  not isinstance(param_values, list):
                raise ValueError("Error param_values type:{}".format(type(param_values)))
            if len(param_values) >1:
                Hp_grid_file[key][param_name]=param_values
        if len(Hp_grid_file[key])==0:
            Hp_grid_file.pop(key)
    parameters_list = []

    for params_values in it.product(*train_params.values()):
        train_param = dict(zip(train_params.keys(), params_values))
        #  print(train_param)

        net_params_copy = net_params.copy()

        for params_values in it.product(*net_params_copy.values()):
            net_param = dict(zip(net_params_copy.keys(), params_values))


            for params_values in it.product(*loss_params.values()):
                loss_param = dict(zip(loss_params.keys(), params_values))
                #  print("policy gradient:", loss_param["sample"])
                for params_values in it.product(*optimizer_params.values()):
                    optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                    parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param), copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))


    return parameters_list,Hp_grid_file
def grid_parameters_generative_learning_spdata(train_params, net_params, loss_params, optimizer_params, **kwargs):
    Hp_grid_analysis_file={"net":net_params.copy(),"train":train_params.copy(),"loss":loss_params.copy(),"optimizer":optimizer_params.copy()}
    Hp_grid_analysis_file["net"].pop("gnn_params")
    if "f_gnn_params" in net_params.keys():
        Hp_grid_analysis_file["net"].pop("f_gnn_params")
    Hp_grid_file=dict()
    for key,hp_params in Hp_grid_analysis_file.items():
        Hp_grid_file[key]=dict()
        for param_name, param_values in hp_params.items():
            if  not isinstance(param_values, list):
                raise ValueError("Error param_values type:{}".format(type(param_values)))
            if len(param_values) >1:
                Hp_grid_file[key][param_name]=param_values
        if len(Hp_grid_file[key])==0:
            Hp_grid_file.pop(key)
    parameters_list = []

    for params_values in it.product(*train_params.values()):
        train_param = dict(zip(train_params.keys(), params_values))
        #  print(train_param)

        net_params_copy = net_params.copy()
      #  print("net_params_copy:",net_params_copy.keys())
        net_params_copy.pop("gnn_params")
        if "f_gnn_params" in net_params_copy.keys():
            net_params_copy.pop("f_gnn_params")
            for params_values in it.product(*net_params_copy.values()):
                net_param = dict(zip(net_params_copy.keys(), params_values))
                for prelayers_values in it.product(*net_params["gnn_params"][net_param["gnn_name"]].values()):
                    net_param["gnn_param"] = dict(
                        zip(net_params["gnn_params"][net_param["gnn_name"]].keys(), prelayers_values))
                    # print(net_param["prelayers_gnn_param"])
                    for enclayers_values in it.product(
                            *net_params["f_gnn_params"][net_param["f_gnn_name"]].values()):
                        net_param["f_gnn_param"] = dict(
                            zip(net_params["f_gnn_params"][net_param["f_gnn_name"]].keys(), enclayers_values))

                        for params_values in it.product(*loss_params.values()):
                            loss_param = dict(zip(loss_params.keys(), params_values))
                            #  print("policy gradient:", loss_param["sample"])
                            for params_values in it.product(*optimizer_params.values()):
                                optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                                parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param), copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))

        else:
            for params_values in it.product(*net_params_copy.values()):
                net_param = dict(zip(net_params_copy.keys(), params_values))
                for prelayers_values in it.product(*net_params["gnn_params"][net_param["gnn_name"]].values()):
                    net_param["gnn_param"] = dict(
                        zip(net_params["gnn_params"][net_param["gnn_name"]].keys(), prelayers_values))
                    # print(net_param["prelayers_gnn_param"])

                    for params_values in it.product(*loss_params.values()):
                        loss_param = dict(zip(loss_params.keys(), params_values))
                        #  print("policy gradient:", loss_param["sample"])
                        for params_values in it.product(*optimizer_params.values()):
                            optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                            parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param),
                                                    copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))
    return parameters_list,Hp_grid_file
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def gen_graph(g_type, num_min=20, num_max=40):
    max_n = num_max
    min_n = num_min
    cur_n = np.random.randint(max_n - min_n + 1) + min_n

    if g_type == 'erdos_renyi':
        while True:
            ER_p=random.uniform(0.1,0.9)
            g = Graph.Erdos_Renyi(n=cur_n,p=ER_p,directed=False)
            if g.is_connected():
                break
    elif g_type == 'small-world':
        while True:
            nei_max=round(0.35*cur_n)
            nei_min=round(0.15*cur_n)
            WS_nei=random.randint(nei_min,nei_max)
            WS_p = random.uniform(0, 0.15)
            g = Graph.Watts_Strogatz(dim=1,size=cur_n,nei=WS_nei,p=WS_p)#可包含最近邻规则图（p=0）
            if g.is_connected():
                break
    elif g_type == 'barabasi_albert':
        while True:
            m_max = round(0.25 * cur_n)
            m_min = round(0.1 * cur_n)
            BA_m = random.randint(m_min, m_max)
            g = Graph.Barabasi(n=cur_n, m=BA_m,directed=False)
            if g.is_connected():

                break
    elif g_type=="static_power_law":
        while True:
            exp=random.uniform(2,3)
            pl_max = round(0.25*cur_n* cur_n)
            pl_min = round(0.05*cur_n* cur_n)
            pl_edges=random.randint(pl_min,pl_max)
            g = Graph.Static_Power_Law(n=cur_n,m=pl_edges,exponent_out=exp)
            if g.is_connected():
                break
    elif g_type == "K_Regular":
        while True:
            k_regular=random.randint(round(0.2*cur_n),cur_n-2)  #可包含完全图（cur-1）
            if k_regular*cur_n %2==0 and cur_n>=(k_regular+1) : #k*n必须为偶数 且k+1 ≤n
                g = Graph.K_Regular(n=cur_n, k=k_regular)
                if g.is_connected():
                    break

    return g

def to_gnngraph(g,features,targets=None,SL=True):

    if "None" in features:
        x = np.ones((g.vcount(), 1))
    elif "one-hot" in features:
        x=np.identity(g.vcount())
    else:
        x = np.column_stack(
            tuple(
                g.vs[feature] for feature in features
            )
        )
    x = torch.from_numpy(x).to(torch.float)
    if SL:
        y = g.vs[targets]
        y = torch.tensor(y).to(torch.float)
    else:
        # G=igraph.Graph.to_networkx(g)
        # y=[pre_graph_op(G)]
        y=None


    source_nodes1 = [edge.source for edge in g.es]
    target_nodes1 = [edge.target for edge in g.es]
    source_nodes = source_nodes1 + target_nodes1
    target_nodes = target_nodes1 + source_nodes1

    return Data(x=x,y=y,edge_index=torch.tensor([source_nodes,target_nodes],dtype=torch.long))

def graph_properties(gnn_g,targets):
    node_v = 1 - gnn_g.x
    node_mask = torch.bernoulli(node_v).to(torch.bool)
    nodes_to_keep = torch.nonzero(node_mask).squeeze(1)
    subgraph = gnn_g.subgraph(nodes_to_keep)
    subgraph =tg_utils.to_networkx(subgraph, to_undirected=True)
    net_G = tg_utils.to_networkx(gnn_g, to_undirected=True)
    if targets == "LCC":
        return len(max(nx.connected_components(subgraph), key=len))
    elif targets == "global_CC":
        return nx.transitivity(subgraph)
    elif targets == "average_CC":
        return nx.average_clustering(subgraph)
    elif targets == "natural_connectivity":
        estrada_index = nx.estrada_index(subgraph)
        # 自然连通性
        n = nx.number_of_nodes(subgraph)
        return np.log(estrada_index / n)
    elif targets=="global_efficiency":
        return nx.global_efficiency(subgraph)
    elif targets=="density":
        return nx.density(subgraph)
    else:
        raise ValueError("Error fitness_func_type:{}".format(targets))
def to_gnngraph_nx(g,features,targets=None,SL=True):

    if "None" in features:
        gnn_g=torch_geometric.utils.from_networkx(g,group_node_attrs=None)
        gnn_g.x = torch.ones(gnn_g.num_nodes, 1).float()

    elif "random" in features:
        gnn_g = torch_geometric.utils.from_networkx(g, group_node_attrs=None)
        x0 = torch.rand(gnn_g.num_nodes)
        gnn_g.x = x0.reshape(-1,).float()
    else:
        gnn_g = torch_geometric.utils.from_networkx(g, group_node_attrs=features)
    if SL:

        if targets in g.nodes[0].keys():
            y_dict = nx.get_node_attributes(g,targets)
            gnn_g.y = torch.tensor(list(y_dict.values())).to(torch.long)
        else:
            gnn_g.y = torch.tensor(graph_properties(gnn_g,targets)).to(torch.float)
            gnn_g.y_name=targets
    else:
        #G=igraph.Graph.to_networkx(g)
        gnn_g.y=None
    return gnn_g


def pre_DataSet_spdata( spdata_file_path,graph_file_path,windows,pred_len,interval_step,sampling_t,filter="*",**params):
    from tqdm import tqdm
    dataSet = []
    pred_times_len = windows + pred_len

    sampling_t_min = 0.1
    assert sampling_t >= sampling_t_min, "Error: sampling_t should be greater than or equal to 0.1"
    sampling_interval = int(sampling_t / sampling_t_min)
   # print("sampling_interval",sampling_interval)

    for file in tqdm(sorted(glob(spdata_file_path+'/'+filter))):
            try:
                file_graph_name=file.split("/")[-1]
                graph_data_path=graph_file_path+'/'+file_graph_name+'.graphml'
                nx_g=nx.read_graphml(graph_data_path)
                nx_g=nx.convert_node_labels_to_integers(nx_g)
                gnn_g = torch_geometric.utils.from_networkx(nx_g, group_node_attrs=None)

                for spdata_file in sorted(glob(file+'/*.pt')):
                    try:

                        loaded_data = torch.load(spdata_file)
                        torch_time_series = loaded_data['ys_dynamic'].t().unsqueeze(-1)  # [Node_num,T_obs_num,F] F=1 时间间隔为0.1
                        #print("torch_time_series shape:{}".format(torch_time_series.shape))
                        sampling_torch_time_series = torch_time_series[:,::sampling_interval, :]
                        # print("torch_time_series shape:{}".format(torch_time_series.shape))
                        # print("torch_time_series type:{}".format(torch_time_series.dtype))
                       # print("sampling_torch_time_series shape:{}".format(sampling_torch_time_series.shape))
                        time_len = sampling_torch_time_series.shape[1]
                    except Exception:

                        raise Exception("There are errors in {}".format(file))
                    assert time_len // pred_times_len > 0, "Error: data length is not enough!!!"
                    timeseries_data = sampling_torch_time_series.unfold(1, pred_times_len,
                                                                        interval_step)  # [Node_num,n,F,pred_times_len]
                    timeseries_data = timeseries_data.permute(0,1, 3, 2)  # [Node_num,n,pred_times_len,F]

                   # print("timeseries_data shape:{}".format(timeseries_data.shape))
                    timeseries_data = timeseries_data.unbind(1)
                    for time_series in timeseries_data:
                      #  print("time_series",time_series.shape)
                          torch_time_series = preprocess_gdata_sequence(time_series)
                          # print("time_series shape:{}".format(time_series.shape))
                          if isinstance(torch_time_series, tuple):
                              for time_series in torch_time_series:
                                  gnn_g.x = time_series.clone()  # [node_num,pred_times_len,F]
                                  assert gnn_g.x.shape[
                                             0] == gnn_g.num_nodes, "Error: node number is not equal to x number"
                                  dataSet.append(copy.deepcopy(gnn_g))
                          else:
                              raise ValueError("Error torch_time_series type:{}".format(type(torch_time_series)))


            except Exception:

                raise Exception("There are errors in {}".format(file))




    return dataSet
def preprocess_gdata_sequence(torch_spdata_series):

    torch_time_series_inversed=torch_spdata_series.clone()
    torch_time_series_inversed=torch.flip(torch_time_series_inversed,dims=[1])
    return  torch_time_series_inversed,torch_spdata_series

def preprocess_data_sequence(torch_time_series,data_filer,file_name=None):
    if data_filer=="*":
        torch_time_series_inversed=torch_time_series.clone()
        torch_time_series_inversed=torch.flip(torch_time_series_inversed,dims=[0])
        return  torch_time_series_inversed,torch_time_series
    elif data_filer=="*_increase":
        if "increase" in file_name:
            return torch_time_series
        else:
            torch_time_series_inversed = torch_time_series.clone()
            torch_time_series_inversed = torch.flip(torch_time_series_inversed, dims=[0])
            return torch_time_series_inversed
    elif data_filer=="*_decrease":
        if "decrease" in file_name:
            return torch_time_series
        else:
            torch_time_series_inversed = torch_time_series.clone()
            torch_time_series_inversed = torch.flip(torch_time_series_inversed, dims=[0])
            return torch_time_series_inversed
    else:
        raise ValueError("Error data_filer:{}".format(data_filer))

def pre_DataSet_Timeseries( file_path,windows,pred_len,interval_step,sampling_t,filter="*",**params):
    from  tqdm import tqdm
    dataSet=[]
    pred_times_len=windows+pred_len
    F_consistency=params["F_consistency"] if "F_consistency" in params.keys() else True
    sampling_t_min=0.1
    assert sampling_t>=sampling_t_min,"Error: sampling_t should be greater than or equal to 0.1"
    sampling_interval=int(sampling_t/sampling_t_min)
    for file in tqdm(sorted(glob(file_path+"/*/*.pt"))):
        try:
            loaded_data=torch.load(file)
            file_name=file.split("/")[-2]
          #  print(file_name)
            torch_time_series=loaded_data['ys_dynamic']#[T_obs_num,F] 时间间隔为0.1

            sampling_torch_time_series=torch_time_series[::sampling_interval,:]
            # print("torch_time_series shape:{}".format(torch_time_series.shape))
            # print("torch_time_series type:{}".format(torch_time_series.dtype))
            time_len=sampling_torch_time_series.shape[0]
        except Exception:

            raise Exception("There are errors in {}".format(file))
        assert time_len//pred_times_len>0,"Error: data length is not enough!!!"
        timeseries_data=sampling_torch_time_series.unfold(0,pred_times_len,interval_step)#[n,F,pred_times_len]
        if F_consistency:
            timeseries_data=timeseries_data.permute(0,2,1)#[n,pred_times_len,F]
        else:
            timeseries_data=timeseries_data.reshape(-1,pred_times_len).unsqueeze(-1)#[n*F,pred_times_len,1]
       # print("timeseries_data shape:{}".format(timeseries_data.shape))
        timeseries_data=timeseries_data.unbind(0)

        for time_series in timeseries_data:
            if "data_dropout" in params.keys() and np.random.uniform() > params["data_dropout"]:
               continue
            torch_time_series = preprocess_data_sequence(time_series, data_filer=filter, file_name=file_name)
            #print("time_series shape:{}".format(time_series.shape))
            if isinstance(torch_time_series,tuple):
                for time_series in torch_time_series:
                    dataSet.append(time_series.clone())
            elif isinstance(torch_time_series,torch.Tensor):
                dataSet.append(torch_time_series.clone())


    return dataSet

def pre_DataSet_Timeseries_old( file_path,windows,pred_len,interval_step,sampling_t,filter="*",**params):
    from  tqdm import tqdm
    dataSet=[]
    pred_times_len=windows+pred_len

    sampling_t_min=0.1
    assert sampling_t>=sampling_t_min,"Error: sampling_t should be greater than or equal to 0.1"
    sampling_interval=int(sampling_t/sampling_t_min)
    for file in tqdm(sorted(glob(file_path+"/{}/*.pt".format(filter)))):
        try:
            loaded_data=torch.load(file)
         #   print(file)
            file_name=file.split("/")[-2]
          #  print(file_name)
            torch_time_series=loaded_data['ys_dynamic']#[T_obs_num,F] 时间间隔为0.1

            sampling_torch_time_series=torch_time_series[::sampling_interval,:]
            # print("torch_time_series shape:{}".format(torch_time_series.shape))
            # print("torch_time_series type:{}".format(torch_time_series.dtype))
            time_len=sampling_torch_time_series.shape[0]
        except Exception:

            raise Exception("There are errors in {}".format(file))
        assert time_len//pred_times_len>0,"Error: data length is not enough!!!"
        timeseries_data=sampling_torch_time_series.unfold(0,pred_times_len,interval_step)#[n,F,pred_times_len]
        timeseries_data=timeseries_data.permute(0,2,1)#[n,pred_times_len,F]
       # print("timeseries_data shape:{}".format(timeseries_data.shape))
        timeseries_data=timeseries_data.unbind(0)

        for time_series in timeseries_data:
            torch_time_series = time_series
            #print("time_series shape:{}".format(time_series.shape))
            if isinstance(torch_time_series,tuple):
                for time_series in torch_time_series:
                    dataSet.append(time_series.clone())
            elif isinstance(torch_time_series,torch.Tensor):
                dataSet.append(torch_time_series.clone())


    return dataSet






def draw_3d(G,ax,pos,color_value):

    x, y, z = zip(*pos.values())

    for n, m in G.edges():
        zline = (z[n], z[m])
        xline = (x[n], x[m])
        yline = (y[n], y[m])
        ax.plot3D(xline, yline, zline,'k',linewidth=1)
    return ax.scatter3D(x, y, z, c=color_value, marker='o',alpha=1, s=100)

def visualization_evalution(pred,graph,target):


    data = graph  #可随机选取
    out =sigmoid( pred)
    G=torch_geometric.utils.to_networkx(data,to_undirected=True)

    ###########################2D可视化展示（可选）###########################################################
    fig=plt.figure("2d visualization of predict")
    pos = nx.kamada_kawai_layout(G)

    ax=fig.add_subplot(211)
    ax.set(title="Prediction  using GAT")
    nodes = nx.draw_networkx_nodes(G,pos=pos, node_color=out)
    nx.draw_networkx_edges(G, pos=pos,width=1)
    fig.colorbar(nodes)

    ax = fig.add_subplot(212)
    ax.set(title="Labels of network key nodes ")
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=target)
    nx.draw_networkx_edges(G, pos=pos, width=1)
    fig.colorbar(nodes)
    #######################################################################################################
    ###########################3D可视化展示（可选）###########################################################
    # fig = plt.figure("3d visualization of predict")
    # pos = nx.kamada_kawai_layout(G,dim=3)
    #
    # ax = fig.add_subplot(121,projection="3d")
    # ax.set(title="Prediction  using GAT")
    # ax0=draw_3d(G=G,pos=pos,ax=ax,color_value=out)
    # fig.colorbar(ax0,ax=ax)
    #
    # ax = fig.add_subplot(122,projection="3d")
    # ax.set(title="Labels of network key nodes ")
    # ax0 = draw_3d(G=G, pos=pos, ax=ax, color_value=target)
    # fig.colorbar(ax0, ax=ax)
    #######################################################################################################




    return 0
def pred_accuracy(pred, y, num_graph,data):  # torch tensor变量 可batch
    sum=0
    last_split=0
    for i in range(num_graph):
        batch_node = data[i].num_nodes
        c_num=math.ceil(batch_node*0.6)
        out_node = pred[last_split: last_split + batch_node]
        label_node = y[last_split: last_split + batch_node]
        out_rank = np.argsort(out_node, axis=0).reshape(-1)
        label_rank = np.argsort(label_node, axis=0).reshape(-1)
        correct = (out_rank[:c_num] == label_rank[:c_num])
        accuracy = int(correct.sum()) / correct.numel()
        sum += accuracy
        last_split = last_split + batch_node

    return sum/num_graph
def kendall_rank_coffecient(out, label, num_graph, data):
    sum = 0
    last_split = 0
    for i in range(num_graph):
        batch_node = data[i].num_nodes
        out_node = out[last_split: last_split + batch_node]
        label_node = label[last_split: last_split + batch_node]
      #  out_rank = np.argsort(out_node, axis=0).reshape(-1)
      #  label_rank = np.argsort(label_node, axis=0).reshape(-1)
        tau, p_value = stats.kendalltau(out_node, label_node)
        sum += tau
        last_split = last_split + batch_node

    return sum/num_graph
def set_correlation_coffecient(out,label):
    pred_set = set(np.where(out == 1)[0].tolist())
    label_set=set(np.where(label == 1)[0].tolist())
    intersection_set_len = len(pred_set & label_set)
    union_set_len = len(pred_set | label_set)
    # difference_set_len=len(set_1-set_2)

    correlation_coefficient = (intersection_set_len) / union_set_len
    return correlation_coefficient
def visualize_node_emb(fig_label,node_emb,node_mean,node_var,target):
    z = TSNE(n_components=1).fit_transform(node_emb)
    # mean=sigmoid(node_mean)
    # var=sigmoid(node_var+node_mean)-mean
    mean=node_mean
    var=node_var
    print("mean:{}".format(node_mean))
    print("var:{}".format(var))
    print("target:{}".format(target))

    fig=plt.figure("3D visualization of  "+fig_label)
   # ax=fig.add_subplot(111,projection="3d")
    # ax.scatter(z[:, 0], z[:, 1], mean.T,   c="k",marker='o', alpha=1, s=10)
    # ax.scatter(z[:, 0], z[:, 1], target.T, c="r", marker='o', alpha=1, s=10)
    # ax.scatter(z[:, 0], z[:, 1], mean.T,   c=mean,    marker='o', alpha=0.7, s=10*var)
    ax = fig.add_subplot(111)
    ax.scatter(z[:, 0], mean.T, c="k", marker='o', alpha=1, s=10)
    ax.scatter(z[:, 0], target.T, c="r", marker='o', alpha=1, s=10)
    ax.scatter(z[:, 0],  mean.T, c='b', marker='o', alpha=0.5, s=10+10 * var)

def save_checkpoint(path: str,model_name: str, model,net_param):
    """
    Saves a model checkpoint.
    """
    # Convert args to namespace for backwards compatibility

    model_state = {
        'net_param': net_param,
        'state_dict': model.state_dict(),
    }
    model_path=os.path.join(path,model_name)
    torch.save(model_state,model_path)


def load_diffusion_model(path: str, device, infer_para=None, **kwargs) :
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :return: The loaded model,loaded network parameters.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)

    loaded_net_param = state["net_param"]
    if infer_para is not None:
        loaded_net_param.update(infer_para)
    loaded_state_dict = state['state_dict']
    loaded_state_dict={k.replace('module.',''): v for k,v in loaded_state_dict.items() }	
    loaded_net_param["device"] = device

    model = diffusion_models(task_model=loaded_net_param["task_model"], net_param=loaded_net_param,train_model_select=kwargs["train_model_select"]).to(
        device)
    model.load_state_dict(loaded_state_dict,strict=False)
    model = model.to(device)

    return model,loaded_net_param



def save_config(path:str,configs_name="configs.yaml",dataset_param=None,net_param=None,train_param=None,optimizer_param=None,loss_param=None):
    train_state = {
        'dataset':dataset_param,
        'train': train_param,
        'net': net_param,
        'optimizer': optimizer_param,
        'loss': loss_param,
    }

    file_path=os.path.join(path,configs_name)


    if os.path.exists(file_path):

        with open(file_path, 'r') as f:
            saved_train_parameters=yaml.safe_load(f)

        if json.dumps(saved_train_parameters,sort_keys=True) ==json.dumps(train_state,sort_keys=True):
            trained_model_path = os.path.join(path, "hold_out/trained_model")
            #print(trained_model_path)
            if os.path.exists(trained_model_path):
                configs_name=configs_name.replace("config_", "")
                print("{} model has existed".format(configs_name))
                with open(os.path.join(path + "/hold_out/train_trace/record_scores.json"), "r") as f:
                    record_scores = yaml.safe_load(f)
                return False, record_scores
            else:
                return True,None
        else:
            with open(file_path, "w") as f:
                yaml.dump(train_state, f)
            return True,None
    else:
        with open(file_path, "w") as f:
            yaml.dump(train_state, f)
        return True,None

if __name__=="__main__":
    pass
