import os

import igraph
import numpy as np
from igraph import *
import networkx as nx
import random
import math
def gen_graph_igraph(g_type, num_node,pl_edges=None,p_er=0.4,p_sw=0.4,connect_sw=2,edge_ba=4,exp=2.55,k_regular=4):

    cur_n = num_node

    if g_type == 'erdos_renyi':
        g = Graph.Erdos_Renyi(n=cur_n,p=p_er,directed=False)

    elif g_type == 'small-world':
        g = Graph.Watts_Strogatz(dim=1,size=cur_n,nei=connect_sw,p=p_sw)
    elif g_type == 'barabasi_albert':
        g = Graph.Barabasi(n=cur_n, m=edge_ba,directed=False)
    elif g_type=="static_power_law":
        g = igraph.GraphBase.Static_Power_Law(n=cur_n,m=pl_edges,exponent_out=exp)
    elif g_type == "K_Regular":
        g = igraph.GraphBase.K_Regular(n=cur_n, k=k_regular)

    return g

def gen_graphset_igraph(num_graph,g_type,num_node,directory,pl_edges=None,p_er=0.4,p_sw=0.4,connect_sw=2,edge_ba=4,exp=2.5,k_regular=4):
    for i in range(num_graph):
        G=gen_graph_igraph(g_type, num_node,pl_edges=pl_edges,p_er=p_er,p_sw=p_sw,connect_sw=connect_sw,edge_ba=edge_ba,exp=exp,k_regular=k_regular)
        G.write_graphml(directory+str(i)+".graphml")
def gen_graph_nx(g_type, num_node,m=5,n=5,m1=5,m2=15,r=2,h=4,n_hyper=4):

    if g_type=="cycle_graph":
        g=nx.cycle_graph(n=num_node)
    elif g_type=="grid_graph":
        g=nx.grid_2d_graph(m=m,n=n)
    elif g_type=="path_graph":
        g=nx.path_graph(n=num_node)
    elif g_type=="barbell_graph":
        g=nx.barbell_graph(m1=m1,m2=m2)
    elif g_type=="wheel_graph":
        g=nx.wheel_graph(n=num_node)
    elif g_type=="ladder_graph":
        g=nx.ladder_graph(n=num_node)
    elif g_type=="balanced_tree":
        g=nx.balanced_tree(r=r,h=h)
    elif g_type=="hypercube_graph":
        g=nx.hypercube_graph(n=n_hyper)

    return g
def gen_graphset_nx(g_type,num_node,directory,num_graph=0,m=5,n=5,m1=5,m2=15,r=2,h=4,n_hyper=4):

    G=gen_graph_nx(g_type, num_node,m=m,n=n,m1=m1,m2=m2,r=r,h=h,n_hyper=n_hyper)
    nx.write_graphml(G,directory+".graphml")
def gen_graph(g_type, cur_n=None, num_min=20, num_max=40):
    if cur_n is None:
        max_n = num_max
        min_n = num_min
        cur_n = np.random.randint(max_n - min_n + 1) + min_n

    if g_type == 'erdos_renyi':
        while True:
            ER_p=random.uniform(0.1,0.9)
            g = Graph.Erdos_Renyi(n=cur_n,p=ER_p,directed=False)
            if g.is_connected(mode='weak'):
                break
    elif g_type == 'small-world':
        while True:
            nei_max=round(0.35*cur_n)
            nei_min=round(0.15*cur_n)
            WS_nei=random.randint(nei_min,nei_max)
            WS_p = random.uniform(0, 0.15)
            g = Graph.Watts_Strogatz(dim=1,size=cur_n,nei=WS_nei,p=WS_p)#可包含最近邻规则图（p=0）
            if g.is_connected(mode='weak'):
                break
    elif g_type == 'barabasi_albert':
        while True:
            m_max = round(0.25 * cur_n)
            m_min = round(0.1 * cur_n)
            BA_m = random.randint(m_min, m_max)
            g = Graph.Barabasi(n=cur_n, m=BA_m,directed=False)
            if g.is_connected(mode='weak'):

                break
    elif g_type=="static_power_law":
        while True:
            exp = random.uniform(2, 3)
            exp_in=random.uniform(2,3)
            pl_max = round(0.25*cur_n* cur_n)
            pl_min = round(0.05*cur_n* cur_n)
            pl_edges=random.randint(pl_min,pl_max)
            g = Graph.Static_Power_Law(n=cur_n,m=pl_edges,exponent_out=exp,exponent_in=exp_in)
            if g.is_connected(mode='weak'):
                break
    elif g_type == "K_Regular":
        while True:
            k_regular=random.randint(round(0.2*cur_n),cur_n-2)  #可包含完全图（cur-1）
            if k_regular*cur_n %2==0 and cur_n>=(k_regular+1) : #k*n必须为偶数 且k+1 ≤n
                g = Graph.K_Regular(n=cur_n, k=k_regular)
                if g.is_connected():
                    break

    return g
if __name__ =="__main__":
    # for num_noode in [25,50,100,1000]:
    #     num_edges=math.ceil(2.5*num_noode)
    #     for edge_ba in [2,4,6]:
    #         gen_graphset_igraph(num_graph=3,g_type='barabasi_albert',num_node=num_noode,directory="target_dataset/barabasi_n_{}_m_{}_".format(num_noode,edge_ba),edge_ba=edge_ba)
    # for num_node in [100]:
    #     for connect_sw in [2,3,4]:
    #         for p_sw in [0.1,0.5,0.8]:
    #             gen_graphset_igraph(num_graph=3,g_type='small-world',num_node=num_node,directory="test2/ws_small-world_n_{}_hop_{}_p_{}_".format(num_node,connect_sw,p_sw),
    #                                 connect_sw=connect_sw ,p_sw=p_sw)
    g_type_list=["erdos_renyi","barabasi_albert","small-world"]#,"barabasi_albert","erdos_renyi"
    directory="train_dataset_graph"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for g_type in g_type_list:
        for num_node in [30,50,70,100]:
            for i in range(0,10):
                G=gen_graph(g_type=g_type, cur_n=num_node)

                G.write_graphml(directory+"/"+g_type+"_{}".format(num_node)+"_"+str(i)+".graphml")

# for r in [2,3,4]:
#     for h in [3,4]:
#         num_node=None
#         gen_graphset_nx(num_graph=1,g_type='balanced_tree',num_node=num_node,directory="test_dataset/balanced_tree_r_{}_h_{}".format(r,h),r=r,h=h)

# for num_node in [3,4,5]:
#     gen_graphset_nx(num_graph=1, g_type='hypercube_graph', num_node=num_node,
#                     directory="test_dataset/hypercube_graph_n_{}".format(num_node),n_hyper=num_node)