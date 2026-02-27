import torch_geometric.nn as gnn
import torch
import torch.nn as nn
import torch_geometric as tg
def gnn_pool(pool_name):
    if pool_name=="add":
        return gnn.global_add_pool
    elif pool_name=="max":
        return gnn.global_max_pool
    elif pool_name=="mean":
        return gnn.global_mean_pool
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")