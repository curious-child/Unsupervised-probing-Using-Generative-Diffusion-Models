import torch.nn as nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing

def set_masks(mask: Tensor, model: nn.Module):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module._explain = True
                module._edge_mask = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._explain = False
            module._edge_mask= None