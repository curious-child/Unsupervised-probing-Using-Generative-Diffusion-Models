import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch





class CEPLoss(nn.Module):
    def __init__(self, diffusion_type,diffusion_params,**kwargs):
        """

        """
        super().__init__()
        if diffusion_type == 'GaussianDiffusion':
            from.diffusion_schedulers import GaussianDiffusion
            self.diffusion = GaussianDiffusion(**diffusion_params)
        elif diffusion_type == 'CategoricalDiffusion':
            from.diffusion_schedulers import CategoricalDiffusion
            self.diffusion = CategoricalDiffusion(**diffusion_params)
        elif diffusion_type == 'DiscreteFlowDiffusion':
            from.diffusion_schedulers import DiscreteFlowDiffusion
            self.diffusion = DiscreteFlowDiffusion(**diffusion_params)
        else:
            raise ValueError('Invalid diffusion type')




    def forward(self, model, batch):
        """
        前向计算（支持图结构数据）
        参数：
            model (nn.Module): 图神经网络模型，需支持时间步输入
            batch (Batch): 包含多个图的批数据，x输入范围为[0,1]，y为标签
            energy_fn (callable): 能量计算函数，输入为Batch对象
        """
        device = batch.x.device#
        noisy_batch=batch.clone()
        B = batch.num_graphs
        ptr = batch.ptr
        point_indicator = ptr[1:] - ptr[:-1]
        # 1. 时间步采样
        t = torch.randint(1, self.diffusion.T+1,  (B,), device=device)
        t = t.repeat_interleave(point_indicator.cpu().reshape(-1), dim=0).numpy()
        # 2. 根据不同扩散过程类型生成噪声图数据，其x的范围也不相同
        noisy_batch.x = self.diffusion.sample(batch, t).reshape( -1).to(device)  #，

        # 3. 计算能量
        original_energy = batch.y.clone().detach().reshape(-1)  # [B]

        # 4. 模型预测能量
        t=torch.from_numpy(t).to(device)
        t=t.reshape(-1)
        pred_energy = model(noisy_batch.float(), t.float().to(device))  # 模型需处理时间步信息 [B]

        # 5. 构造对比矩阵
        energy_matrix = original_energy[None, :].expand(B, -1)  # [B, B]
        pred_matrix = pred_energy[None, :].expand(B, -1)  # [B, B]

        # 6. 损失计算
        weights = torch.softmax(- energy_matrix, dim=1)
        log_probs = torch.log_softmax(-pred_matrix, dim=1)
        loss = -torch.sum(weights * log_probs) / B

        return loss