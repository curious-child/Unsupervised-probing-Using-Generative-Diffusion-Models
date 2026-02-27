# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import math
from models.layer.gnn_conv import gnn_conv


"""
Implementation of UGnet
Tcnblock: extract time feature
SpatialBlock: extract the spatial feature
"""

def TimeEmbedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class SpatialBlock(nn.Module):
    def __init__(self, c_in, c_out,gnn_name, gnn_param ):
        super(SpatialBlock, self).__init__()
        self.gnn=gnn_conv(gnn_name=gnn_name,in_channels=c_in,out_channels=c_out,gnn_param=gnn_param,)

    def forward(self, x, edge_index):
        # x: [B*V,c_in]
        # edge_index: [2,E]

        return torch.relu(self.gnn(x,edge_index))# [B*V,c_out]

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, droupout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))

        self.chomp = Chomp(self.padding)
        self.drop =  nn.Dropout(droupout)

        self.net = nn.Sequential(self.conv, self.chomp, self.drop)

        self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None


    def forward(self, x):
        # x: (B, C_in, V, T) -> (B, C_out, V, T)
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)

        return out + x_skip

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, T_in,net_param, kernel_size=3):
        """
        :param c_in: in channels
        :param c_out: out channels
        :param kernel_size:
        TCN convolution
            input: (B*V, c_in, 1, T)
            output:(B*V, c_out, 1, T)
        """
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1,1))
        self.t_conv = nn.Conv2d(net_param["d_h"], c_out, (1,1))
        self.T = T_in
        self.Td_h=net_param["Td_h"]
        #print("Td_h:",self.Td_h)
        # W_in+2Pw-Kw=Sw*(W_out-1)+delta 0<=delta<=Sw-1
        down_kernel_size = self.T + 1
        down_padding = self.Td_h // 2  # d_T默认为偶数
        down_stride = 1
        self.downsampling = nn.Conv2d(c_out, c_out, (1, down_kernel_size), (1, down_stride),
                                      (0, down_padding))  # downsampling
        # W_out=(W_in-1)*Sw-2Pw+Kw+Ow
        up_kernel_size = self.T + 1
        up_padding = self.Td_h // 2  # d_T默认为偶数
        up_stride = 1
        self.upsampling = nn.ConvTranspose2d(c_out, c_out, (1, up_kernel_size), (1, up_stride),
                                             (0, up_padding))
        self.spatial = SpatialBlock(c_in=self.Td_h*c_out, c_out=self.Td_h*c_out,gnn_name=net_param["gnn_name"],gnn_param=net_param["gnn_param"])

        self.norm = nn.LayerNorm([1, c_out])
    def forward(self, x, t, edge_index):
        # x: (B*V, c_in, 1, T), return (B*V, c_out, 1, T)

        h = self.tcn1(x)

        h += self.t_conv(t[:, :, None, None])

        h = self.tcn2(h)# (B*V, c_out, 1, T_i)

        h = self.norm(h.transpose(1,3)).transpose(1,3) # (B*V, c_out, 1, T_i)


        h=self.downsampling(h).transpose(1, 3).squeeze(2) # (B*V,T_h, c_out)
        spatial_h = h.reshape(h.shape[0], -1) # (B*V, T_h*c_out)
        spatial_h = self.spatial(spatial_h, edge_index) #  (B*V, T_h*c_out)
        spatial_h=spatial_h.reshape(spatial_h.shape[0],self.Td_h,-1)# (B*V, T_h,c_out)
        h = spatial_h.unsqueeze(2).transpose(1, 3)  # (B*V, c_out, 1,T_d)
        h = self.upsampling(h)  # (B*V, c_out, 1, T )

        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out,T_in, net_param):
        """
        :param c_in: in channels, out channels
        :param c_out:
        """
        super().__init__()
        self.res = ResidualBlock(c_in, c_out,T_in, net_param, kernel_size=3)

    def forward(self, x, t, edge_index):
        # x: (B*v, c_in, 1, T), return (B*v, c_out, 1, T)

        return self.res(x, t, edge_index)

class Downsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in,  kernel_size= (1,3), stride=(1,2), padding=(0,1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, supports):
        _ = t
        _ = supports
        return self.conv(x)#


class  UpBlock(nn.Module):
    def __init__(self, c_in, c_out, T_in,config):
        super().__init__()
        self.res = ResidualBlock(c_in + c_out, c_out, T_in,config, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)

class Upsample(nn.Module):
    def  __init__(self, c_in):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))

    def forward(self, x, t, supports):
        _ = t
        _ = supports
        return  self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, c_in,T_in, config):
        super().__init__()
        self.res1 = ResidualBlock(c_in, c_in,T_in, config, kernel_size=3)
        self.res2 = ResidualBlock(c_in, c_in,T_in, config, kernel_size=3)

    def forward(self, x, t, edge_index):
        x = self.res1(x, t, edge_index)

        x = self.res2(x, t, edge_index)

        return x


class UGnet(nn.Module):
    def __init__(self, net_param) -> None:
        super().__init__()
        self.net_param = net_param
        self.d_h = net_param["d_h"]
        self.T_p = net_param["T_p"]
        self.T_h = net_param["T_h"]
        self.T = self.T_p + self.T_h
        self.F = net_param["F"]
        self.channel_multipliers = net_param["channel_multipliers"]
        self.n_blocks = net_param['n_blocks']

        # number of resolutions
        n_resolutions = len(self.channel_multipliers)

        # first half of U-Net = decreasing resolution
        down = []
        # number of channels
        T_in=2*self.T
        out_channels = in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels *self.channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels,T_in, self.net_param))
                in_channels = out_channels

            # down sample at all resolution except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))#change time dimension
                T_in = math.floor((T_in-1)/2+1)

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels,T_in, self.net_param)

        # #### Second half of U-Net - increasing resolution
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks):
                up.append(UpBlock(in_channels, out_channels, T_in, self.net_param))

            out_channels = in_channels // self.channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels,T_in, self.net_param))
            in_channels = out_channels
            # up sample at all resolution except last
            if i > 0:
                up.append(Upsample(in_channels))
                T_in = T_in*2

        self.up = nn.ModuleList(up)
        assert T_in == 2*self.T, "T_in should be equal to 2*T"
        self.x_proj = nn.Conv2d(self.F, self.d_h, (1,1))
        self.out = nn.Sequential(nn.Conv2d(self.d_h, self.F, (1,1)),
                                 nn.Linear(2 * self.T, self.T),)
        # for gcn



    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        """
        :param x: x_t of current diffusion step, (B*V,T,F)
        :param t: diffsusion step (B*V)
        :param c: condition information
            used information in c:
                x_masked: (B*V, T, F)
        :return:
        """

        x_masked, edge_index, batch_index = c  # x_masked: (B*V,T,F), edge_index: (2, E), batch_index: (B*V)
        x = x.unsqueeze(2).transpose(1, 3)  # (B*V, F, 1,T)

       # print("x:",x.shape)
        x_masked = x_masked.unsqueeze(2).transpose(1, 3)  # (B*V, F, 1,T)
      #  print("x_masked:", x_masked.shape)
        x = torch.cat((x, x_masked), dim=-1) #(B*V, F, 1,2*T)


        x = self.x_proj(x)# (B*V, d_h,1, 2 * T)

        t = TimeEmbedding(t, self.d_h)#(B*V, d_h)

        h = [x]



        for m in self.down:
            x = m(x, t, edge_index)# (B*V, c_out, 1, Ti)#Ti=d_h
            h.append(x)

        x = self.middle(x, t, edge_index)

        for m in self.up:
            if isinstance(m,  Upsample):
                x = m(x, t, edge_index)
            else:
                s =h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x,t, edge_index)

        e = self.out(x)#(B*V, F,1,T)
        # print("e:",e.shape)
        return e.squeeze(2).transpose(1, 2) # (B*V, T,F)

