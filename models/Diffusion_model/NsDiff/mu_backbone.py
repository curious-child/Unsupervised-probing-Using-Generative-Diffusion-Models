import math

import torch
import torch.nn as nn
from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
from torch_timeseries.nn.embedding import DataEmbedding

from models.layer.gnn_conv import gnn_conv


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y

class SpatialBlock(nn.Module):
    def __init__(self, c_in, c_out,gnn_name, gnn_param ):
        super(SpatialBlock, self).__init__()
        self.gnn=gnn_conv(gnn_name=gnn_name,in_channels=c_in,out_channels=c_out,gnn_param=gnn_param,)

    def forward(self, x, edge_index):
        # x: [B*V,c_in]
        # edge_index: [2,E]

        return torch.relu(self.gnn(x,edge_index))# [B*V,c_out]
class Model(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout,time_embed=False)#without time_mark
        self.dec_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout,time_embed=False)#without time_mark
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.dataset_nf, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

        # self.z_mean = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        # self.z_logvar = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        #
        # self.z_out = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=1)
        return torch.mean(KL)
    


    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample
        if self.training:
            posterior_mean = posterior_mean.repeat(100, 1, 1, 1)
            posterior_var = posterior_var.repeat(100, 1, 1, 1)
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            z = z.mean(0)
        else:
            z = posterior_mean
        # z = posterior_mean
        return z

    def forward(self, x_enc,  x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #print("x_enc",x_enc.shape)
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x=x_enc, x_mark=None)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)


        

        dec_out = self.dec_embedding(x=x_dec_new, x_mark=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            # return dec_out[:, -self.pred_len:, :], dec_out, KL_z, z_sample  # [B, L, D]
            return dec_out[:, -self.pred_len:, :], dec_out  # [B, L, D]


class Model_spatial(nn.Module):
    """
    Non-stationary Transformer with spatial GNN
    """

    def __init__(self, configs):
        super(Model_spatial, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.T=configs.windows
        self.spatial_layers=configs.spatial_layers
        self.fT_h=configs.fT_h #默认 偶数
        self.d_model=configs.d_model

        # Embedding
        self.enc_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout, time_embed=False)  # without time_mark
        self.dec_embedding = DataEmbedding(configs.dataset_nf, configs.d_model,
                                           configs.dropout, time_embed=False)  # without time_mark
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.spatial_encoder=nn.ModuleList(
            [SpatialBlock(c_in=self.fT_h*configs.d_model,c_out=self.fT_h*configs.d_model,gnn_name=configs.f_gnn_name,gnn_param=configs.f_gnn_param)
             for l in range(self.spatial_layers)])

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.dataset_nf, bias=True)
        )

        self.tau_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len,
                                     hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.dataset_nf, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

        # W_in+2Pw-Kw=Sw*(W_out-1)+delta 0<=delta<=Sw-1
        down_kernel_size=self.T+1
        down_padding=self.fT_h//2 #d_T默认为偶数
        down_stride=1
        self.downsampling = nn.Conv2d(configs.d_model, configs.d_model, (1, down_kernel_size), (1, down_stride), (0, down_padding))#downsampling
        # W_out=(W_in-1)*Sw-2Pw+Kw+Ow
        up_kernel_size=self.T+1
        up_padding=self.fT_h//2 #d_T默认为偶数
        up_stride=1
        self.upsampling=nn.ConvTranspose2d(configs.d_model, configs.d_model, (1, up_kernel_size), (1, up_stride), (0, up_padding))
        # self.z_mean = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        # self.z_logvar = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )
        #
        # self.z_out = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)
        # )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=1)
        return torch.mean(KL)

    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample
        if self.training:
            posterior_mean = posterior_mean.repeat(100, 1, 1, 1)
            posterior_var = posterior_var.repeat(100, 1, 1, 1)
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            z = z.mean(0)
        else:
            z = posterior_mean
        # z = posterior_mean
        return z

    def forward(self, x_enc, x_dec,edge_index,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x=x_enc, x_mark=None)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
        #
        #downsampling
      #  print('enc_out',enc_out.shape)
        enc_out=enc_out.unsqueeze(2).transpose(1, 3)#(B*V, enc_out, 1,T_window)
        enc_out=self.downsampling(enc_out).transpose(1, 3).squeeze(2)#(B*V, fT_h, enc_out)
        #spatial encoding

        spatial_enc_out=enc_out.reshape(enc_out.shape[0],-1)#B*V,fT_h*enc_out
        for spatial_block in self.spatial_encoder:
            spatial_enc_out=spatial_block(spatial_enc_out,edge_index)
        enc_out=spatial_enc_out.reshape(enc_out.shape[0],self.fT_h,-1)#B*V, fT_h,enc_out
        #upsampling
        enc_out=enc_out.unsqueeze(2).transpose(1, 3)#(B*V, enc_out, 1,T_d)
        enc_out=self.upsampling(enc_out).transpose(1, 3).squeeze(2)#(B*V, T_window, enc_out)


        dec_out = self.dec_embedding(x=x_dec_new, x_mark=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            # return dec_out[:, -self.pred_len:, :], dec_out, KL_z, z_sample  # [B, L, D]
            return dec_out[:, -self.pred_len:, :], dec_out  # [B, L, D]