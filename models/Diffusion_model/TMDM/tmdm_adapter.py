import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .TMDM import TMDM
from .tmdm_diffusion_utils import p_sample_loop, q_sample
from . import tmdm_ns_transformer as ns_Transformer


def log_normal(x, mu, var):
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


class TMDM_model(nn.Module):
    def __init__(self, net_param):
        super(TMDM_model, self).__init__()
        self.device = net_param["device"]
        self.dataset_nf = net_param["dataset_nf"]
        self.windows = net_param["windows"]
        self.pred_len = net_param["pred_len"]
        self.seq_len = net_param["seq_len"] = self.windows
        self.label_len = net_param["label_len"] = net_param.get("label_len", self.windows // 2)
        self.diffusion_steps =  net_param.get("diffusion_steps", 100)
        self.n_z_samples = net_param.get("n_z_samples", 100)
        self.parallel_sample = net_param.get("parallel_sample", min(10, self.n_z_samples))
        self.scaler = net_param.get("scaler_type", None)
        self.k_z = net_param.get("k_z", 0.01)

        net_param.setdefault("enc_in", self.dataset_nf)
        net_param.setdefault("dec_in", self.dataset_nf)
        net_param.setdefault("c_out", self.dataset_nf)
        net_param.setdefault("features", "M")
        net_param.setdefault("embed", "fixed")
        net_param.setdefault("freq", "h")
        net_param.setdefault("dropout", 0.05)
        net_param.setdefault("output_attention", False)
        net_param.setdefault("d_model", 64)
        net_param.setdefault("CART_input_x_embed_dim", net_param["d_model"])
        net_param.setdefault("factor", 3)
        net_param.setdefault("n_heads", 4)
        net_param.setdefault("d_ff", 128)
        net_param.setdefault("activation", "gelu")
        net_param.setdefault("e_layers", 2)
        net_param.setdefault("d_layers", 1)
        net_param.setdefault("p_hidden_dims", [64, 64])
        net_param.setdefault("p_hidden_layers", 2)
        net_param.setdefault("d_z", net_param["d_model"])
        net_param.setdefault("k_cond", 1.0)
        net_param.setdefault("timesteps", self.diffusion_steps)
        net_param.setdefault(
            "diffusion_config_dir",
            os.path.join(os.path.dirname(__file__), "tmdm.yml"),
        )

        self.configs = SimpleNamespace(**net_param)
        self.register_buffer("scaler_mean", torch.zeros(self.dataset_nf))
        self.register_buffer("scaler_std", torch.ones(self.dataset_nf))

        self.model = TMDM(self.configs, self.device).to(self.device)
        self.model.diffusion_config.testing.n_z_samples = self.n_z_samples
        self.model.diffusion_config.testing.n_z_samples_depart = 1
        self.cond_pred_model = ns_Transformer.Model(self.configs).float().to(self.device)

    def scaler_fit(self, data):
        data_std = data.std(axis=0)
        data_std[data_std == 0] = 1
        self.scaler_mean = data.mean(axis=0)
        self.scaler_std = data_std

    def scaler_transform(self, data):
        return (data - self.scaler_mean) / self.scaler_std

    def scaler_inverse_transform(self, data):
        return (data * self.scaler_std) + self.scaler_mean

    def _decoder_input(self, batch_x):
        dec_inp_pred = torch.zeros(batch_x.size(0), self.pred_len, self.dataset_nf, device=self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :]
        return torch.cat([dec_inp_label, dec_inp_pred], dim=1)

    def training_step(self, batch):
        batch_x = batch[:, : self.windows, :].to(self.device)
        target_y = batch[:, self.windows : self.windows + self.pred_len, :].to(self.device)
        batch_y = torch.cat([batch_x[:, -self.label_len :, :], target_y], dim=1)
        dec_inp = self._decoder_input(batch_x)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,), device=self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        _, y_0_hat_batch, kl_loss, _ = self.cond_pred_model(batch_x, None, dec_inp, None)
        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.ones(1, device=self.device))
        loss_vae_all = loss_vae + self.k_z * kl_loss

        noise = torch.randn_like(batch_y).to(self.device)
        y_t_batch = q_sample(
            batch_y,
            y_0_hat_batch,
            self.model.alphas_bar_sqrt,
            self.model.one_minus_alphas_bar_sqrt,
            t,
            noise=noise,
        )
        output = self.model(batch_x, None, batch_y, y_t_batch, y_0_hat_batch, t)
        return (noise - output).square().mean() + self.configs.k_cond * loss_vae_all

    def evaluation_step(self, batch):
        batch_x = batch[:, : self.windows, :].to(self.device)
        if batch.shape[1] - self.windows >= self.pred_len:
            batch_y = batch[:, self.windows : self.windows + self.pred_len, :].to(self.device)
        else:
            batch_y = None

        dec_inp = self._decoder_input(batch_x)
        _, y_0_hat_batch, _, _ = self.cond_pred_model(batch_x, None, dec_inp, None)
        parallel_sample = min(self.parallel_sample, self.n_z_samples)
        if self.n_z_samples % parallel_sample != 0:
            raise ValueError("n_z_samples must be divisible by parallel_sample")

        preds = []
        for _ in range(self.n_z_samples // parallel_sample):
            y_0_hat_tile = y_0_hat_batch.repeat(parallel_sample, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            y_T_mean_tile = y_0_hat_tile
            x_tile = batch_x.repeat(parallel_sample, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            with torch.no_grad():
                y_tile_seq = p_sample_loop(
                    self.model,
                    x_tile,
                    None,
                    y_0_hat_tile,
                    y_T_mean_tile,
                    self.model.num_timesteps,
                    self.model.alphas,
                    self.model.one_minus_alphas_bar_sqrt,
                )
            gen_y = y_tile_seq[self.model.num_timesteps].reshape(
                batch_x.shape[0], parallel_sample, self.label_len + self.pred_len, self.dataset_nf
            )
            preds.append(gen_y[:, :, -self.pred_len :, :].detach().cpu())

        preds = torch.cat(preds, dim=1)
        outs = preds.permute(0, 2, 3, 1)
        return outs, batch_y

