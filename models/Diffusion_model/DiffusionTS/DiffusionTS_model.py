from types import SimpleNamespace

import torch
import torch.nn as nn

from .DiffusionTS import Diffusion_TS


class DiffusionTS_model(nn.Module):
    def __init__(self, net_param):
        super(DiffusionTS_model, self).__init__()
        self.device = net_param["device"]
        self.dataset_nf = net_param["dataset_nf"]
        self.windows = net_param["windows"]
        self.pred_len = net_param["pred_len"]
        self.seq_len = net_param["seq_len"] = self.windows
        self.label_len = net_param["label_len"] = self.windows // 2
        self.n_z_samples = net_param.get("n_z_samples", 100)
        self.parallel_sample = net_param.get("parallel_sample", min(10, self.n_z_samples))
        self.sampling_timesteps = net_param.get("diffusion_steps", 100)
        self.scaler = net_param.get("scaler_type", None)
        self.configs = SimpleNamespace(**net_param)
        self.register_buffer("scaler_mean", torch.zeros(self.dataset_nf))
        self.register_buffer("scaler_std", torch.ones(self.dataset_nf))

        self.model = Diffusion_TS(
            seq_length=self.windows + self.pred_len,
            feature_size=self.dataset_nf,
            n_layer_enc=net_param.get("n_layer_enc", 3),
            n_layer_dec=net_param.get("n_layer_dec", 6),
            d_model=net_param.get("d_model", 64),
            timesteps=net_param.get("timesteps", 100),
            sampling_timesteps=self.sampling_timesteps,
            loss_type=net_param.get("loss_type", "l2"),
            beta_schedule=net_param.get("beta_schedule", "cosine"),
            n_heads=net_param.get("n_heads", 4),
            mlp_hidden_times=net_param.get("mlp_hidden_times", 4),
            eta=net_param.get("eta", 0.0),
            attn_pd=net_param.get("attn_pd", 0.0),
            resid_pd=net_param.get("resid_pd", 0.0),
            kernel_size=net_param.get("kernel_size"),
            padding_size=net_param.get("padding_size"),
            use_ff=net_param.get("use_ff", True),
            reg_weight=net_param.get("reg_weight"),
        ).to(self.device)

        gt_mask = torch.cat(
            [
                torch.ones(self.windows, self.dataset_nf, dtype=torch.bool),
                torch.zeros(self.pred_len, self.dataset_nf, dtype=torch.bool),
            ],
            dim=0,
        )
        self.register_buffer("gt_mask", gt_mask)

    def scaler_fit(self, data):
        data_std = data.std(axis=0)
        data_std[data_std == 0] = 1
        self.scaler_mean = data.mean(axis=0)
        self.scaler_std = data_std

    def scaler_transform(self, data):
        return (data - self.scaler_mean) / self.scaler_std

    def scaler_inverse_transform(self, data):
        return (data * self.scaler_std) + self.scaler_mean

    def training_step(self, batch):
        data = batch[:, : self.windows + self.pred_len, :].to(self.device)
        return self.model(data, target=data)

    def evaluation_step(self, batch):
        batch_x = batch[:, : self.windows, :].to(self.device)
        if batch.shape[1] - self.windows >= self.pred_len:
            batch_y = batch[:, self.windows : self.windows + self.pred_len, :].to(self.device)
        else:
            batch_y = None

        future_seed = torch.zeros(batch_x.shape[0], self.pred_len, self.dataset_nf, device=self.device)
        x = torch.cat([batch_x, future_seed], dim=1)
        partial_mask = self.gt_mask.expand(x.shape[0], -1, -1)
        parallel_sample = min(self.parallel_sample, self.n_z_samples)
        if self.n_z_samples % parallel_sample != 0:
            raise ValueError("n_z_samples must be divisible by parallel_sample")

        model_kwargs = {
            "coef": self.configs.__dict__.get("infill_coef", 1e-1),
            "learning_rate": self.configs.__dict__.get("infill_learning_rate", 5e-2),
        }
        samples = []
        for _ in range(self.n_z_samples // parallel_sample):
            repeat_x = x.repeat(parallel_sample, 1, 1)
            repeat_mask = partial_mask.repeat(parallel_sample, 1, 1)
            with torch.no_grad():
                sample = self.model.fast_sample_infill(
                    shape=repeat_x.shape,
                    target=repeat_x * repeat_mask,
                    partial_mask=repeat_mask,
                    sampling_timesteps=self.sampling_timesteps,
                    model_kwargs=model_kwargs,
                )
            sample = sample[:, -self.pred_len :, :].reshape(
                x.shape[0], parallel_sample, self.pred_len, self.dataset_nf
            )
            samples.append(sample.detach().cpu())

        preds = torch.cat(samples, dim=1)
        outs = preds.reshape(x.shape[0], self.n_z_samples, self.pred_len, self.dataset_nf).permute(0, 2, 3, 1)
        return outs, batch_y

