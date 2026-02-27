import functools

import torch.nn as nn
from torch_geometric.data import Batch

from models.Diffusion_model.NsDiff.NsDiff_net import NsDiff_net, NsDiff_net_spatial
import models.Diffusion_model.NsDiff.mu_backbone as ns_Transformer
import torch
import models.Diffusion_model.NsDiff.g_backbone as G
from models.Diffusion_model.NsDiff.nsdiff_utils import cal_forward_noise, cal_sigma_tilde, q_sample,p_sample_loop
from models.Diffusion_model.NsDiff.sigma import wv_sigma_trailing
from types import SimpleNamespace



class NsDiff_model(nn.Module):
    def __init__(self,
                 net_param,
                 train_model_select,
                 pretrain_f_path="results/pre_model_F",
                 pretrain_g_path="results/pre_model_G",
                ):
        super(NsDiff_model, self).__init__()



        self.device = net_param['device']
        self.dataset_nf = net_param['dataset_nf']
        self.windows = net_param['windows']
        self.pred_len = net_param['pred_len']
        self.rolling_length = net_param['rolling_length']
        self.load_pretrain = net_param['load_pretrain']
        self.seq_len=net_param["seq_len"]=self.windows
        self.label_len=net_param["label_len"]=self.windows//2
        self.diffusion_steps=net_param["diffusion_steps"]
        self.freeze_pretrain = net_param["freeze_pretrain"] if "freeze_pretrain" in net_param else False
        self.EPS = 10e-8
        self.configs = SimpleNamespace(**net_param)
        self.scaler = net_param['scaler_type']
        self.register_buffer("scaler_mean", torch.zeros(self.dataset_nf))
        self.register_buffer("scaler_std",torch.zeros(   self.dataset_nf))
        if train_model_select == "NsDiff_model":
            self.model = NsDiff_net(self.configs, self.device).to(self.device)
            
            if self.load_pretrain:
                model_f_path = pretrain_f_path + "/model_trained"
                model_g_path = pretrain_g_path + "/model_trained"
                print("using pretrained model...")
                print(f"f(x): {model_f_path}")
                print(f"g(x): {model_g_path}")
                pref_state=torch.load(model_f_path, map_location=self.device, weights_only=True)
                preg_state=torch.load(model_g_path, map_location=self.device, weights_only=True)
                pref_params=pref_state["net_param"]
                preg_params=preg_state["net_param"]
                pref_state_dict=pref_state["state_dict"]
                preg_state_dict=preg_state["state_dict"]
                self.cond_pred_model = ns_Transformer.Model(SimpleNamespace(**pref_params)).float().to(self.device)
                self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset_nf, 512,
                                                   preg_params["rolling_length"]).float().to(self.device)
                if torch.cuda.device_count() > 1:
                    self.cond_pred_model = nn.DataParallel(self.cond_pred_model)
                    self.cond_pred_model_g = nn.DataParallel(self.cond_pred_model_g)
                else:
                    pref_state_dict = {k.replace('module.', ''): v for k, v in pref_state_dict.items()}
                    preg_state_dict={k.replace('module.',''): v for k,v in preg_state_dict.items() }

                self.cond_pred_model.load_state_dict(pref_state_dict,strict=True)
                self.cond_pred_model_g.load_state_dict(preg_state_dict,  strict=True )

            else:
                self.cond_pred_model = ns_Transformer.Model(self.configs).float().to(self.device)
                self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset_nf, 512,
                                                       self.rolling_length).float().to(self.device)
                if torch.cuda.device_count() > 1:
                    self.cond_pred_model = nn.DataParallel(self.cond_pred_model)
                    self.cond_pred_model_g = nn.DataParallel(self.cond_pred_model_g)
            if self.freeze_pretrain:
                for param in self.cond_pred_model.parameters():
                    param.requires_grad = False
                for param in self.cond_pred_model_g.parameters():
                    param.requires_grad = False

        elif train_model_select == "pretrain_f":
            self.cond_pred_model = ns_Transformer.Model(self.configs).float().to(self.device)
            if torch.cuda.device_count() > 1:
                self.cond_pred_model = nn.DataParallel(self.cond_pred_model)
        elif train_model_select == "pretrain_g":
            self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset_nf, 512,
                                                       self.rolling_length).float().to(self.device)
            if torch.cuda.device_count() > 1:
                self.cond_pred_model_g = nn.DataParallel(self.cond_pred_model_g)
        else:
            raise ValueError("train_model_select should be in ['NsDiff_model', 'pretrain_f', 'pretrain_g']")
    def scaler_fit(self,data):

        data_std = data.std(axis=0)
        data_std[data_std == 0] = 1
        self.scaler_mean = data.mean(axis=0)
        self.scaler_std = data_std

    def scaler_transform(self, data):
        return (data - self.scaler_mean) / self.scaler_std

    def scaler_inverse_transform(self, data):
        return (data * self.scaler_std) + self.scaler_mean
    def pretrain_f(self,batch):
        batch_x=batch[:,:self.windows,:].to(self.device)#(B, T, N) T=windows  N=dataset_nf
        batch_y=batch[:,self.windows:,:].to(self.device)#(B, O, N) O=pred_len
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        y_0_hat_batch, _ = self.cond_pred_model(batch_x, dec_inp)
        loss = (y_0_hat_batch - batch_y).square().mean()#mse loss
        return loss
    def pretrain_g(self,batch):
        print("pretrain_g")
        batch_x=batch[:,:self.windows,:].to(self.device)#(B, T, N) T=windows  N=dataset_nf
        batch_y=batch[:,self.windows:,:].to(self.device)#(B, O, N) O=pred_len
        y_sigma = wv_sigma_trailing(torch.concat([batch_x, batch_y], dim=1), self.rolling_length)
        y_sigma = y_sigma[:, -self.pred_len:, :] + 10e-8
        gx = self.cond_pred_model_g(batch_x)
        loss=(torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()
        return loss
    def training_step(self,batch):
        batch_x=batch[:,:self.windows,:].to(self.device)#(B, T, N) T=windows  N=dataset_nf
        batch_y=batch[:,self.windows:,:].to(self.device)#(B, O, N) O=pred_len
        assert batch_y.size(1) == self.pred_len, "pred_len is not equal to the length of the prediction"
        y_sigma = wv_sigma_trailing(torch.concat([batch_x, batch_y], dim=1), self.rolling_length)  # [B, T+O, N]
        y_sigma = y_sigma[:, -self.pred_len:, :] + self.EPS  # truth sigma_y0 [B, O, N]


        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]
        y_0_hat_batch, _ = self.cond_pred_model(batch_x,  dec_inp)
        gx = self.cond_pred_model_g(batch_x) + self.EPS  # (B, O, N)
        if torch.isnan(gx).any():
        	print(torch.isnan(gx).all())
        	print(torch.isnan(batch_x).any())
        loss1 = (y_0_hat_batch - batch_y).square().mean()
        loss2 = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)

        forward_noise = cal_forward_noise(self.model.betas_tilde, self.model.betas_bar, gx, y_sigma, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(self.model.alphas, self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                      self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                      self.model.betas_tilde_m_1, self.model.betas_bar_m_1, gx, y_sigma, t)

        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt,
                             self.model.one_minus_alphas_bar_sqrt, t, noise=noise)

        output, sigma_theta = self.model( y_t_batch, y_0_hat_batch, gx, t)
        sigma_theta = sigma_theta + self.EPS

        kl_loss = ((e - output)).square().mean() + (sigma_tilde / sigma_theta).mean() - torch.log(
            sigma_tilde / sigma_theta).mean()
        loss = kl_loss + loss1 + loss2
        return loss
    def evaluation_step(self,batch):
       # print("batch",batch)
        batch_x = batch[:, :self.windows, :].to(self.device)  # (B, T, N) T=windows  N=dataset_nf
        if batch.shape[1]-self.windows >=self.pred_len   :
            batch_y = batch[:, self.windows:,:].to(self.device)  # (B, O, N) O=pred_len
            assert batch_y.size(1) == self.pred_len, "pred_len is not equal to the length of the prediction"
        else:
            batch_y = None


        b = batch_x.shape[0]
        gen_y_by_batch_list = [[] for _ in range(self.diffusion_steps + 1)]
        parallel_sample = self.configs.parallel_sample


        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        def store_gen_y_at_step_t(config, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.diffusion_steps - idx
            gen_y = y_tile_seq[idx].reshape(b,
                                            # int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                            parallel_sample,
                                            (config.pred_len),
                                            config.dataset_nf).cpu()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y.detach().cpu()
            else:
                gen_y_by_batch_list[current_t] = torch.concat([gen_y_by_batch_list[current_t], gen_y],
                                                              dim=0).detach().cpu()
            return gen_y



        y_0_hat_batch, _ = self.cond_pred_model(batch_x,  dec_inp)
        gx = self.cond_pred_model_g(batch_x)
       # print("gx",gx.shape)

        preds = []
        for i in range(self.configs.n_z_samples // parallel_sample):
            repeat_n = int(parallel_sample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            y_T_mean_tile = y_0_hat_tile



            gx_tile = gx.repeat(repeat_n, 1, 1, 1)
            gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            y_tile_seq = p_sample_loop(self.model,  y_0_hat_tile, gx_tile, y_T_mean_tile,
                                           self.model.num_timesteps,
                                           self.model.alphas, self.model.one_minus_alphas_bar_sqrt,
                                           self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                           self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                           self.model.betas_tilde, self.model.betas_bar,
                                           self.model.betas_tilde_m_1, self.model.betas_bar_m_1,
                                           )
            gen_y = store_gen_y_at_step_t(config=self.model.args,
                                          idx=self.model.num_timesteps, y_tile_seq=y_tile_seq)





            outputs = gen_y[:, :, -self.pred_len:, :]  # B, S, O, N

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred.detach().cpu())  # numberof_testbatch,  B, parallel_sample, O, N


        preds = torch.concat(preds, dim=1)#B,n_z_samples , O, N



        outs = preds.permute(0, 2, 3, 1)#B, O, N, n_z_samples
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (
        self.pred_len, self.dataset_nf, self.configs.n_z_samples)

        return outs, batch_y

class NsDiff_model_spatial(nn.Module):
    def __init__(self,
                 net_param,
                 train_model_select,
                 pretrain_f_path="results/pre_model_F",
                 pretrain_g_path="results/pre_model_G",
                ):
        super(NsDiff_model_spatial, self).__init__()
        self.scaler = net_param['scaler_type']
        self.register_buffer("scaler_mean", None)
        self.register_buffer("scaler_std", None)
        self.device = net_param['device']
        self.dataset_nf = net_param['dataset_nf']
        self.windows = net_param['windows']
        self.pred_len = net_param['pred_len']
        self.rolling_length = net_param['rolling_length']
       # print("rolling_length",self.rolling_length)
        self.diffusion_steps = net_param["diffusion_steps"]
        self.load_pretrain = net_param['load_pretrain']
        self.seq_len=net_param["seq_len"]=self.windows
        self.label_len=net_param["label_len"]=self.windows//2
        self.freeze_pretrain = net_param["freeze_pretrain"] if "freeze_pretrain" in net_param else False
        self.EPS = 10e-8
        self.configs = SimpleNamespace(**net_param)
        self.scaler = net_param['scaler_type']
        self.register_buffer("scaler_mean", torch.zeros(self.dataset_nf))
        self.register_buffer("scaler_std", torch.zeros(self.dataset_nf))
        self.train_model_select=train_model_select
        if self.train_model_select == "NsDiff_model":
            self.model = NsDiff_net_spatial(self.configs,self.device).to(self.device)
            
            if self.load_pretrain:
                model_f_path = pretrain_f_path + "/model_trained"
                model_g_path = pretrain_g_path + "/model_trained"
                print("using pretrained model...")
                print(f"f(x): {model_f_path}")
                print(f"g(x): {model_g_path}")
                pref_state=torch.load(model_f_path, map_location=self.device, weights_only=True)
                preg_state=torch.load(model_g_path, map_location=self.device, weights_only=True)
                pref_params=pref_state["net_param"]
                preg_params=preg_state["net_param"]
                pref_state_dict=pref_state["state_dict"]
                preg_state_dict=preg_state["state_dict"]
                self.cond_pred_model = ns_Transformer.Model_spatial(SimpleNamespace(**pref_params)).float().to(self.device)
                self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset_nf, 512,
                                                   preg_params["rolling_length"]).float().to(self.device)
              #  if torch.cuda.device_count() > 1:
                pref_state_dict = {k.replace('module.', ''): v for k, v in pref_state_dict.items()}
                preg_state_dict={k.replace('module.',''): v for k,v in preg_state_dict.items() }
                self.cond_pred_model.load_state_dict(pref_state_dict,strict=True)
                self.cond_pred_model_g.load_state_dict(preg_state_dict, strict=True)
            else:
                self.cond_pred_model = ns_Transformer.Model_spatial(self.configs).float().to(self.device)
                self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset_nf, 512,
                                                   self.rolling_length).float().to(self.device)
            if self.freeze_pretrain:
                for param in self.cond_pred_model.parameters():
                    param.requires_grad = False
                for param in self.cond_pred_model_g.parameters():
                    param.requires_grad = False
            # if torch.cuda.device_count() > 1:
            #
            #     self.cond_pred_model = nn.DataParallel(self.cond_pred_model)
            #     self.cond_pred_model_g = nn.DataParallel(self.cond_pred_model_g)
        elif self.train_model_select == "pretrain_f":
            self.cond_pred_model = ns_Transformer.Model_spatial(self.configs).to(self.device)
            # if torch.cuda.device_count() > 1:
            #     self.cond_pred_model = nn.DataParallel(self.cond_pred_model)
        elif self.train_model_select == "pretrain_g":
            self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset_nf, 512,
                                                       self.rolling_length).float().to(self.device)
            # if torch.cuda.device_count() > 1:
            #     self.cond_pred_model_g = nn.DataParallel(self.cond_pred_model_g)
        else:
            raise ValueError("train_model_select should be in ['NsDiff_model', 'pretrain_f', 'pretrain_g']")

    def forward(self, gdatalist):
        #dataparallel training
        gdata = Batch.from_data_list(gdatalist).to(self.device)
        if self.scaler == "StandardScaler":
            gdata.x = self.scaler_transform(gdata.x)
        gdata = gdata
        if self.train_model_select == "NsDiff_model":
            loss = self.training_step(gdata=gdata)  # diffusion training
        elif self.train_model_select == "pretrain_f":
            loss = self.pretrain_f(gdata=gdata)  # pretrain f
        elif self.train_model_select == "pretrain_g":
            loss = self.pretrain_g(gdata=gdata)  # pretrain g
        else:
            raise ValueError("train_model_select error")
        return loss
    def scaler_fit(self,data):

        data_std = data.std(axis=(0,1))
        data_std[data_std == 0] = 1
        self.scaler_mean = data.mean(axis=(0,1))
        self.scaler_std = data_std

    def scaler_transform(self, data):
        return (data - self.scaler_mean) / self.scaler_std

    def scaler_inverse_transform(self, data):
        return (data * self.scaler_std) + self.scaler_mean
    def pretrain_f(self,gdata):
        batch=gdata.x
        edge_index=gdata.edge_index
        batch_x=batch[:,:self.windows,:].to(self.device)#(V, T, N) T=windows  N=dataset_nf
        batch_y=batch[:,self.windows:,:].to(self.device)#(V, O, N) O=pred_len
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        y_0_hat_batch, _ = self.cond_pred_model(batch_x, dec_inp,edge_index)
        loss = (y_0_hat_batch - batch_y).square().mean()#mse loss
        #print("pref loss",loss)
        return loss
    def pretrain_g(self,gdata):

        batch = gdata.x
        batch_x=batch[:,:self.windows,:].to(self.device)#(V, T, N) T=windows  N=dataset_nf
        batch_y=batch[:,self.windows:,:].to(self.device)#(V, O, N) O=pred_len
        y_sigma = wv_sigma_trailing(torch.concat([batch_x, batch_y], dim=1), self.rolling_length)
        y_sigma = y_sigma[:, -self.pred_len:, :] + 10e-8
        gx = self.cond_pred_model_g(batch_x)
        loss=(torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()
        return loss
    def training_step(self,gdata):
        batch = gdata.x
        edge_index = gdata.edge_index
        batch_x=batch[:,:self.windows,:].to(self.device)#(V, T, N) T=windows  N=dataset_nf
        batch_y=batch[:,self.windows:,:].to(self.device)#(V, O, N) O=pred_len
        assert batch_y.size(1) == self.pred_len, "pred_len is not equal to the length of the prediction"
        y_sigma = wv_sigma_trailing(torch.concat([batch_x, batch_y], dim=1), self.rolling_length)  # [B, T+O, N]
        y_sigma = y_sigma[:, -self.pred_len:, :] + self.EPS  # truth sigma_y0 [B, O, N]


        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]
        y_0_hat_batch, _ = self.cond_pred_model(batch_x,  dec_inp,edge_index)
        gx = self.cond_pred_model_g(batch_x) + self.EPS  # (B, O, N)
        if torch.isnan(y_0_hat_batch).any():
                print("y_0_hat_batch")
                print(torch.isnan(y_0_hat_batch).all())
        	#print(torch.isnan(batch_x).any())
        if torch.isnan(gx).any():
                print("gx")
                print(torch.isnan(gx).all())
        	#print(torch.isnan(batch_x).any())

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)

        forward_noise = cal_forward_noise(self.model.betas_tilde, self.model.betas_bar, gx, y_sigma, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(self.model.alphas, self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                      self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                      self.model.betas_tilde_m_1, self.model.betas_bar_m_1, gx, y_sigma, t)
        if torch.isnan(sigma_tilde).any():
                print("sigma_tilde")
                print(torch.isnan(sigma_tilde).all())
        	#print(torch.isnan(batch_x).any())                              

        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt,
                             self.model.one_minus_alphas_bar_sqrt, t, noise=noise)

        output, sigma_theta = self.model( y_t_batch, y_0_hat_batch, gx, t,edge_index)
        if torch.isnan(output).any():
                print("y_t_batch")
                print(torch.isnan(y_t_batch).all())
                print(torch.isnan(batch_x).any())  
                print("output")
                print(torch.isnan(output).all())
                print("sigma_theta")
                print(torch.isnan(sigma_theta).all())
        	
        sigma_theta = sigma_theta + self.EPS

        kl_loss = ((e - output)).square().mean() + (sigma_tilde / sigma_theta).mean() - torch.log(
            sigma_tilde / sigma_theta).mean()
        if not self.freeze_pretrain:
            loss1 = (y_0_hat_batch - batch_y).square().mean()
            loss2 = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()
            loss = kl_loss + loss1 + loss2
        else:
            loss = kl_loss
        return loss
    def evaluation_step(self,gdata):
        num_nodes = gdata.num_nodes
        edge_index = gdata.edge_index
        edge_index = edge_index.to(self.device).reshape(2, -1)

        batch = gdata.x
        batch_x = batch[:, :self.windows, :].to(self.device)  # (B, T, N) T=windows  N=dataset_nf
        if batch.shape[1]-self.windows >=self.pred_len   :
            batch_y = batch[:, self.windows:,:].to(self.device)  # (B, O, N) O=pred_len
            assert batch_y.size(1) == self.pred_len, "pred_len is not equal to the length of the prediction"
        else:
            batch_y = None


        b = batch_x.shape[0]
        gen_y_by_batch_list = [[] for _ in range(self.diffusion_steps + 1)]
        parallel_sample = self.configs.parallel_sample


        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset_nf]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        def store_gen_y_at_step_t(config, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.diffusion_steps - idx
            gen_y = y_tile_seq[idx].reshape(b,
                                            # int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                            parallel_sample,
                                            (config.pred_len),
                                            config.dataset_nf).cpu()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y.detach().cpu()
            else:
                gen_y_by_batch_list[current_t] = torch.concat([gen_y_by_batch_list[current_t], gen_y],
                                                              dim=0).detach().cpu()
            return gen_y



        y_0_hat_batch, _ = self.cond_pred_model(batch_x,  dec_inp,edge_index)
        gx = self.cond_pred_model_g(batch_x)


        if parallel_sample > 1:
            parellel_edge_index = self.duplicate_edge_index(parallel_sample, edge_index, num_nodes, self.device)
        self.model.set_edge_index(parellel_edge_index)
        preds = []
        for i in range(self.configs.n_z_samples // parallel_sample):
            repeat_n = int(parallel_sample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            y_T_mean_tile = y_0_hat_tile



            gx_tile = gx.repeat(repeat_n, 1, 1, 1)
            gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            y_tile_seq = p_sample_loop(self.model,  y_0_hat_tile, gx_tile, y_T_mean_tile,
                                           self.model.num_timesteps,
                                           self.model.alphas, self.model.one_minus_alphas_bar_sqrt,
                                           self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                           self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                           self.model.betas_tilde, self.model.betas_bar,
                                           self.model.betas_tilde_m_1, self.model.betas_bar_m_1
                                           )
            gen_y = store_gen_y_at_step_t(config=self.model.args,
                                          idx=self.model.num_timesteps, y_tile_seq=y_tile_seq)





            outputs = gen_y[:, :, -self.pred_len:, :]  # B, S, O, N

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred.detach().cpu())  # numberof_testbatch,  B, parallel_sample, O, N


        preds = torch.concat(preds, dim=1)#B,n_z_samples , O, N



        outs = preds.permute(0, 2, 3, 1)#B, O, N, n_z_samples
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (
        self.pred_len, self.dataset_nf, self.configs.n_z_samples)

        return outs, batch_y

    def duplicate_edge_index(self,parallel_sampling, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        # print("edge_index",edge_index.device)
        # print("edge_index_indent", edge_index_indent.device)
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index
