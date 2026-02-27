import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import Batch

from models.Diffusion_model.DiffSTG.ugnet import UGnet
from models.Diffusion_model.DiffSTG.diffusion_schedulers import GaussianDiffusion, InferenceSchedule


class graph_Diffusion_model(nn.Module):
    def __init__(self,
                 net_param,
                ):
        super(graph_Diffusion_model, self).__init__()

        self.diffusion_steps = net_param["diffusion_steps"]
        self.inference_diffusion_steps = net_param["inference_diffusion_steps"]
        self.inference_trick = net_param["inference_trick"] if net_param.get("inference_trick") else "ddim"
        self.device = net_param["device"]
        self.diffusion_schedule = net_param["diffusion_schedule"]


        self.inference_schedule=net_param["inference_schedule"]

        self.loss_weight_schedule=net_param["loss_weight_schedule"]
        self.parallel_sampling = net_param["parallel_sampling"]
        self.sequential_sampling = net_param["sequential_sampling"]
        self.sparse = True


        self.diffusion = GaussianDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule,
            loss_weight_schedule=self.loss_weight_schedule)









    def gaussian_posterior(self, target_t, t, pred, xt):
        """Sample (or deterministically denoise) from the Gaussian posterior for a given time step.
           See https://arxiv.org/pdf/2010.02502.pdf for details.
        """
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = target_t

        atbar = diffusion.alphabar[t]
        atbar_target = diffusion.alphabar[target_t]

        if self.inference_trick =="ddpm"or t <= 1:
            # Use DDPM posterior
            at = diffusion.alpha[t]
            z = torch.randn_like(xt)
            atbar_prev = diffusion.alphabar[t - 1]
            beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)

            xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
            xt_target = xt_target + np.sqrt(beta_tilde).item() * z
        elif self.inference_trick == 'ddim':
            xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
            xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
        else:
            raise ValueError('Unknown inference trick {}'.format(self.args.inference_trick))
        return xt_target.detach()



    def duplicate_edge_index(self,parallel_sampling, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index

    def duplicate_batch_index(self,parallel_sampling, batch_index, num_graphs, num_nodes, device):
        """Duplicate the batch index (in batch-based graphs) for parallel sampling."""
        if batch_index is not None:
            batch_index = batch_index.reshape((1, 1, -1))

            batch_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)

            batch_index_indent = batch_index_indent * num_graphs
            batch_index = batch_index + batch_index_indent
            batch_index = batch_index.reshape(-1)
        else:
            batch_index = torch.arange(0, parallel_sampling).view(-1).to(device)
            batch_index = batch_index.repeat_interleave(repeats=num_nodes)

        return batch_index.to(device)


class DiffSTG(graph_Diffusion_model):
    def __init__(self,
                 net_param,
                ):
        super(DiffSTG, self).__init__(net_param=net_param)
        self.device = net_param["device"]
        self.T_p = net_param["T_p"]#
        self.T_h = net_param["T_h"]
        self.T = self.T_p + self.T_h
        self.F = net_param["F"]
        self.mask_ratio = net_param["mask_ratio"]
        self.model = UGnet(net_param=net_param).to(self.device)
        self.scaler = net_param['scaler_type']
        self.register_buffer("scaler_mean", torch.zeros(self.F))
        self.register_buffer("scaler_std", torch.zeros(self.F))
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

    def forward(self, gdatalist):
        gdata = Batch.from_data_list(gdatalist).to(self.device)
        if self.scaler == "StandardScaler":
            gdata.x = self.scaler_transform(gdata.x)
        gdata = gdata
        return self.training_step(gdata=gdata)
    def scaler_fit(self,data):

        data_std = data.std(axis=(0,1))
        data_std[data_std == 0] = 1
        self.scaler_mean = data.mean(axis=(0,1))
        self.scaler_std = data_std

    def scaler_transform(self, data):
        # print("scaler mean:{}".format(self.scaler_mean))
        # print("scaler std:{}".format(self.scaler_std))
        return (data - self.scaler_mean) / self.scaler_std

    def scaler_inverse_transform(self, data):
        return (data * self.scaler_std) + self.scaler_mean
    def training_step(self, gdata):
        #

        ptr = gdata.ptr
        point_indicator = ptr[1:] - ptr[:-1]
        edge_index = gdata.edge_index
        batch_index = gdata.batch
        history = gdata.x[:, :self.T_h, :].to(self.device)  # [B*V,T_h,F]
        future = gdata.x[:,self.T_h:,:].to(self.device) #[B*V,T_p,F]
        assert future.size(1) == self.T_p, "pred_len is not equal to the length of the prediction"

        # get x0
        x0=torch.cat([history,future],dim=1)#[B*V,T,F]

        #get x0_masked
        mask = torch.randint_like(history, low=0, high=100) < int(
            self.mask_ratio * 100)  # mask the history in a ratio with mask_ratio
        history[mask] = 0
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(self.device) #[B*V,T,F]



        timesteps = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        timesteps = torch.from_numpy(timesteps).long()  # [B]
        t = timesteps.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()#[B*V]
        loss_weight = self.diffusion.get_loss_weights(t).reshape(-1).to(self.device)#[B*V]

        xt, epsilon = self.diffusion.sample(x0, t)#[B*V,T,F]
        t = torch.from_numpy(t).float()
        edge_index = edge_index.to(self.device).reshape(2, -1)


        # Denoise
        epsilon_pred = self.model(
            xt.float().to(self.device),
            t.float().to(self.device),
            (x_masked,edge_index, batch_index)

        )#[B*V,T,F]
        # print("epsilon_pred shape",epsilon_pred.shape)
        # print("epsilon shape",epsilon.shape)
        loss = F.mse_loss(epsilon_pred, epsilon.float(),reduction='none')#[B*V,T,F]
        loss_time=loss.mean(dim=-1)#[B*V,T]
        loss_node=loss_time.mean(dim=-1)#[B*V]

        weight_loss = loss_node * loss_weight

        all_loss = weight_loss.mean()


        return all_loss

    def gaussian_denoise_step(self,x_masked, xt, t, device, edge_index=None, target_t=None, batch_index=None):
        with torch.no_grad():
            epsilon_pred = self.model(
                xt.float().to(device),
                t.float().to(device),
                (x_masked, edge_index, batch_index)
            )##[B*V,T,F]

            xt = self.gaussian_posterior(target_t, t, epsilon_pred, xt)#[B*V,T,F]
            return xt.detach().to(device)

    def evaluation_step(self, data):
        if isinstance(data, Batch):
            ptr = data.ptr
            point_indicator = ptr[1:] - ptr[:-1]
            batch_index = data.batch
            num_graphs = point_indicator.shape[0]
        else:
            batch_index = None
            num_graphs = 1
        history = data.x[:, :self.T_h, :].to(self.device)  # [B*V,T_h,F]
        if data.x.shape[1]-self.T_h >=self.T_p :
            future = data.x[:, self.T_h:,:].to(self.device)  # (B, O, N) O=pred_len
            assert future.size(1) == self.T_p, "pred_len is not equal to the length of the prediction"
        else:
            future = None
        if future is not None:
        # get x0
            x0_truth = torch.cat([history, future], dim=1)  # [B*V,T,F]
            # get x0_masked

            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(self.device)  # [B*V,T,F]
        else:
            x0_truth=None
            x_masked=torch.cat((history, torch.zeros(history.shape[0],self.T_p,history.shape[2]).to(self.device)  ), dim=1).to(self.device)  # [B*V,T,F]


       # print("x_masked shape",x_masked.shape)
        graph_data = copy.deepcopy(data)
        edge_index = graph_data.edge_index
        edge_index = edge_index.to(self.device).reshape(2, -1)
        num_nodes = data.num_nodes

        stacked_predict_labels = []

        if self.parallel_sampling > 1:

            edge_index = self.duplicate_edge_index(self.parallel_sampling, edge_index, num_nodes, self.device)
            batch_index = self.duplicate_batch_index(self.parallel_sampling, batch_index,
                                                     num_graphs=num_graphs,
                                                     num_nodes=num_nodes,
                                                     device=self.device)
            x_masked=x_masked.repeat(self.parallel_sampling,1,1)




        with torch.no_grad():
            for _ in range(self.sequential_sampling):

                xt=torch.randn_like(x_masked).to(self.device)#[B*V,T,F]
               # print("xt shape",xt.shape)

                batch_size = 1
                steps = self.inference_diffusion_steps
                time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                  T=self.diffusion.T, inference_T=steps)

                for i in range(steps):
                    t1, t2 = time_schedule(i)
                    t1 = torch.tensor([t1 for _ in range(batch_size)]).int()
                    t2 = torch.tensor([t2 for _ in range(batch_size)]).int()


                    xt = self.gaussian_denoise_step(x_masked,xt, t1, self.device, edge_index, target_t=t2,
                                                    batch_index=batch_index)


                predict_labels = xt.float().cpu().detach()#[parallel_sampling*B*V,T,1]

                stacked_predict_labels.append(predict_labels)
            predict_labels = torch.cat(stacked_predict_labels, dim=0)#[all_sampling*B*V,T,1]
            all_sampling = self.sequential_sampling * self.parallel_sampling

            splitted_predict_x0 = predict_labels.reshape(all_sampling, -1, self.T, 1)#[all_sampling,B*V,T,1]
            splitted_predict_x0=splitted_predict_x0.permute(1,2,3,0) #[B*V,T,1,all_sampling]



        return splitted_predict_x0, x0_truth