import os
from glob import glob

import networkx as nx
import torch
import torch_geometric
import yaml
from tqdm import tqdm

from utils.utils import load_diffusion_model

def torch_data_preprocessing(time_data,sampling_t,return_numpy=False):
    sampling_t_min = 0.1

    sampling_interval = int(sampling_t / sampling_t_min)


    if return_numpy:
        sampling_torch_time_series = time_data[::sampling_interval]  #
        return sampling_torch_time_series.cpu().detach().numpy()
    else:
        sampling_torch_time_series = time_data[:,::sampling_interval,:]  #
        return sampling_torch_time_series

def uncertainty_ews(model_save_file,torch_time_series,gnn_g,
                    time_data,save_ews_files_path,
                    infer_params=None,sample_window_step=50,sampling_t=None):

    if infer_params is None:
        infer_params = {  "parallel_sample": 50,  # parallel sampling number
                            "n_z_samples": 100}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    method_config_path=model_save_file+"/model_trained.yaml"
    with open(method_config_path,'r') as f:
        method_config_param=yaml.safe_load(f)
    model_path=model_save_file+"/model_trained"
    model,loaded_net_param = load_diffusion_model(model_path,device=device,infer_para=infer_params,
                                                  train_model_select=method_config_param["train"]["train_model_select"]
                                                  if "train_model_select" in method_config_param["train"].keys() else None)
    model.eval()
    if sampling_t is None:
        sampling_t = method_config_param["dataset"]['sampling_t']
    #sampling_t=1
  #  print("sampling_t:{}".format(sampling_t))
    sampling_torch_time_series = torch_data_preprocessing(torch_time_series,sampling_t=sampling_t)
  #  print("sampling_torch_time_series shape:{}".format(sampling_torch_time_series.shape))
    time_points = torch_data_preprocessing(time_data,sampling_t=sampling_t,return_numpy=True)
  #  print("time_points shape:{}".format(time_points.shape))


    rolling_window = method_config_param["dataset"]['windows']
    pred_times_len = method_config_param["dataset"]['pred_len']
   # scaler_type = method_config_param["dataset"]['scaler_type']
    timeseries_data = sampling_torch_time_series.unfold(1, rolling_window, sample_window_step)  # [Node_num,n,F,windows_len]
  #  print("timeseries_data shape:{}".format(sampling_torch_time_series.shape))
    pred_data=sampling_torch_time_series[:,rolling_window:,:]
   # print("pred_data shape:{}".format(pred_data.shape))
    pred_data=pred_data.unfold(1,pred_times_len,sample_window_step)
    timeseries_data = timeseries_data.permute(0,1, 3, 2)  # [Node_num,n,windows_len,F]
    pred_data = pred_data.permute(0,1, 3, 2)  # [Node_num,n,pred_times_len,F]
    time_points = time_points[rolling_window-1::sample_window_step]
   # print("timeseries_data shape:{}".format(time_points.shape))
    timeseries_datas = timeseries_data.unbind(1)
    pred_datas = pred_data.unbind(1)

    # if scaler_type ==  "StandardScaler":
    #     from torch_timeseries.scaler.standard import StandardScaler
    #     scaler = StandardScaler()
    #     scaler.fit(sampling_torch_time_series)
    # elif scaler_type == "None":
    #     scaler=None
    # else:
    #     raise ValueError("scaler_type should be StandardScaler or None")

    #save_path="results"
    data_save_path=save_ews_files_path

    if not os.path.exists(data_save_path):
        data_save_list = []
    else:
        data_save_list = torch.load(data_save_path)

    uncertainty_ews_list=[]
    if not data_save_list:
        with torch.no_grad():
           # for time_series,pred_data in tqdm(zip(timeseries_datas,pred_datas),leave=False):
            for time_series in tqdm(timeseries_datas, leave=False):
                graph_data_copy = gnn_g.clone()
                if model.scaler is not None:
                    time_series_trans = model.scaler_transform(time_series.to(device))
                   # pred_data_trans = model.scaler_transform(pred_data.to(device)).to("cpu")

              #  graph_data_copy.x = time_series_trans.clone().to(device)  # [node_num,pred_times_len,1]

               # print("time series shape",time_series.shape)
                pred_future,_ = model.evaluation_step(time_series_trans)##node_num,pred_times_len,1,n_z_samples
                pred_future=pred_future.squeeze(-2) #node_num,pred_times_len,n_z_samples
                pred_future = pred_future[:, -pred_times_len:, :]
                # print("pred_future device:{}".format(pred_future.device))
                # pred_error = pred_future[0, :, :].mean(dim=-1) - pred_data_trans[0, :, 0]
                # plt.plot(pred_error, color='r', label='pred_error')
                # plt.plot(pred_future[0, :, :].mean(dim=-1), color='g', label='pred_future')
                # plt.plot(pred_data_trans[0, :, 0], color='b', label='pred_data')
                # plt.legend()
                # plt.show()
                data_save_list.append(pred_future)


              #  print("pred_future shape:{}".format(pred_future.shape))
                # if scaler is not None:
                #     pred_future = scaler.inverse_transform(pred_future)
                pred_uncertainty = pred_future.var(dim=-1)# node_num,pred_times_len
                pred_uncertainty_nodes = pred_uncertainty.mean(dim=1) #node_num
                pred_uncertainty_mean = pred_uncertainty_nodes.mean()
                pred_uncertainty = pred_uncertainty_mean.cpu().detach().numpy()
                uncertainty_ews_list.append(pred_uncertainty)
        torch.save(data_save_list, data_save_path)
    else:
        #for pred_future,pred_data in tqdm(zip(data_save_list,pred_datas)):
            # if model.scaler is not None:
            #     pred_future = model.scaler_inverse_transform(pred_future.to(device)).to("cpu")
            # pred_error = pred_future.mean(dim=-1) - pred_data[:,:,0]#[nodes,pred_len]
            # pred_error = torch.abs( pred_error) # node_num,pred_times_len
            # pred_error_mean = pred_error.mean()
            # uncertainty_ews_list.append(pred_error_mean.cpu().detach().numpy())


           # pred_uncertainty = torch.abs((pred_future.max(dim=2).values+pred_future.min(dim=2).values)/2-pred_future.mean(dim=2))  # node_num,pred_times_len
          #  print("pred_uncertainty shape:{}".format(pred_uncertainty.shape))
        for pred_future in tqdm(data_save_list):
            dy=torch.zeros_like(pred_future)
            dy[:,:-1,:] = pred_future[:,1:,:] - pred_future[:,:-1,:]
            dy[:,-1,:]=dy[:,-2,:]

            dy=torch.abs(dy)
            dy=dy.sum(dim=1)
            pred_uncertainty = dy.var(dim=-1)  # node_num,pred_ti
            #pred_uncertainty = pred_future.var( dim=2) # node_num,pred_times_len
           #pred_uncertainty_nodes = pred_uncertainty.mean(dim=1)  # node_num
            pred_uncertainty_mean = pred_uncertainty.mean()
            pred_uncertainty = pred_uncertainty_mean.cpu().detach().numpy()
            uncertainty_ews_list.append(pred_uncertainty)
    return uncertainty_ews_list,time_points

def plot_dynamic_trajectory(ax,ts, ys,sampling_interval,dynamic_type):
    print("plot_dynamic_trajectory")

    
    ys=ys.mean(dim=1)
    
    # 1. 状态变量X随时间变化

    ax.plot(ts[::sampling_interval], ys[::sampling_interval], 'b-', label='dynamic: {}'.format(dynamic_type), linewidth=2)
    #ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,1,2,0"
    data_trends = ["decrease","increase"]
    model_types=["neuronal"]#"neuronal","SIS",""biomass"
    dataset_types = ["neuronal"]
   # model_type="NsDiff" 

    model_config="dataset__w100p100"
    #graph_file_path="dataset/test_graph"
    graph_data_path = "dataset/test_graph/barabasi_albert_30_0.graphml"
    # sample_window_step=5
    # sampling_t=10
    sampling_interval = 1

    # model_names = ["dataset__decrease_w200p400st1000",
    #                "dataset__increase_w200p400st1000"]
    for model_type in model_types:
        model_name = "NsDiff_{}_all".format(model_type)
        print("!"*10+model_name+"!"*10)
        if model_type in ["SIS"]:

            models_save_file = "ews_results/" + model_name + "/" + model_config + "st0.1"

        #  sampling_interval=1
        elif model_type in ["biomass", "neuronal"]:

            models_save_file = "ews_results/" + model_name + "/" + model_config + "st10"

        else:
            raise ValueError("dataset_type should be SIS, Kuramoto, biomass, or neuronal")
        for dataset_type in dataset_types:
            print("*"*10+"dataset_type:{}".format(dataset_type)+"*"*10+"\n")
            if dataset_type=="biomass":
                data_fliter = "*eta0.005*"
                sample_window_step = 5
                sampling_t=10
           
            elif dataset_type=="neuronal":
                data_fliter = "*tau2.0*"
                sample_window_step = 5
                sampling_t=10
            elif dataset_type=="SIS":
                data_fliter = "*d0.5*"
                sample_window_step = 20
                sampling_t=0.1
            else:
                raise ValueError("dataset type don't exist!")

            spdata_file_path = "dataset/spdata_sde_{}".format(dataset_type)

           # for graph_data_path in sorted(glob(graph_file_path + '/*' )):

            file_graph_name = graph_data_path.split("/")[-1].replace(".graphml", "")
            print(file_graph_name+"\n")
            # if file_graph_name !="barabasi_albert_50_1":
            #     continue

            nx_g = nx.read_graphml(graph_data_path)
            nx_g = nx.convert_node_labels_to_integers(nx_g)
            gnn_g = torch_geometric.utils.from_networkx(nx_g, group_node_attrs=None)
            spdata_files_path = spdata_file_path + "/{}".format(file_graph_name)
            for spdata_file in sorted(glob(spdata_files_path + '/{}.pt'.format(data_fliter))):
                print(spdata_file+"\n")
                loaded_data = torch.load(spdata_file)
                torch_time_series = loaded_data['ys_dynamic'].t().unsqueeze(
                    -1)  # [Node_num,T_obs_num,1] F=1 时间间隔为0.1
              #  print("torch_time_series shape:{}".format(torch_time_series.shape))
                time_data = loaded_data['ts_dynamic']
                for model_save_file in sorted(glob(models_save_file + '/*_0')):
                    print(model_save_file + "\n")
                    save_ews_files = model_save_file + "/{}".format(file_graph_name)
                    if not os.path.exists(save_ews_files):
                        os.makedirs(save_ews_files)
                    save_ews_files_path=save_ews_files+"/sampling_{}_t_{}".format(sample_window_step,sampling_t)+spdata_file.split("/")[-1]
                    fig_save_path=save_ews_files+"/sampling_{}_t_{}".format(sample_window_step,sampling_t)+spdata_file.split("/")[-1]
                   # print(model_save_file)
                    uncertainty_ews_list,sample_timepoints=uncertainty_ews(model_save_file=model_save_file,
                                                                           torch_time_series=torch_time_series,
                                                                           gnn_g=gnn_g,
                                                                           time_data=time_data,
                                                                           save_ews_files_path=save_ews_files_path,
                                                                           sample_window_step=sample_window_step,sampling_t=sampling_t)
                   # assert len(uncertainty_ews_list)==len(sample_timepoints), "uncertainty_ews_list length {} is not equal to sample_timepoints length {}".format(len(uncertainty_ews_list),len(sample_timepoints))
                    fig=plt.figure()
                    ax1 = fig.add_subplot(2, 1, 1)
                    plot_dynamic_trajectory(ax=ax1, ts=time_data,ys= loaded_data['ys_dynamic'], sampling_interval=sampling_interval, dynamic_type=dataset_type)
                    ax1.set_ylabel('Population Density', fontsize=12)
                    ax1.legend(loc='upper left', fontsize=10)
                    ax1.grid(alpha=0.3)
                    plt.xlim([-0.05, time_data[-1] + 0.05])

                    ax2 = fig.add_subplot(2, 1, 2)
                    ax2.plot(sample_timepoints[:len(uncertainty_ews_list)], uncertainty_ews_list, 'r.', label='Uncertainty EWS', linewidth=2)
                    #ax2.set_xlabel('Time', fontsize=12)
                    ax2.set_ylabel('uncertainy_ews', fontsize=12)
                    ax2.legend(loc='upper left', fontsize=10)
                    ax2.grid(alpha=0.3)

                    plt.xlim([-0.05, time_data[-1]+0.05])
                    plt.show()
                  #   plt.savefig(fig_save_path.replace(".pt",".png"))
                  #   plt.close()
