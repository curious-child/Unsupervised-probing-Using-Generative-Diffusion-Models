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

def uncertainty_ews(model_save_file,torch_time_series,gnn_g, time_data,save_ews_files_path,infer_params=None,sample_window_step=50):
    if infer_params is None:
        infer_params = {  "parallel_sampling": 100,  # parallel sampling number
                            "sequential_sampling": 1}

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

    sampling_torch_time_series = torch_data_preprocessing(torch_time_series,sampling_t=method_config_param["dataset"]['sampling_t'])
    #print("sampling_torch_time_series shape:{}".format(sampling_torch_time_series.shape))
    time_points = torch_data_preprocessing(time_data,sampling_t=method_config_param["dataset"]['sampling_t'],return_numpy=True)

    pred_times_len = method_config_param["dataset"]['pred_len']
    rolling_window = method_config_param["dataset"]['windows']
   # scaler_type = method_config_param["dataset"]['scaler_type']
    timeseries_data = sampling_torch_time_series.unfold(1, rolling_window, sample_window_step)  # [Node_num,n,F,windows_len]
    timeseries_data = timeseries_data.permute(0,1, 3, 2)  # [Node_num,n,windows_len,F]
    time_points = time_points[rolling_window-1::sample_window_step]
    # print("timeseries_data shape:{}".format(timeseries_data.shape))
    timeseries_datas = timeseries_data.unbind(1)

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
            for time_series in tqdm(list(timeseries_datas),leave=False):
                graph_data_copy = gnn_g.clone()
                if model.scaler is not None:
                    time_series_trans = model.scaler_transform(time_series.to(device))
                graph_data_copy.x = time_series_trans.clone().to(device)  # [node_num,pred_times_len,1]

               # print("time series shape",time_series.shape)
                pred_future,_ = model.evaluation_step(graph_data_copy)##node_num,pred_times_len,1,n_z_samples
                pred_future=pred_future.squeeze(-2) #node_num,pred_times_len,n_z_samples
                pred_future = pred_future[:, -pred_times_len:, :]
                data_save_list.append(pred_future)


              #  print("pred_future shape:{}".format(pred_future.shape))
                # if scaler is not None:
                #     pred_future = scaler.inverse_transform(pred_future)
               # pred_future=pred_future[:,rolling_window:,:]
                pred_uncertainty = pred_future.var(dim=-1)# node_num,pred_times_len
                pred_uncertainty_nodes = pred_uncertainty.mean(dim=1) #node_num
                pred_uncertainty_mean = pred_uncertainty_nodes.mean()
                pred_uncertainty = pred_uncertainty_mean.cpu().detach().numpy()
                uncertainty_ews_list.append(pred_uncertainty)
        torch.save(data_save_list, data_save_path)
    else:
        for pred_future in tqdm(data_save_list):
           # pred_future=pred_future[:,rolling_window:,:]
            pred_uncertainty = pred_future.var(dim=-1)  # node_num,pred_times_len
            pred_uncertainty_nodes = pred_uncertainty.mean(dim=1)  # node_num
            pred_uncertainty_mean = pred_uncertainty_nodes.mean()
            pred_uncertainty = pred_uncertainty_mean.cpu().detach().numpy()
            uncertainty_ews_list.append(pred_uncertainty)
    return uncertainty_ews_list,time_points
def plot_dynamic_trajectory(ax,ts, ys,sampling_interval,dynamic_type):
    print("plot_dynamic_trajectory")

    if dynamic_type=='Kuramoto':
    # complex_phase = np.exp(1j * ys)
    # ys = np.abs(np.mean(complex_phase, axis=1))
#完全同步时，dθ/dt=0
        dy=torch.zeros_like(ys)
        dy[:-1]=ys[1:]-ys[:-1]
        dy[-1]=dy[-2]
        dy=torch.abs(dy)
        ys=dy.sum(dim=1)
    elif dynamic_type in ['SIS',"biomass","neuronal"]:
        ys=ys.mean(dim=1)
    else:
        raise ValueError("dynamic_type should be Kuramoto or SIS")

    # 1. 状态变量X随时间变化

    ax.plot(ts[::sampling_interval], ys[::sampling_interval], 'b-', label='dynamic: {}'.format(dynamic_type), linewidth=2)
    #ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    #data_trends = ["decrease","increase"]
    dataset_types=["biomass"]#"neuronal","SIS","""biomass",
    model_type="DiffSTG"
    model_config="dataset__w100p100"
    graph_file_path="dataset/test_graph"
    sampling_interval = 1
    # model_names = ["dataset__decrease_w200p400st1000",
    #                "dataset__increase_w200p400st1000"]
    for dataset_type in dataset_types:
        print("*"*10+"dataset_type:{}".format(dataset_type)+"*"*10+"\n")
        model_name = "{}_{}".format(model_type, dataset_type)
        if dataset_type in ["SIS"]:
            model_save_file="ews_results/"+model_name+"/"+model_config+"st0.1"
            sample_window_step=20
          #  sampling_interval=1
        elif dataset_type in ["biomass","neuronal"]:
            model_save_file="ews_results/"+model_name+"/"+model_config+"st1"
            sample_window_step = 50
        else:
            raise ValueError("dataset_type should be SIS, biomass, or neuronal")

        spdata_file_path = "dataset/spdata_sde_{}".format(dataset_type)

        for graph_data_path in sorted(glob(graph_file_path + '/*' )):
            file_graph_name = graph_data_path.split("/")[-1].replace(".graphml","")
            print(file_graph_name+"\n")
            if file_graph_name !="barabasi_albert_30_0":
                continue

            nx_g = nx.read_graphml(graph_data_path)
            nx_g = nx.convert_node_labels_to_integers(nx_g)
            gnn_g = torch_geometric.utils.from_networkx(nx_g, group_node_attrs=None)
            spdata_files_path=spdata_file_path+"/{}".format(file_graph_name)
            for spdata_file in sorted(glob(spdata_files_path + '/*.pt')):
                print(spdata_file+"\n")
                loaded_data = torch.load(spdata_file)
                torch_time_series = loaded_data['ys_dynamic'].t().unsqueeze(
                    -1)  # [Node_num,T_obs_num,1] F=1 时间间隔为0.1
              #  print("torch_time_series shape:{}".format(torch_time_series.shape))
                time_data = loaded_data['ts_dynamic']

                save_ews_files = model_save_file + "/{}".format(file_graph_name)
                if not os.path.exists(save_ews_files):
                    os.makedirs(save_ews_files)
                save_ews_files_path=save_ews_files+"/"+spdata_file.split("/")[-1]
               # print(model_save_file)
                uncertainty_ews_list,sample_timepoints=uncertainty_ews(model_save_file,
                                                                       torch_time_series,
                                                                       gnn_g,
                                                                       time_data,
                                                                       save_ews_files_path,
                                                                        sample_window_step=sample_window_step)
                assert len(uncertainty_ews_list)==len(sample_timepoints), "uncertainty_ews_list and sample_timepoints should have the same length"

                # ax1 = plt.subplot(2, 1, 1)
                # ax1.plot(time_data[::sampling_interval], torch_time_series[:,::sampling_interval, 0].mean(dim=0), 'b.', label='Bream (X)', linewidth=2)
                # ax1.set_ylabel('Population Density', fontsize=12)
                # ax1.legend(loc='upper left', fontsize=10)
                # ax1.grid(alpha=0.3)
                # plt.xlim([-0.05, time_data[-1] + 0.05])
                #
                # ax2 = plt.subplot(2, 1, 2)
                # ax2.plot(sample_timepoints[:len(uncertainty_ews_list)], uncertainty_ews_list, 'r.', label='Uncertainty EWS', linewidth=2)
                # #ax2.set_xlabel('Time', fontsize=12)
                # ax2.set_ylabel('Uncertainty EWS', fontsize=12)
                # ax2.legend(loc='upper left', fontsize=10)
                # ax2.grid(alpha=0.3)
                #
                # plt.xlim([-0.05, time_data[-1]+0.05])
                # plt.show()
