import os
from glob import glob

import networkx as nx
import numpy as np
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
        sampling_torch_time_series = time_data[::sampling_interval,:]  #
        return sampling_torch_time_series

def uncertainty_ews( time_data,model_save_file,save_ews_files_path,sample_window_step=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_config_path = model_save_file + "/model_trained.yaml"
    with open(method_config_path, 'r') as f:
        method_config_param = yaml.safe_load(f)
    model_path = model_save_file + "/model_trained"
    model, loaded_net_param = load_diffusion_model(model_path, device=device, infer_para=None,
                                                   train_model_select=method_config_param["train"]["train_model_select"]
                                                   if "train_model_select" in method_config_param[
                                                       "train"].keys() else None)
    sampling_t=method_config_param["dataset"]["sampling_t"]
    model.eval()
    time_points = torch_data_preprocessing(time_data,sampling_t=sampling_t,return_numpy=True)



    rolling_window = method_config_param["dataset"]["windows"]


    time_points = time_points[rolling_window-1::sample_window_step]



    data_save_path=save_ews_files_path

    if not os.path.exists(data_save_path):

        return ValueError("data_save_path not exist")
    else:
        data_save_list = torch.load(data_save_path)

    uncertainty_ews_list=[]
    pred_mean_list = []
    for pred_future in tqdm(data_save_list):
        if model.scaler is not None:
            pred_future = model.scaler_inverse_transform(pred_future.to(device)).to("cpu")
        pred_uncertainty = pred_future.var(dim=-1)  # node_num,pred_times_len
      #  print("pred_uncertainty shape:{}".format(pred_uncertainty.shape))
        pred_uncertainty_nodes = pred_uncertainty.mean(dim=1)  # node_num
        pred_uncertainty_mean = pred_uncertainty_nodes.mean()
        pred_uncertainty = pred_uncertainty_mean.cpu().detach().numpy()
        pred_mean=pred_future.mean().cpu().detach().numpy()
        pred_mean_list.append(pred_mean)
        uncertainty_ews_list.append(pred_uncertainty)
    return pred_mean_list,uncertainty_ews_list,time_points


def buishand_u_test(time,data):
    """
    执行Buishand U检验用于单变点检测。
    返回变点的索引（从0开始计数）。
    """
    n = len(data)
    data = np.array(data)
    mean_total = np.mean(data)

    window_size = 10
    if dataset_type == "biomass" or dataset_type == "neuronal":
        Sk = np.abs(data[window_size:] - data[:-window_size]) / window_size
        change_point_index = np.argmax(np.abs(Sk)) + window_size // 2
    elif dataset_type == "SIS":

        mean = np.array([np.mean(data[i:i + window_size]) for i in range(n - window_size)])
        if data_trend == "increase":
            change_point_index = np.argwhere(mean > 1e-2).flatten()[0] + window_size // 2
        else:
            change_point_index = np.argwhere(mean < 1e-2).flatten()[0] + window_size // 2
    change_point_time = time[change_point_index]
    return change_point_time
def plot_ews_compare(plt_data_dict,sample_ts,dynamic_type,sample_window_step):
    print("plot_dynamic_trajectory")

    fig, axs = plt.subplots(2, 1, figsize=(6, 6),
                            gridspec_kw={'hspace': 0.00})  # 减少子图间距

    # if dynamic_type == "biomass":
    #     fig.suptitle('Resource biomass dynamics')
    # elif dynamic_type == "neuronal":
    #     fig.suptitle('Wilson-Cowan neuronal dynamics')
    # elif dynamic_type == "SIS":
    #     fig.suptitle('SIS dynamics')
    # else:
    #     raise ValueError("dynamic_type don't exist!")
    # 1. 绘制时间序列子图

    ts= torch_data_preprocessing(plt_data_dict["ts"], sampling_t=sample_ts, return_numpy=True)
    ys=torch_data_preprocessing(plt_data_dict["ys"], sampling_t=sample_ts, return_numpy=True)

    axs[0].plot(ts, ys.mean(axis=1), 'b-', label='Original Data', linewidth=2)
    tipping_point_time=buishand_u_test(ts,ys.mean(axis=1))
    y1=np.array(plt_data_dict["pred_mean"])
    axs[0].plot(plt_data_dict["ews_ts"][:len(plt_data_dict["pred_mean"])], y1, 'r-', label='Predicted Future', linewidth=2)

    y1_uncertainty=np.sqrt(np.array(plt_data_dict["ews"]))


    axs[0].fill_between(plt_data_dict["ews_ts"][:len(plt_data_dict["pred_mean"])], y1 - y1_uncertainty, y1 + y1_uncertainty, color='r', alpha=0.2)

    axs[0].set_ylabel('Time Series')
    axs[0].legend(loc='best', frameon=False)

    # 2. 绘制自相关子图
    tipping_point_time_pv_index = np.argmin(plt_data_dict["ews"] )
    tipping_point_time_pv = plt_data_dict["ews_ts"][tipping_point_time_pv_index]
    axs[1].plot(plt_data_dict["ews_ts"][:len(plt_data_dict["ews"])], plt_data_dict["ews"], 'r.', linewidth=2)
    axs[1].set_ylabel('Predicted Uncertainty')
    axs[1].sharex(axs[0])
    axs[1].set_xlabel('Time')





    # 设置共享x轴属性
    for i in range(1):  #
        axs[i].xaxis.set_visible(False)

    # 设置所有子图的x轴范围
    for ax in axs:
        # 在x=0处添加临界点虚线
        ax.axvline(x=tipping_point_time, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=tipping_point_time_pv, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # 调整子图间距 - 使子图更紧凑
    plt.tight_layout()

    # 显示图形
    return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    data_trends = ["decrease","increase"]

    dataset_types = ["neuronal","biomass","SIS"]

    graph_types = ["BA","ER","WS"]
    graph_name = "barabasi_albert_30_0"


    diffusion_model_name = "NsDiff"

    for data_trend in data_trends:
        print("*" * 10 + "data_trend:{}".format(data_trend) + "*" * 10 + "\n")
        for dataset_type in dataset_types:
            print("*" * 10 + "dataset_type:{}".format(dataset_type) + "*" * 10 + "\n")
            if dataset_type == "biomass":
                spdata_name="biomass_dynamic_eta0.005r0.7_{}.pt".format(data_trend)
                sample_window_step = 5
                sample_ts = 10


            elif dataset_type == "neuronal":
                spdata_name="neuronal_dynamic_eta0.01tau2.0_{}.pt".format(data_trend)
                sample_window_step = 5
                sample_ts = 10

            elif dataset_type == "SIS":
                spdata_name="SIS_dynamic_eta0.0001d0.5_{}.pt".format(data_trend)
                sample_window_step = 20
                sample_ts = 0.1

            else:
                raise ValueError("dataset type don't exist!")

            plt_data_dict = {}

            model_save_file = "ews_results/ews_generalization/graph/{}".format(dataset_type)

            for graph_type in graph_types:
                if graph_type == "BA":
                    graph_name = "barabasi_albert_30_0"
                elif graph_type == "ER":
                    graph_name = "erdos_renyi_50_0"
                elif graph_type == "WS":
                    graph_name = "small-world_70_0"
                else:
                    raise ValueError("graph type don't exist!")
                print("*" * 10 + "graph_type:{}".format(graph_type) + "*" * 10 + "\n")

                spdata_file = "dataset/spdata_sde_{}/{}/".format(dataset_type,graph_name)+spdata_name
                ews_file_path=model_save_file+"/{}_{}.pt".format(graph_type,data_trend)
                print(ews_file_path+"\n")
                loaded_data = torch.load(spdata_file)
                time_data = loaded_data['ts_dynamic']
                plt_data_dict["ts"]=time_data
                plt_data_dict["ys"]=loaded_data['ys_dynamic']
                   # print(model_save_file)
                pred_mean,uncertainty_ews_list,sample_timepoints=uncertainty_ews(
                                                                       time_data=time_data,
                                                                       model_save_file=model_save_file,
                                                                       save_ews_files_path=ews_file_path,
                                                                       sample_window_step=sample_window_step,
                                                                       )
                plt_data_dict["pred_mean"] = pred_mean
                plt_data_dict["ews"]=uncertainty_ews_list
                plt_data_dict["ews_ts"]=sample_timepoints
                 #   assert len(uncertainty_ews_list)==len(sample_timepoints), "uncertainty_ews_list length {} is not equal to sample_timepoints length {}".format(len(uncertainty_ews_list),len(sample_timepoints))

                fig = plot_ews_compare(plt_data_dict,dynamic_type=dataset_type,sample_ts=sample_ts,sample_window_step=sample_window_step)
                fig.savefig("ews_results/ews_generalization/graph"+"/{}_{}_{}.svg".format(dataset_type,data_trend,graph_name))
             #   plt.show()