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
        sampling_torch_time_series = time_data[:,::sampling_interval]  #
        return sampling_torch_time_series

def uncertainty_ews( time_data,model_save_file,save_ews_files_path,sampling_t,sample_window_step=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_config_path = model_save_file + "/model_trained.yaml"
    with open(method_config_path, 'r') as f:
        method_config_param = yaml.safe_load(f)
    model_path = model_save_file + "/model_trained"
    model, loaded_net_param = load_diffusion_model(model_path, device=device, infer_para=None,
                                                   train_model_select=method_config_param["train"]["train_model_select"]
                                                   if "train_model_select" in method_config_param[
                                                       "train"].keys() else None)
    model.eval()
    time_points = torch_data_preprocessing(time_data,sampling_t=sampling_t,return_numpy=True)



    rolling_window = 100


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
    if dataset_type=="biomass" or dataset_type=="neuronal":
	    Sk = np.abs(data[window_size:]-data[:-window_size])/window_size
	    change_point_index = np.argmax(np.abs(Sk))
    elif dataset_type=="SIS":

	    mean= np.array([np.mean(data[i:i+window_size]) for i in range(n-window_size)])
	    if data_trend=="increase":
		    change_point_index=np.argwhere(mean>1e-2).flatten()[0]
	    else:
		    change_point_index=np.argwhere(mean<1e-2).flatten()[0]
    change_point_time = time[change_point_index]
    return change_point_time
def plot_ews_compare(plt_data_dict,sample_ts,dynamic_type,sample_window_step):
    print("plot_dynamic_trajectory")

    fig, axs = plt.subplots(3, 1, figsize=(8, 10),
                            gridspec_kw={'hspace': 0.00})  # 减少子图间距

    if dynamic_type == "biomass":
        fig.suptitle('Resource Biomass Dynamics')
    elif dynamic_type == "neuronal":
        fig.suptitle('Wilson-Cowan Neuronal Dynamics')
    elif dynamic_type == "SIS":
        fig.suptitle('SIS Dynamics')
    else:
        raise ValueError("dynamic_type don't exist!")
    # 1. 绘制时间序列子图
    ts = torch_data_preprocessing(plt_data_dict["ts"], sampling_t=sample_ts, return_numpy=True)
    ys = torch_data_preprocessing(plt_data_dict["ys"], sampling_t=sample_ts, return_numpy=True)

    axs[0].plot(ts, ys.mean(axis=1), 'b-', label='Original Data', linewidth=2)
    tipping_point_time = buishand_u_test(ts, ys.mean(axis=1))
    y1=np.array(plt_data_dict["NsDiff"]["pred_mean"])
    axs[0].plot(plt_data_dict["NsDiff"]["ts"][:len(plt_data_dict["NsDiff"]["pred_mean"])], y1, 'r-', label='Predicted Future (NsDiff)', linewidth=2)
    y2=np.array(plt_data_dict["DiffSTG"]["pred_mean"])
    axs[0].plot(plt_data_dict["DiffSTG"]["ts"][:len(plt_data_dict["DiffSTG"]["pred_mean"])], y2, 'g-', label='Predicted Future (DiffSTG)', linewidth=2)
    y1_uncertainty=np.sqrt(np.array(plt_data_dict["NsDiff"]["ews"]))
    y2_uncertainty=np.sqrt(np.array(plt_data_dict["DiffSTG"]["ews"]))

    axs[0].fill_between(plt_data_dict["NsDiff"]["ts"][:len(plt_data_dict["NsDiff"]["pred_mean"])], y1 - y1_uncertainty, y1 + y1_uncertainty, color='r', alpha=0.2)
    axs[0].fill_between(plt_data_dict["DiffSTG"]["ts"][:len(plt_data_dict["DiffSTG"]["pred_mean"])], y2 -y2_uncertainty, y2 + y2_uncertainty, color='g', alpha=0.2)
    axs[0].set_ylabel('Time Series')
    axs[0].legend(loc='best', frameon=False)

    # 2. 绘制自相关子图
    tipping_point_time_pv_index = np.argmin(plt_data_dict["NsDiff"]["ews"] )
    tipping_point_time_pv = plt_data_dict["NsDiff"]["ts"][tipping_point_time_pv_index]
    axs[1].plot(plt_data_dict["NsDiff"]["ts"][:len(plt_data_dict["NsDiff"]["ews"])], plt_data_dict["NsDiff"]["ews"], 'r.', linewidth=2)
    axs[1].set_ylabel('Predicted Uncertainty \n(NsDiff)')
    axs[1].sharex(axs[0])

    # 3. 绘制方差子图
    axs[2].plot(plt_data_dict["DiffSTG"]["ts"][:len(plt_data_dict["DiffSTG"]["ews"])], plt_data_dict["DiffSTG"]["ews"], 'g.', linewidth=2)
    axs[2].set_ylabel('Predicted Uncertainty \n (DiffSTG)')
    axs[2].sharex(axs[0])
    axs[2].set_xlabel('Time')



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


    graph_name = "barabasi_albert_30_0"

    sampling_interval = 1
    diffusion_model_names = ["NsDiff","DiffSTG"   ]

    for data_trend in data_trends:
        print("*" * 10 + "data_trend:{}".format(data_trend) + "*" * 10 + "\n")
        for dataset_type in dataset_types:
            print("*" * 10 + "dataset_type:{}".format(dataset_type) + "*" * 10 + "\n")
            if dataset_type == "biomass":

                sample_window_step = 5
                sampling_t = 10
            elif dataset_type == "Kuramoto":

                sample_window_step = 20
                sampling_t = 0.1
            elif dataset_type == "neuronal":

                sample_window_step = 5
                sampling_t = 10
            elif dataset_type == "SIS":

                sample_window_step = 20
                sampling_t = 0.1
            else:
                raise ValueError("dataset type don't exist!")

            plt_data_dict = {}
            for diffusion_model_name in diffusion_model_names:

                save_ews_files_path = "ews_results/model_compare/{}/{}".format(diffusion_model_name,dataset_type)
                plt_data_dict[diffusion_model_name] = {}
                if diffusion_model_name=="DiffSTG" and (dataset_type=="biomass" or dataset_type=="neuronal"):
                    sample_window_step = 50
                    sampling_t = 1

                for ews_file_path in sorted(glob(save_ews_files_path + '/*_{}.pt'.format(data_trend))):

                    spdata_file = "dataset/spdata_sde_{}/{}/".format(dataset_type,graph_name)+ews_file_path.split("/")[-1]
                    print(spdata_file+"\n")
                    loaded_data = torch.load(spdata_file)
                    time_data = loaded_data['ts_dynamic']
                    plt_data_dict["ts"]=time_data
                    plt_data_dict["ys"]=loaded_data['ys_dynamic']
                       # print(model_save_file)
                    pred_mean,uncertainty_ews_list,sample_timepoints=uncertainty_ews(
                                                                           time_data=time_data,
                                                                           model_save_file=save_ews_files_path,
                                                                           save_ews_files_path=ews_file_path,
                                                                           sample_window_step=sample_window_step,
                                                                           sampling_t=sampling_t)
                    plt_data_dict[diffusion_model_name]["pred_mean"] = pred_mean
                    plt_data_dict[diffusion_model_name]["ews"]=uncertainty_ews_list
                    plt_data_dict[diffusion_model_name]["ts"]=sample_timepoints
                 #   assert len(uncertainty_ews_list)==len(sample_timepoints), "uncertainty_ews_list length {} is not equal to sample_timepoints length {}".format(len(uncertainty_ews_list),len(sample_timepoints))

            fig = plot_ews_compare(plt_data_dict,dynamic_type=dataset_type,sample_ts=sampling_t,sample_window_step=sample_window_step)
            fig.savefig("ews_results/model_compare"+"/{}_{}_{}.svg".format(dataset_type,data_trend,graph_name))
          #  plt.show()