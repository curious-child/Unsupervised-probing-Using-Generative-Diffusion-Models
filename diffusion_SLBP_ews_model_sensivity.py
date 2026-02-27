import os

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from models.Diffusion_model.NsDiff.sigma import wv_sigma_trailing
from utils.utils import load_diffusion_model

def torch_data_preprocessing(time_data,sampling_t,return_numpy=False):
    sampling_t_min = 0.1

    sampling_interval = int(sampling_t / sampling_t_min)

    sampling_torch_time_series = time_data[::sampling_interval]  #
    if return_numpy:
        return sampling_torch_time_series.cpu().detach().numpy()
    else:
        return sampling_torch_time_series

def uncertainty_ews(model_name,torch_time_series, time_data,model_save_path,data_trend="increase",pred_dim=0,infer_params=None):

    sample_window_step=10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_config_path="{}/models/{}.yaml".format(model_save_path,model_name)
    with open(method_config_path,'r') as f:
        method_config_param=yaml.safe_load(f)
    model_path="{}/models/{}".format(model_save_path,model_name)
    model,loaded_net_param = load_diffusion_model(model_path,device=device,infer_para=infer_params,
                                                  train_model_select=method_config_param["train"]["train_model_select"])
    model.eval()
    sampling_t = method_config_param["dataset"]['sampling_t']
    #sampling_t = 100
    sampling_torch_time_series = torch_data_preprocessing(torch_time_series,sampling_t=sampling_t)
    time_points = torch_data_preprocessing(time_data,sampling_t=sampling_t,return_numpy=True)


    rolling_window = method_config_param["dataset"]['windows']
    pred_len = method_config_param["dataset"]['pred_len']
    print("pred_len:{}".format(pred_len))

    time_points = time_points[rolling_window-1::sample_window_step]
    pred_data = sampling_torch_time_series[rolling_window:, :]
    # print("pred_data shape:{}".format(pred_data.shape))
    pred_data = pred_data.unfold(0, pred_len, sample_window_step)
    pred_data = pred_data.permute(0, 2, 1)  # [n,pred_times_len,F]
    pred_datas = pred_data.unbind(0)

    data_save_path=model_save_path+"/datas/{}_pred_future_{}_{}.pt".format(model_name,data_trend,sample_window_step)
   # print(data_save_path)
    if not os.path.exists(data_save_path):
        raise ValueError("data_save_path not exists")
    else:
        data_save_list = torch.load(data_save_path)

    uncertainty_ews_list=[]
    pred_error_list=[]

    for pred_future,pred_data in tqdm(zip(data_save_list,pred_datas)):
       # print(pred_future.shape)

        if model.scaler is not None:
            pred_data = model.scaler_transform(pred_data.to(device)).to("cpu")
        pred_error = pred_future.mean(dim=-1) - pred_data  # [O,F]
        pred_error = torch.abs(pred_error)  #
        pred_error_mean = pred_error.mean(dim=0) .cpu().detach().numpy() # F
        uncertainty_ews_list.append(pred_error_mean[pred_dim])
    # for pred_future in tqdm(data_save_list):
    #
    #
    #     pred_uncertainty = pred_future.var(dim=-1)  # pred_len, F
    #     pred_uncertainty = pred_uncertainty.mean(dim=0)  # F
    #     pred_uncertainty = pred_uncertainty.cpu().detach().numpy()
    #     uncertainty_ews_list.append(pred_uncertainty[pred_dim])

    return uncertainty_ews_list,time_points


def buishand_u_test(time,data):
    """
    执行Buishand U检验用于单变点检测。
    返回变点的索引（从0开始计数）。
    """
    n = len(data)
    data = np.array(data)


    Sk = np.abs(data[10:]-data[:-10])/10
    change_point_index = np.argmax(np.abs(Sk))

    change_point_time = time[change_point_index]
    return change_point_time
def intrinsic_dimension(trajectories):
    n_trajectories, feature_dim = trajectories.shape
   # print(f"分析 {n_trajectories} 条预测轨迹，每条轨迹维度: {feature_dim}")

    # 1. 数据标准化 (中心化)
    mean_ = trajectories.mean(dim=0)
    trajectories_centered = trajectories - mean_

    # 2. 计算协方差矩阵[1,5](@ref)
    covariance_matrix = torch.mm(trajectories_centered.T, trajectories_centered) / (n_trajectories - 1)

    # 3. 特征值分解[4](@ref)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # 4. 按特征值降序排列
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues_ = eigenvalues[sorted_indices]
    components_ = eigenvectors[:, sorted_indices]

    # 5. 计算解释方差比例
    total_variance = torch.sum(eigenvalues_)
    explained_variance_ratio_ = eigenvalues_ / total_variance
    cumulative_variance_ratio_ = torch.cumsum(explained_variance_ratio_, dim=0)

    # 6. 估计本征维度
    above_threshold = cumulative_variance_ratio_ >= 0.95
   # print("above_threshold:", above_threshold)
    if above_threshold.any():
       # print("找到本征维度：", torch.where(above_threshold)[0][0] + 1)
        intrinsic_dimension_ = torch.where(above_threshold)[0][0] + 1
    else:
        intrinsic_dimension_ = feature_dim

    return intrinsic_dimension_
def plot_uncertainty_ews(plt_ews_dict):
    colors=["#1f77b4","#ff7f0e","#2ca02c"]
    linestyles=["-","--",":"]

    fig, axs = plt.subplots(3, 1, figsize=(6, 10),
                            gridspec_kw={'hspace': 0.00})  # 减少子图间距
    ts = plt_ews_dict["ts"]
    ys = plt_ews_dict["ys"]
    axs[0].plot(ts, ys, 'b.',  linewidth=2)
    axs[0].set_ylabel('Time Series')

    tipping_point_time = buishand_u_test(ts[1000:], ys[1000:])
    plt.xlim([-0.05, time_data[-1] + 0.05])
    label_="Pred-len:"
    model_params=list(plt_ews_dict["pred_ews_ts"].keys())
    #label_ = "Window-len:"
    for i,model_param in enumerate(model_params):
        sample_timepoints=plt_ews_dict["pred_ews_ts"][model_param]
        uncertainty_ews_list=plt_ews_dict["pred_ews"][model_param]
        label=label_+"{}".format(model_param)
        axs[1].plot(sample_timepoints[:len(uncertainty_ews_list)], uncertainty_ews_list,
                    color=colors[i],
                    linestyle=linestyles[i],
                    alpha=0.7, label=label,
                 linewidth=1)
    axs[1].sharex(axs[0])
    # ax2.set_xlabel('Time', fontsize=12)
    axs[1].legend(loc='best', frameon=False)
    axs[1].set_ylabel('Prediction Error')

    label_ = "Window-len:"
    model_params = list(plt_ews_dict["win_ews_ts"].keys())
    for i,model_param in enumerate(model_params):
        sample_timepoints=plt_ews_dict["win_ews_ts"][model_param]
        uncertainty_ews_list=plt_ews_dict["win_ews"][model_param]
        label = label_ + "{}".format(model_param)
        axs[2].plot(sample_timepoints[:len(uncertainty_ews_list)], uncertainty_ews_list,
                     color=colors[i],
                    linestyle=linestyles[i],
                    alpha=0.4, label=label,
                 linewidth=1)
    axs[2].sharex(axs[0])
    # ax2.set_xlabel('Time', fontsize=12)
    axs[2].set_ylabel('Prediction Error')
    axs[2].legend(loc='best',frameon=False)
    axs[2].set_xlabel('Time')
    for i in range(2):  #
        axs[i].xaxis.set_visible(False)
    for ax in axs:
        # 在x=0处添加临界点虚线
        ax.axvline(x=tipping_point_time, color='black', linestyle='--', linewidth=1, alpha=1)

    # 调整子图间距 - 使子图更紧凑

    plt.xlim([-0.05, time_data[-1] + 0.05])

    plt.tight_layout()

    return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_trends = ["decrease","increase"]
    pred_dim = 0
    model_predlens = [200,500,1000]
    model_windowlen_fix=200

    model_windowlens=[200,500,1000]
    model_predlen_fix=200
    for data_trend in data_trends:
        print("*"*10+"data trend:{}".format(data_trend)+"*"*10)
        data_path = "dataset/SLBP_model_data/SLBP_dynamic_total_time_1000000.0_N_{}/SLBP_dynamic_D_1e-05.pt".format(
            data_trend)
        loaded_data = torch.load(data_path)
        time_data = loaded_data['ts_dynamic']

        torch_time_series = loaded_data['ys_dynamic']

        plt_ews_dict = {}
        plt_ews_dict["ts"]=time_data[::1000]
        plt_ews_dict["ys"]=torch_time_series[::1000,pred_dim]


        model_save_path = "ews_results/NsDiff_pred"
        plt_ews_dict["pred_ews_ts"] = {}
        plt_ews_dict["pred_ews"] = {}
        for model_predlen in model_predlens:
        #for model_windowlen in model_windowlens:

            model_name = "dataset__w{}p{}st100".format(model_windowlen_fix,model_predlen)
            print(model_name)
            uncertainty_ews_list,sample_timepoints=uncertainty_ews(model_name,torch_time_series,time_data,
                                                                                   model_save_path=model_save_path,
                                                                                   data_trend=data_trend,
                                                                                   pred_dim=pred_dim)
            plt_ews_dict["pred_ews"][model_predlen]=uncertainty_ews_list

            plt_ews_dict["pred_ews_ts"][model_predlen]=sample_timepoints
        model_save_path = "ews_results/NsDiff_windows"
        plt_ews_dict["win_ews"] = {}
        plt_ews_dict["win_ews_ts"] = {}
        for model_windowlen in model_windowlens:

            model_name = "dataset__w{}p{}st100".format(model_windowlen, model_predlen_fix)
            print(model_name)
            uncertainty_ews_list, sample_timepoints = uncertainty_ews(model_name, torch_time_series, time_data,
                                                                      model_save_path=model_save_path,
                                                                      data_trend=data_trend,
                                                                      pred_dim=pred_dim)
            plt_ews_dict["win_ews"][model_windowlen] = uncertainty_ews_list

            plt_ews_dict["win_ews_ts"][model_windowlen] = sample_timepoints
        fig=plot_uncertainty_ews(plt_ews_dict)
        fig.savefig("ews_results" + "/pred_error_{}_SLBP_model_sensivity.svg".format(data_trend))
        plt.show()


