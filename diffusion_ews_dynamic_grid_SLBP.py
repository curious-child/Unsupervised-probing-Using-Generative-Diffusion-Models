import os

import torch
import yaml
from tqdm import tqdm

from utils.utils import load_diffusion_model

def torch_data_preprocessing(time_data,sampling_t,return_numpy=False):
    sampling_t_min = 0.1

    sampling_interval = int(sampling_t / sampling_t_min)

    sampling_torch_time_series = time_data[::sampling_interval]  #
    if return_numpy:
        return sampling_torch_time_series.cpu().detach().numpy()
    else:
        return sampling_torch_time_series

def uncertainty_ews(model_name,torch_time_series, time_data,data_trend="increase",pred_dim=0,
                    infer_params=None,
                    mode_save_path = "ews_results/NsDiff_nopred_trends",
                    data_save_path="datas"):
    sample_window_step=10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_config_path="{}/models".format(mode_save_path)+"/{}.yaml".format(model_name)
    with open(method_config_path,'r') as f:
        method_config_param=yaml.safe_load(f)
    model_path="{}/models/{}".format(mode_save_path,model_name)
    model,loaded_net_param = load_diffusion_model(model_path,device=device,infer_para=infer_params,
                                                  train_model_select=method_config_param["train"]["train_model_select"])
    model.eval()

    sampling_torch_time_series = torch_data_preprocessing(torch_time_series,sampling_t=method_config_param["dataset"]['sampling_t'])
    sampling_torch_time_series_smooth = lowess_denoise_torch(sampling_torch_time_series).to(sampling_torch_time_series.dtype)
    time_points = torch_data_preprocessing(time_data,sampling_t=method_config_param["dataset"]['sampling_t'],return_numpy=True)

    # print(sampling_torch_time_series_smooth.dtype)
    # print(sampling_torch_time_series.dtype)
    rolling_window = method_config_param["dataset"]['windows']
    pred_times_len = method_config_param["dataset"]['pred_len']
   # scaler_type = method_config_param["dataset"]['scaler_type']
    timeseries_data = sampling_torch_time_series_smooth.unfold(0, rolling_window, sample_window_step)  # [n,F,rolling_window]
    timeseries_data = timeseries_data.permute(0, 2, 1)  # [n,rolling_window,F]
    time_points = time_points[rolling_window-1::sample_window_step]
    timeseries_datas = timeseries_data.unbind(0)
   # print("sampling_torch_time_series ",sampling_torch_time_series.shape)
   #  pred_data = sampling_torch_time_series[ rolling_window:, :]
   #  # print("pred_data shape:{}".format(pred_data.shape))
   #  pred_data = pred_data.unfold(0, pred_times_len, sample_window_step)
   #  pred_data = pred_data.permute(0, 2,1)  # [n,pred_times_len,F]
   #  pred_datas = pred_data.unbind(0)
    # print("timeseries_data shape:{}".format(timeseries_data.shape))


    # if scaler_type ==  "StandardScaler":
    #     from torch_timeseries.scaler.standard import StandardScaler
    #     scaler = StandardScaler()
    #     scaler.fit(sampling_torch_time_series)
    # elif scaler_type == "None":
    #     scaler=None
    # else:
    #     raise ValueError("scaler_type should be StandardScaler or None")


    data_save_path1="{}/{}_pred_future_{}_{}.pt".format(data_save_path,model_name,data_trend,sample_window_step)

    if not os.path.exists(data_save_path1):
        data_save_list = []
    else:
        data_save_list = torch.load(data_save_path1)
    data_save_path_smooth = "{}_smooth/{}_pred_future_{}_{}.pt".format(data_save_path, model_name, data_trend, sample_window_step)

    if not os.path.exists(data_save_path_smooth):
        data_save_list_smooth = []
        print("no save path",data_save_path_smooth)
    else:
        data_save_list_smooth = torch.load(data_save_path_smooth)

    uncertainty_ews_list=[]
    uncertainty_ews_list_smooth = []
    if not data_save_list:
        with torch.no_grad():
            for time_series in tqdm(timeseries_datas):

                if model.scaler is not None:
                    time_series = model.scaler_transform(time_series.to(device))

                time_series = time_series.clone().unsqueeze(0).to(device)  # [1,rolling_window,F]
               # print("time series shape",time_series.shape)
                pred_future,_ = model.evaluation_step(time_series)##1, pred_len, F, n_z_samples
                pred_future=pred_future.squeeze(0) #pred_len, F, n_z_samples
                data_save_list.append(pred_future)


              #  print("pred_future shape:{}".format(pred_future.shape))
                # if scaler is not None:
                #     pred_future = scaler.inverse_transform(pred_future)
                pred_uncertainty = pred_future.var(dim=-1)# O, F
                pred_uncertainty = pred_uncertainty.mean(dim=0) #F
                pred_uncertainty = pred_uncertainty.cpu().detach().numpy()
                uncertainty_ews_list.append(pred_uncertainty[pred_dim])
        torch.save(data_save_list, data_save_path)
    else:
        # for pred_future,pred_data in tqdm(zip(data_save_list,pred_datas)):
        #     # print("pred future",pred_future.shape)
        #     # print("pred_data",pred_data.shape)
        #     if model.scaler is not None:
        #         pred_data = model.scaler_transform(pred_data.to(device)).to("cpu")
        #     pred_error = pred_future.mean(dim=-1) - pred_data # [O,F]
        #     pred_error = torch.abs(pred_error)  #
        #     pred_error_mean = pred_error.mean(dim=0)#F
        #     uncertainty_ews_list.append(pred_error_mean.cpu().detach().numpy()[pred_dim])
        for pred_future in tqdm(data_save_list):
            pred_uncertainty = pred_future.var(dim=-1)# O, F
            pred_uncertainty = pred_uncertainty.mean(dim=0) #F
            pred_uncertainty = pred_uncertainty.cpu().detach().numpy()
            uncertainty_ews_list.append(pred_uncertainty[pred_dim])
        for pred_future in tqdm(data_save_list_smooth):
            pred_uncertainty = pred_future.var(dim=-1)# O, F
            pred_uncertainty = pred_uncertainty.mean(dim=0) #F
            pred_uncertainty = pred_uncertainty.cpu().detach().numpy()
            uncertainty_ews_list_smooth.append(pred_uncertainty[pred_dim])
    return uncertainty_ews_list,time_points,uncertainty_ews_list_smooth,sampling_torch_time_series_smooth
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

def lowess_denoise_torch(timeseries_tensor,frac=0.1,it=3):

   # print(timeseries_tensor)
    num_feature=timeseries_tensor.shape[1]
    smoothed_list=[]
    for i in range(num_feature):
        timeseries_np = timeseries_tensor[:,i].cpu().numpy()
        time_index=np.arange(len(timeseries_np))
        timeseries_np_lowess=sm.nonparametric.lowess(timeseries_np,
                                                     time_index,
                                                     frac=frac,
                                                     it=it)
        smoothed_list.append(torch.from_numpy(timeseries_np_lowess[:,1]))
    data_smoothed=torch.stack(smoothed_list,dim=1)
    return data_smoothed

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    import statsmodels.api as sm

    mode_save_path = "ews_results/NsDiff_nopred_trends"
    data_trends = ["decrease","increase"]
    #model_names=["dataset__w200p200st100.yaml"]
    model_names = ["dataset__w200p200st100"]
                   #  "dataset__decrease_w200p200st100",
                   # "dataset__increase_w200p200st100"]
    total_times = [2e6,3e6]
    Ds=[1e-5 ]#5e-4, 1e-4,
    for data_trend in data_trends:
        print("*"*10+"data trend:{}".format(data_trend)+"*"*10)
        for total_time in total_times:
            for D in Ds:
                data_path = "dataset/SLBP_model_data_test/SLBP_dynamic_total_time_{}_N_{}/SLBP_dynamic_D_{}.pt".format(total_time,
                    data_trend,D)
                loaded_data = torch.load(data_path)
                time_data = loaded_data['ts_dynamic']
                torch_time_series = loaded_data['ys_dynamic']

                for model_name in model_names:
                    print(model_name)

                    data_save_path=mode_save_path+"/datas/T_{}_D{}".format(total_time,D)
                    if not os.path.exists(data_save_path):
                        os.makedirs(data_save_path)
                    uncertainty_ews_list,sample_timepoints,uncertainty_ews_list_smooth,sampling_torch_time_series_smooth=uncertainty_ews(model_name,torch_time_series,time_data,
                                                                           data_trend=data_trend,
                                                                           pred_dim=0,
                                                                           mode_save_path=mode_save_path,
                                                                           data_save_path=data_save_path)

#                    assert len(uncertainty_ews_list)==len(sample_timepoints), "uncertainty_ews_list and sample_timepoints should have the same length"
                    fig, axs = plt.subplots(2, 1, figsize=(6, 6),
                                            gridspec_kw={'hspace': 0.00})  # 减少子图间距
                    ts=time_data[::1000]
                    ys=torch_time_series[::1000, 0]
                    ys_smooth=sampling_torch_time_series_smooth[:, 0]
                    tipping_point_time = buishand_u_test(ts[1000:], ys_smooth[1000:])
                    axs[0].plot(ts,ys, 'b.', label='Orign', linewidth=2)
                    axs[0].plot(ts, ys_smooth, 'r.', label='Smooth', linewidth=1)
                    axs[0].set_ylabel('Time Series')
                    axs[0].legend(loc='best')
                 #   ax1.grid(alpha=0.3)
                    plt.xlim([-0.05, time_data[-1] + 0.05])
                    # print("sample_timepoints",len(sample_timepoints))
                    # print("uncertainty_ews_list",len(uncertainty_ews_list))

                    axs[1].plot(sample_timepoints[:len(uncertainty_ews_list)], uncertainty_ews_list, 'b.', label='Orign', linewidth=2)
                    axs[1].plot(sample_timepoints[:len(uncertainty_ews_list)], uncertainty_ews_list_smooth, 'r.', label='Smooth', linewidth=1)

                    #ax2.set_xlabel('Time', fontsize=12)
                    axs[1].set_ylabel('Predicted Uncertainty')
                    axs[1].legend(loc='best',)
                    axs[1].sharex(axs[0])
                    axs[1].set_xlabel('Time')
                 #   ax2.grid(alpha=0.3)
                    for i in range(1):  #
                        axs[i].xaxis.set_visible(False)
                    for ax in axs:
                        # 在x=0处添加临界点虚线
                        ax.axvline(x=tipping_point_time, color='black', linestyle='--', linewidth=1, alpha=1)

                    plt.xlim([-0.05, time_data[-1]+0.05])
                    plt.savefig(data_save_path+"/T_{}_D{}_trends_{}.svg".format(total_time,D,data_trend))
                    plt.close()
                 #   plt.show()