import os
import itertools as it
import torch
import torchsde
import numpy as np
import matplotlib.pyplot as plt


# 模型参数 (来自论文表2)
dy_params_dict = {
    'i_b': [3e-4], 'i_p': [3e-4], 'r': [7.5e-3],
    'H1': [0.5],
    'H2': [0.1],
    'H3': [20.0],
    'H4': [15.0],
    'c_b': [7.5e-5],
    'c_p': [2.75e-4],
    'p_r': [5e-2],
    'c_e': [0.1],
    'm_p': [2.25e-3],
    'K': [1.0],
    'D': [1e-5, 5e-6]
}


class DynamicSLBPSDE(torchsde.SDEIto):

    def __init__(self, N_func, params):
        super().__init__(noise_type="diagonal")
        self.N_func = N_func  # 随时间变化的营养水平函数
        self.params = params

    # 漂移函数 f(x,t)
    def f(self, t, y):
        X, Y = y[0], y[1]
        p = self.params

        # 获取当前时间的营养水平
        current_N = self.N_func(t.item())

        # 中间函数计算
        V = p['K'] * p['H3'] ** 2 / (X ** 2 + p['H3'] ** 2)
        F_R = X ** 2 / (X ** 2 + p['H4']** 2)

        # bream 方程
        dX = (p['i_b'] + p['r'] * X * (current_N / (current_N + p['H1']))
              - p['c_b'] * X ** 2 - p['p_r'] * F_R * Y)

        # pike 方程
        dY = (p['i_p'] + p['c_e'] * p['p_r'] * F_R * Y * (V / (V + p['H2']))
              - p['m_p'] * Y - p['c_p'] * Y ** 2)

        return torch.stack([dX, dY], dim=0)

    # 扩散函数 g(x,t)
    def g(self, t, y):
        p = self.params
        return torch.tensor([(2 * p['D'])** 0.5, (0.2 * p['D'])** 0.5]).reshape(-1,1)


def simulate_with_burn_in(N_func,params,N_init=0.0, file_save_path=None,total_time=100.0, burn_time=20, dt=0.1,max_one_time=1e6):
    """
    带预热期的动态模拟
    total_time: 总模拟时间
    burn_time: 预热期时间
    dt: 时间步长
    """
    # 第一阶段：预热期 (固定N=0.0)
    sde_burn = DynamicSLBPSDE(lambda t: N_init,params)  # 预热期固定N=0.0
    y0_burn = torch.rand(2, 1)  # 初始状态
    ts_burn = torch.linspace(0, burn_time, int(burn_time / dt))

    with torch.no_grad():
        ys_burn = torchsde.sdeint(sde_burn, y0_burn, ts_burn, method='milstein',adaptive=True).squeeze(-1)

    # 第二阶段：动态变化期 (N随时间变化)
    sde_dynamic = DynamicSLBPSDE(N_func, params)
    y0_dynamic = ys_burn[-1].reshape(-1, 1)  # 从预热结束状态开始
    dy_data_record={}
    if os.path.exists(file_save_path+"/"+'SLBP_dynamic_D_' + str(sde_dynamic.params['D']) + '.pt'):
        dy_data_record = torch.load(file_save_path+"/"+'SLBP_dynamic_D_' + str(sde_dynamic.params['D']) + '.pt')
        return dy_data_record['ts_dynamic'].to(dtype=torch.float32).numpy(), dy_data_record['ys_dynamic'].to(dtype=torch.float32).numpy(), dy_data_record['N_values'].to(dtype=torch.float32).numpy()
    else:
        if max_one_time>total_time:
            ts_dynamic = torch.linspace(0,  total_time, int(total_time / dt),dtype=torch.float64)
            with torch.no_grad():
                ys_dynamic = torchsde.sdeint(sde_dynamic, y0_dynamic, ts_dynamic, method='milstein',adaptive=True).squeeze(-1)
                # 计算每个时间点的N值
            N_values = torch.tensor([N_func(t) for t in ts_dynamic])
            dy_data_record['ys_dynamic'] = ys_dynamic
            dy_data_record['N_values'] = N_values
            dy_data_record['ts_dynamic'] = ts_dynamic
            torch.save(dy_data_record, file_save_path+"/" + 'SLBP_dynamic_D_' + str(sde_dynamic.params['D']) + '.pt')
            return ts_dynamic.numpy(), ys_dynamic.numpy(), N_values.numpy()
        else:
            temp_record=0
            for i in range(int(total_time//max_one_time)):
                print("i",i)
                temp_ts_dynamic = torch.linspace(max_one_time*i,  max_one_time*(i+1),   int(max_one_time / dt),dtype=torch.float64)

                with torch.no_grad():
                    temp_ys_dynamic = torchsde.sdeint(sde_dynamic, y0_dynamic, temp_ts_dynamic, method='milstein',adaptive=True).squeeze(-1)
                temp_N_values = torch.tensor([N_func(t ) for t in temp_ts_dynamic])

                torch.save({"ys_dynamic":temp_ys_dynamic,"N_values":temp_N_values,"ts_dynamic":temp_ts_dynamic}, file_save_path+"/"+'SLBP_dynamic_temp'+str(i)+'.pt')

                y0_dynamic = temp_ys_dynamic[-1].reshape(-1, 1)
                del temp_ts_dynamic, temp_ys_dynamic, temp_N_values
                temp_record+=1
            if total_time%max_one_time!=0:
                temp_ts_dynamic = torch.linspace(max_one_time*temp_record,  max_one_time*temp_record+total_time%max_one_time,   int(max_one_time / dt),dtype=torch.float64)
                with torch.no_grad():
                    temp_ys_dynamic = torchsde.sdeint(sde_dynamic, y0_dynamic, temp_ts_dynamic, method='milstein',
                                                      adaptive=True).squeeze(-1)
                temp_N_values = torch.tensor([N_func(t) for t in temp_ts_dynamic])
                temp_record+=1
                torch.save({"ys_dynamic": temp_ys_dynamic, "N_values": temp_N_values, "ts_dynamic": temp_ts_dynamic},
                           file_save_path + "/" + 'SLBP_dynamic_temp' + str(temp_record) + '.pt')
                del temp_ts_dynamic, temp_ys_dynamic, temp_N_values
            dy_data_record['ys_dynamic'] = []
            dy_data_record['N_values'] = []
            dy_data_record['ts_dynamic'] = []

            for i in range(temp_record):
                temp_data = torch.load(file_save_path+"/"+'SLBP_dynamic_temp'+str(i)+'.pt')
                dy_data_record['ys_dynamic'].append(temp_data['ys_dynamic'])
                dy_data_record['N_values'].append(temp_data['N_values'])
                dy_data_record['ts_dynamic'].append(temp_data['ts_dynamic'])
                del temp_data
                os.remove(file_save_path+"/"+'SLBP_dynamic_temp'+str(i)+'.pt')
            print("data_record save")
            dy_data_record['ys_dynamic'] = torch.cat(dy_data_record['ys_dynamic'],dim=0)#T,2
            dy_data_record['N_values'] = torch.cat(dy_data_record['N_values'],dim=0)
            dy_data_record['ts_dynamic'] = torch.cat(dy_data_record['ts_dynamic'],dim=0)
            torch.save(dy_data_record, file_save_path + "/" + 'SLBP_dynamic_D_' + str(sde_dynamic.params['D']) + '.pt')
            return dy_data_record['ts_dynamic'].to(dtype=torch.float32).numpy(), dy_data_record['ys_dynamic'].to(dtype=torch.float32).numpy(), dy_data_record['N_values'].to(dtype=torch.float32).numpy()



# 可视化函数
def plot_dynamic_trajectory(ts, ys, N_values,file_save_path,save_name='dynamic_trajectory'):
    print("plot_dynamic_trajectory")
    fig = plt.figure(figsize=(15, 8))

    # 1. 状态变量X随时间变化
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(ts, ys[:, 0], 'b-', label='Bream (X)', linewidth=2)
    ax1.set_ylabel('Population Density', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)



    if np.min(N_values) < 1.34 < np.max(N_values):
        t_trans1 = ts[np.argmin(np.abs(N_values - 1.34))]
        ax1.axvline(x=t_trans1, color='purple', linestyle='--', label='Turbid Emerges (N=1.34)')

    if np.min(N_values) < 3.04 < np.max(N_values):
        t_trans2 = ts[np.argmin(np.abs(N_values - 3.04))]
        ax1.axvline(x=t_trans2, color='cyan', linestyle='--', label='Clear Disappears (N=3.04)')
    #  状态变量Y随时间变化
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(ts, ys[:, 1], 'r-', label='Pike (Y)', linewidth=2)
    ax2.set_ylabel('Population Density', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(alpha=0.3)


    if np.min(N_values) < 1.34 < np.max(N_values):
        t_trans1 = ts[np.argmin(np.abs(N_values - 1.34))]
        ax2.axvline(x=t_trans1, color='purple', linestyle='--', label='Turbid Emerges (N=1.34)')

    if np.min(N_values) < 3.04 < np.max(N_values):
        t_trans2 = ts[np.argmin(np.abs(N_values - 3.04))]
        ax2.axvline(x=t_trans2, color='cyan', linestyle='--', label='Clear Disappears (N=3.04)')
    # 2. 营养水平随时间变化
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(ts, N_values, 'g-', label='Nutrient Level (N)', linewidth=2)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Nutrient Level', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(alpha=0.3)



    plt.tight_layout()
    plt.savefig(file_save_path + '/' + save_name+".png", dpi=300)
    #plt.show()

def grid_dy_params(dy_params_dict):
    dy_params_list=[]
    for dy_params_value in it.product(*dy_params_dict.values()):
        dy_params = dict(zip(dy_params_dict.keys(), dy_params_value))
        dy_params_list.append(dy_params)
    return dy_params_list



# 主程序
if __name__ == "__main__":
    # 配置参数
    burn_time = 1000  # 预热期时长

    #total_times = [5e5,1e6,5e6,1e7] # 动态变化期时长[5e5,1e6,5e6,1e7]
    total_times = [1e6,2e6,3e6]
    N_min = 0.0
    N_max = 3.5

    # 定义营养水平随时间变化的函数
    def N_func_increase(t):
        """营养水平随时间线性增加"""

        return N_min + (N_max-N_min) * (t / total_time)  #
    def N_func_decrease(t):
        """营养水平随时间线性减少"""
        return N_max - (N_max-N_min) * (t / total_time)  #

 #   params_trend="decrease"#increase or decrease
    for params_trend in ["increase","decrease"]:
        if params_trend=="increase":
            N_func=N_func_increase
            N_init=N_min
        elif params_trend=="decrease":
            N_init=N_max
            N_func=N_func_decrease
        else:
            raise ValueError("params_trend should be 'increase' or 'decrease'")
        # 带预热期的动态模拟
        results_save_path = "SLBP_model_data_test"

        dy_params_list=grid_dy_params(dy_params_dict)
        for total_time in total_times:
            file_save_path =results_save_path+"/"+ "SLBP_dynamic_total_time_{}_N_{}".format(total_time,params_trend)
            if os.path.exists(file_save_path):

                print("{} already exists, skip".format(file_save_path))

            else:
                os.makedirs(file_save_path)
            for dy_params in dy_params_list:
                ts, ys, N_values = simulate_with_burn_in(N_func=N_func,params=dy_params,N_init=N_init,total_time= total_time,burn_time= burn_time,file_save_path=file_save_path)

                # 可视化结果
                plot_dynamic_trajectory(ts, ys, N_values,file_save_path,save_name='SLBPmodel_D_' + str(dy_params['D']))

