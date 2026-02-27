import os
import itertools as it
import time
from glob import glob

import igraph
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torchsde

from torch_geometric.utils import to_dense_adj, from_networkx
import torch_sparse
from tqdm import tqdm

# 配置计算设备
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # 临时使用CPU
print(f"使用设备: {device}")


class SISDynamicsSDE(torch.nn.Module):
    # 严格遵循图片中的公式
    noise_type = 'diagonal'  # 每个节点独立噪声 (W_i)
    sde_type = 'ito'  # Ito 随机积分

    def __init__(self, d, epsilon,eta,graph_data):
        super().__init__()

        self.d = d
        self.epsilon = epsilon
        self.eta=eta
        # 网络结构处理
        self.edge_index = graph_data.edge_index.to(device)
        self.num_nodes = graph_data.num_nodes

        # 创建密集邻接矩阵用于精确计算
        adj_dense = to_dense_adj(graph_data.edge_index, max_num_nodes=self.num_nodes)[0]
        self.A = adj_dense.to(device)

    # 漂移项 (f) - 图片中的确定性部分
    def f(self, t, x):
        f = -self.d * x
        f += self.epsilon(t.item()) * torch.mul(1 - x, torch.mm(self.A, x))
        return f

    # 扩散项 (g) - 图片中的随机部分
    def g(self, t, x):
        noise = self.eta * torch.randn_like(x)
        return noise  # 限制噪声幅度
def generate_network(net_type, num_nodes,** params):
    """生成指定类型的网络并转换为PyG Data对象"""
    if net_type == 'ER':
        # ER随机网络
        p = params.get('p', 0.1)
        G = nx.erdos_renyi_graph(n=num_nodes, p=p)
    elif net_type == 'BA':
        # BA无标度网络
        m = params.get('m', 3)
        G = nx.barabasi_albert_graph(n=num_nodes, m=m)
    elif net_type == 'WS':
        # WS小世界网络
        k = params.get('k', 4)
        p = params.get('p', 0.1)
        G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)
    else:
        raise ValueError(f"未知网络类型: {net_type}")
    # # 计算网络的拉普拉斯矩阵
    laplacian = np.array(nx.laplacian_matrix(G).todense())
    # 计算图的拉普拉斯矩阵的最大特征值
    lambda_graph = np.abs(np.linalg.eigvals(laplacian))

    lambda_max = np.max(lambda_graph)
    print(f'最大特征值为λ_max={lambda_max:.4f}')
    lambda_min2=sorted(lambda_graph)[1]
    print(f'第二小特征值为λ_min2={lambda_min2:.4f}')
    #第二小特征值


    # print(f'λ_min={nx.algebraic_connectivity(G):.4f}')

    # 转换为PyG Data对象
    pyg_data = from_networkx(G)
    pyg_data.num_nodes = num_nodes

    return pyg_data
def to_gnngraph(g):
    from torch_geometric.data import Data

    x = np.ones((g.vcount(), 1))

    x = torch.from_numpy(x).to(torch.float)

    source_nodes1 = [edge.source for edge in g.es]
    target_nodes1 = [edge.target for edge in g.es]
    source_nodes = source_nodes1 + target_nodes1
    target_nodes = target_nodes1 + source_nodes1
    graph_data=Data(x=x,edge_index=torch.tensor([source_nodes,target_nodes],dtype=torch.long))
    adj_dense = to_dense_adj(graph_data.edge_index, max_num_nodes=graph_data.num_nodes)[0]
    eigenvalues = torch.linalg.eigvals(adj_dense).real
  #  lambda_max = torch.max(eigenvalues).item()
    return graph_data
def simulate_with_burn_in(graph_data,epsilon_func,base_params,trend,epsilon_init=0.0, file_save_path=None,total_time=100.0, burn_time=20, dt=0.1,max_one_time=1e6):
    """
    带预热期的动态模拟
    total_time: 总模拟时间
    burn_time: 预热期时间
    dt: 时间步长
    """
    method = 'euler'  # 固定Euler法'

    # 第一阶段：预热期 (固定N=0.0)
    sde_burn = SISDynamicsSDE(epsilon=lambda t: epsilon_init,**base_params,graph_data=graph_data).to(device)  # 预热期固定N=0.0
    y0_burn = torch.rand(graph_data.num_nodes, 1).to(device)  # 初始状态
    ts_burn = torch.linspace(0, burn_time, int(burn_time / dt))

    with torch.no_grad():
        ys_burn = torchsde.sdeint(sde_burn, y0_burn, ts_burn, method=method,dt=dt).squeeze(-1)#[time_tick,nodes]

    # 第二阶段：动态变化期 (N随时间变化)
    sde_dynamic = SISDynamicsSDE(epsilon=epsilon_func, **base_params, graph_data=graph_data).to(device)  # 动态变化期N随时间变化
    y0_dynamic = ys_burn[-1].reshape(-1, 1).to(device)  #   # 从预热结束状态开始
    dy_data_record={}
    if  file_save_path is not None and os.path.exists(file_save_path+"/"+'SIS_dynamic_eta{}d{}_{}'.format(sde_dynamic.eta,sde_dynamic.d,trend) + '.pt'):
        dy_data_record = torch.load(file_save_path+"/"+'SIS_dynamic_eta{}d{}_{}'.format(sde_dynamic.eta,sde_dynamic.d,trend) + '.pt')
        return dy_data_record['ts_dynamic'].to(dtype=torch.float32).numpy(), dy_data_record['ys_dynamic'].to(dtype=torch.float32).numpy(), dy_data_record['tp_values'].to(dtype=torch.float32).numpy()
    else:
        if max_one_time>=total_time:
            ts_dynamic = torch.linspace(0,  total_time, int(total_time / dt),dtype=torch.float64).to(device)  # 动态变化期时间序列
            with torch.no_grad():
                ys_dynamic = torchsde.sdeint(sde_dynamic, y0_dynamic, ts_dynamic, method=method,dt=dt).squeeze(-1)#[time_tick,nodes]
                # 计算每个时间点的N值
            if torch.max(ys_dynamic.mean(dim=1))>1 or torch.min(ys_dynamic.mean(dim=1))< -0.1 or torch.isnan(ys_dynamic.mean(dim=1)).any():
             #   print("Warning: the mean of ys_dynamic is out of range [0,1]")
                return None,None,None
            tp_values = torch.tensor([epsilon_func(t) for t in ts_dynamic])
            dy_data_record['ys_dynamic'] = ys_dynamic
            dy_data_record['tp_values'] = tp_values
            dy_data_record['ts_dynamic'] = ts_dynamic
            if file_save_path is not None:
                torch.save(dy_data_record, file_save_path+"/" + 'SIS_dynamic_eta{}d{}_{}'.format(sde_dynamic.eta,sde_dynamic.d,trend) + '.pt')
            return ts_dynamic.cpu().numpy(), ys_dynamic.cpu().numpy(), tp_values.cpu().numpy()
        else:
            temp_record=0
            for i in range(int(total_time//max_one_time)):
              #  print("i",i)
                temp_ts_dynamic = torch.linspace(max_one_time*i,  max_one_time*(i+1),   int(max_one_time / dt),dtype=torch.float64).to(device)  # 动态变化期时间序列

                with torch.no_grad():
                    temp_ys_dynamic = torchsde.sdeint(sde_dynamic, y0_dynamic, temp_ts_dynamic, method=method,dt=dt).squeeze(-1)
                temp_tp_values = torch.tensor([epsilon_func(t ) for t in temp_ts_dynamic])
                if torch.max(temp_ys_dynamic.mean(dim=1)) > 1 or torch.min(temp_ys_dynamic.mean(dim=1)) < -0.1 or torch.isnan(temp_ys_dynamic.mean(dim=1)).any():
                 #   print("Warning: the mean of ys_dynamic is out of range [0,1]")
                    for file in sorted(glob(file_save_path + "/*.pt")):
                        if "dynamic_temp" in file:
                            os.remove(file)
                    return None, None, None
                torch.save({"ys_dynamic":temp_ys_dynamic,"tp_values":temp_tp_values,"ts_dynamic":temp_ts_dynamic}, file_save_path+"/"+'dynamic_temp'+str(i)+'.pt')

                y0_dynamic = temp_ys_dynamic[-1].reshape(-1, 1)

                del temp_ts_dynamic, temp_ys_dynamic, temp_tp_values
                temp_record+=1
            if total_time%max_one_time!=0:
                temp_ts_dynamic = torch.linspace(max_one_time*temp_record,  max_one_time*temp_record+total_time%max_one_time,   int(max_one_time / dt),dtype=torch.float64).to(device)  # 动态变化期时间序列
                with torch.no_grad():
                    temp_ys_dynamic = torchsde.sdeint(sde_dynamic, y0_dynamic, temp_ts_dynamic, method=method,
                                                      dt=dt).squeeze(-1)
                temp_tp_values = torch.tensor([epsilon_func(t) for t in temp_ts_dynamic])
                temp_record+=1
                if torch.max(temp_ys_dynamic.mean(dim=1)) > 1 or torch.min(temp_ys_dynamic.mean(dim=1)) < -0.1 or torch.isnan(temp_ys_dynamic.mean(dim=1)).any():
                 #   print("Warning: the mean of ys_dynamic is out of range [-0.1,1]")
                    for file in sorted(glob(file_save_path + "/*.pt")):
                        if "dynamic_temp" in file:
                            os.remove(file)
                    return None, None, None
                torch.save({"ys_dynamic": temp_ys_dynamic, "tp_values": temp_tp_values, "ts_dynamic": temp_ts_dynamic},
                           file_save_path + "/" + 'dynamic_temp' + str(temp_record) + '.pt')
                del temp_ts_dynamic, temp_ys_dynamic, temp_tp_values
            dy_data_record['ys_dynamic'] = []
            dy_data_record['tp_values'] = []
            dy_data_record['ts_dynamic'] = []

            for i in range(temp_record):
                temp_data = torch.load(file_save_path+"/"+'dynamic_temp'+str(i)+'.pt')
                dy_data_record['ys_dynamic'].append(temp_data['ys_dynamic'])
                dy_data_record['tp_values'].append(temp_data['tp_values'])
                dy_data_record['ts_dynamic'].append(temp_data['ts_dynamic'])
                del temp_data
                os.remove(file_save_path+"/"+'dynamic_temp'+str(i)+'.pt')
           #
            dy_data_record['ys_dynamic'] = torch.cat(dy_data_record['ys_dynamic'],dim=0)#T,2
            dy_data_record['tp_values'] = torch.cat(dy_data_record['tp_values'],dim=0)
            dy_data_record['ts_dynamic'] = torch.cat(dy_data_record['ts_dynamic'],dim=0)
            torch.save(dy_data_record, file_save_path + "/" + 'SIS_dynamic_eta{}d{}_{}'.format(sde_dynamic.eta,sde_dynamic.d,trend) + '.pt')
            return dy_data_record['ts_dynamic'].to(dtype=torch.float32).cpu().numpy(), dy_data_record['ys_dynamic'].to(dtype=torch.float32).cpu().numpy(), dy_data_record['tp_values'].to(dtype=torch.float32).cpu().numpy()

def plot_dynamic_trajectory(ts, ys, N_values,file_save_path,save_name='dynamic_trajectory'):
    print("plot_dynamic_trajectory")
    fig = plt.figure(num=save_name, figsize=(15, 8))

    ys=ys.mean(axis=1)

    # 1. 状态变量X随时间变化
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(ts, ys, 'b-', label='Bream (X)', linewidth=2)
    ax1.set_ylabel('Population Density', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)



    # if np.min(N_values) < 1.28 < np.max(N_values):
    #     t_trans1 = ts[np.argmin(np.abs(N_values - 1.28))]
    #     ax1.axvline(x=t_trans1, color='purple', linestyle='--', label='Turbid Emerges (N=1.34)')
    #
    # if np.min(N_values) < 1.79 < np.max(N_values):
    #     t_trans2 = ts[np.argmin(np.abs(N_values - 1.79))]
    #     ax1.axvline(x=t_trans2, color='cyan', linestyle='--', label='Clear Disappears (N=3.04)')




    # 2. 营养水平随时间变化
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(ts, N_values, 'g-', label='Nutrient Level (N)', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Nutrient Level', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(alpha=0.3)



    plt.tight_layout()
    if file_save_path is not None:
        plt.savefig(file_save_path + '/' + save_name+".png", dpi=300)
    else:
        plt.show()
    #plt.show()
def grid_dy_params(dy_params_dict):
    dy_params_list=[]
    for dy_params_value in it.product(*dy_params_dict.values()):
        dy_params = dict(zip(dy_params_dict.keys(), dy_params_value))
        dy_params_list.append(dy_params)
    return dy_params_list


if __name__ == '__main__':
    graph_file_path="test_graph"
    file_save_path="spdata_sde_SIS"
   # file_save_path=None
    base_params_dict = {

        'd': [0.5],  # 恢复率 0.1,0.5,0.7
        'eta': [1e-4, 1e-6]  # 噪声强度 , 1e-6
    }
    burn_time = 100  # 预热期时长

    total_time = 1000  #
    epsilon_min=0.001
    epsilon_max=0.11
    # 网络参数设置





    # for i, net_type in enumerate(["ER", 'BA', 'WS']):
    #     print(f"\n{'=' * 50}")
    #     print(f"模拟 {net_type} 网络")
    #     print(f"{'=' * 50}")
    #     graph_data = generate_network(net_type, net_params['num_nodes'], **net_params[net_type])
    #     graph_file_path = os.path.join(file_save_path, f"{net_type}_graph")
    for file in tqdm(sorted(glob(graph_file_path+'/'+"*"+".graphml"))):
        try:
            graph_name=file.split("/")[-1].replace(".graphml","")
            print(graph_name)
            # print("\n")
            # print("---" * 10)
            # print(file_graph_name)
            g=igraph.Graph.Read_GraphML(file)
        except Exception:
            raise Exception("There are errors in {}".format(file))
        graph_data = to_gnngraph(g)
       # print("lambda_max",lambda_max)

        graph_file_path = os.path.join(file_save_path, graph_name)
        if not os.path.exists(graph_file_path) :
            os.mkdir(graph_file_path)
        for trend in ["increase","decrease"]:
           # print("trend", trend)
            if trend == "increase":
                epsilon_func = lambda t: epsilon_min + (epsilon_max - epsilon_min) * t / total_time
                epsilon_init = epsilon_min
            elif trend == "decrease":
                epsilon_func = lambda t: epsilon_max - (epsilon_max - epsilon_min) * t / total_time
                epsilon_init = epsilon_max
            else:
                raise ValueError("trend must be 'increase' or 'decrease'")
            for base_params in grid_dy_params(base_params_dict):
              #  start_time = time.time()
                count=0
                while True:
                    ts, ys, tp_values = simulate_with_burn_in(graph_data=graph_data, epsilon_func=epsilon_func, base_params=base_params, epsilon_init=epsilon_init, total_time=total_time,
                                                             burn_time=burn_time,file_save_path=graph_file_path,max_one_time=1000,dt=0.1,trend=trend)
                    count+=1

                    if ys is not None :
                        break
                    if count>10:
                        print(graph_name)
                        print('SIS_dynamic_eta{}d{}_{}'.format(base_params['eta'],base_params['d'],trend))
                        print("Warning: the simulation failed")
                        break
               # print("运行时间", time.time() - start_time)
                # 可视化结果
                # print('SIS_dynamic_eta{}d{}_{}'.format(base_params['eta'],base_params['d'],trend))
                plot_dynamic_trajectory(ts, ys, tp_values, file_save_path=None,save_name='SIS_dynamic_eta{}d{}_{}'.format(base_params['eta'],base_params['d'],trend))