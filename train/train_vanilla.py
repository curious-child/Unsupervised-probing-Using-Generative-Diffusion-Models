import os
import json
from torch_geometric.utils import degree
import torch
from torch_geometric.loader import  DataLoader
from utils.utils import save_checkpoint
from loss_functions.loss_functions import train_loss,evaluation_score,loss_wrapper
from optimizers.optimizers import train_optimizers,train_schedulers
from models.models import graph_models

def run_training( trainset, validationset,train_param,net_param,loss_param,optimizer_param,records_path):
    # 评价指标 记录变量初始化
    record_scores={
        "epoch":list(),
    }
    evaluation_metrics=dict()
    train_score = dict()
    val_score = dict()
    # 加载数据集

    train_loader = DataLoader(trainset, batch_size=train_param["batch_size"], shuffle=True, drop_last=False)
    val_loader   = DataLoader(validationset, batch_size=train_param["batch_size"], shuffle=False, drop_last=False)
    #设置训练模型参数、优化器参数、损失函数参数
    if net_param["prelayers_gnn"]=="PNAConv"  :
        net_param["prelayers_gnn_param"]["deg"]=deg_histogram(trainset=trainset)
    train_GNN = graph_models(gnn_name=net_param["gnn_name"],net_param=net_param).to(train_param["device"])
    optimizer = train_optimizers(train_GNN.parameters(),optimizer_param)
    if optimizer_param["scheduler_set"]==True:
        scheduler= train_schedulers(optimizer,optimizer_param)
    criterion=train_loss(loss_metric=loss_param["loss_metric"],loss_param=loss_param).to(train_param["device"])
    #设置模型性能评价指标参数
    for score_metric in train_param["score_metrics"]:
        evaluation_metrics[score_metric]=evaluation_score(score_metric=score_metric)
        record_scores[score_metric]={
                                    "train_scores":list(),
                                    "val_scores":list()
                                    }
    #开始训练
    for epoch in range(train_param["train_epochs"]):
        train_GNN.train()
        # 模型性能评价指标值 初始化
        for score_metric in train_param["score_metrics"]:
            train_score[score_metric] = 0
            val_score[score_metric]=0

        #批量训练
        for n,data in enumerate(train_loader):
            optimizer.zero_grad()  # Clear gradients.
            data = data.to(train_param["device"])
            if not loss_param["loss_metric"]=="CEPLoss":
                out = train_GNN(data).squeeze() #regression task
            else:
                out=data
            loss=loss_wrapper(loss_metric=loss_param["loss_metric"],
                              criterion=criterion,
                              input=out,target=data.y,train_GNN=train_GNN,
                              epoch=epoch,iter=n,
                             device=train_param["device"])
            loss.backward()
            optimizer.step()
            # record evaluation metrics for machine learning tool
            with torch.no_grad():
                for score_metric in train_param["score_metrics"]:
                    if score_metric in ["rank_accuracy","kendall_rank"]:
                        score = evaluation_metrics[score_metric](data.y.cpu(), out.cpu(), data.num_graphs, data)
                        train_score[score_metric] = (score + n * train_score[score_metric]) / (n + 1)
                    elif score_metric=="loss_metric":
                        train_score[score_metric] = (loss.cpu().item() + n * train_score[score_metric]) / (n + 1)
                    else:
                        score=evaluation_metrics[score_metric](data.y.cpu(),out.cpu()).item()
                        train_score[score_metric] = (score+n*train_score[score_metric])/(n+1)
        if optimizer_param["scheduler_set"] == True:
            scheduler.step()
        print("\rtrain_epoch:{}".format(epoch),end=" ")
        #validation
        train_GNN.eval()
        with torch.no_grad():
            for n,data in enumerate(val_loader):
                data = data.to(train_param["device"])
                if not loss_param["loss_metric"] == "CEPLoss":
                    out = train_GNN(data).squeeze()  # regression task
                else:
                    out = data
                loss = loss_wrapper(loss_metric=loss_param["loss_metric"],
                                    criterion=criterion,
                                    input=out, target=data.y, train_GNN=train_GNN,
                                    epoch=epoch, iter=n,
                                    device=train_param["device"]).detach()
                # record evaluation metrics for machine learning tool
                for score_metric in train_param["score_metrics"]:
                    if score_metric in ["rank_accuracy","kendall_rank"]:
                        score = evaluation_metrics[score_metric](data.y.cpu(), out.cpu(), data.num_graphs, data)
                        val_score[score_metric] = (score + n * val_score[score_metric]) / (n + 1)
                    elif score_metric=="loss_metric":
                        val_score[score_metric] = (loss.cpu().item() + n * val_score[score_metric]) / (n + 1)
                    else:
                        score = evaluation_metrics[score_metric](data.y.cpu(), out.cpu()).item()
                        val_score[score_metric] = (score + n * val_score[score_metric]) / (n + 1)
        #save related data in training process
        record_scores["epoch"].append(epoch)
        for score_metric in train_param["score_metrics"]:
            record_scores[score_metric]["train_scores"].append(train_score[score_metric])
            record_scores[score_metric]["val_scores"].append(val_score[score_metric])
        #save model parameters in training process
        if epoch%train_param["ckpt_period"]==0 and epoch!=0 and train_param["ckpt"]:
            sckpt_path =os.path.join(records_path,"ckpt")
            if os.path.exists(sckpt_path):
                print("ckpt文件夹目录已存在")
                pass
            else:
                os.mkdir(sckpt_path)
            save_checkpoint(path=sckpt_path,model_name="{}_metrics_model_{}iter".format(trainset[0].y_name,epoch),model=train_GNN,net_param=net_param)
    #save fininal trainde model 文件夹可略去
    model_path = os.path.join(records_path,"trained_model")
    if os.path.exists(model_path):
        print("trained_model文件夹目录已存在")
    else:
        os.mkdir(model_path)
    save_checkpoint(path=model_path, model_name="{}_metrics_model_trained".format(trainset[0].y_name), model=train_GNN,
                    net_param=net_param)
    #保存训练模型性能指标数据 文件夹可略去
    data_path = os.path.join(records_path, "train_trace")
    if os.path.exists(data_path):
        print("warning:train_trace文件夹目录已存在")
    else:
        os.mkdir(data_path)
    with open(data_path+'/record_scores.json', 'w') as f:
        json.dump(record_scores, f, indent=4, separators=(',', ':'))
    return record_scores


def deg_histogram(trainset):
    max_degree = -1
    for data in trainset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in trainset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg




