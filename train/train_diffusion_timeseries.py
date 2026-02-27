import os
import json
import time

from torch_geometric.utils import degree
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import  DataLoader
from tqdm import tqdm

from utils.utils import save_checkpoint, load_emergency_checkpoint, emergency_checkpoint
from loss_functions.loss_functions import train_loss,evaluation_score,loss_wrapper
from optimizers.optimizers import train_optimizers,train_schedulers
from models.models import diffusion_models

def run_training( trainset, validationset,train_param,net_param,loss_param,optimizer_param,records_path):
    # 评价指标 记录变量初始化

    # 加载数据集
    trainset_data = torch.cat(trainset, dim=0)  # n*pred_len,F
  #  validationset_data = torch.cat(validationset["data"], dim=0)  # n*pred_len,F

    train_loader = DataLoader(trainset, batch_size=train_param["train_batch_size"], shuffle=True, drop_last=False)
    val_loader   = DataLoader(validationset, batch_size=train_param["val_batch_size"], shuffle=False, drop_last=False)

    #设置训练模型参数、优化器参数、损失函数参数

    train_GNN = diffusion_models(task_model=net_param["task_model"],net_param=net_param,
                                 train_model_select=train_param["train_model_select"],
                                 pretrain_f_path=net_param["pretrain_f_path"],
                                 pretrain_g_path=net_param["pretrain_g_path"],
                                 ).to(net_param["device"])

    if train_GNN.scaler =="StandardScaler":
        train_GNN.scaler_fit(trainset_data)
    if train_param.get("train_model_select") is None or train_param["train_model_select"]=="NsDiff_model":
        optimizer = train_optimizers(filter(lambda p: p.requires_grad,train_GNN.parameters()),optimizer_param)
    elif train_param["train_model_select"]=="pretrain_f":
        optimizer = train_optimizers(train_GNN.cond_pred_model.parameters(), optimizer_param)
    elif train_param["train_model_select"]=="pretrain_g":
        optimizer = train_optimizers(train_GNN.cond_pred_model_g.parameters(), optimizer_param)
    else:
        raise ValueError("train_model_select error")
    if optimizer_param["scheduler_set"]==True:
        scheduler= train_schedulers(optimizer,optimizer_param)
    else:
        scheduler=None

    #设置模型性能评价指标参数
    current_step,record_scores = load_emergency_checkpoint(checkpoint_path=records_path,
                                             model=train_GNN,
                                             optimizer=optimizer,
                                             device=net_param["device"],
                                             scheduler=scheduler)
    init_epoch = current_step

    #开始训练
    try:
        for epoch in range(init_epoch,train_param["train_epochs"]):
            train_GNN.train()
            # 模型性能评价指标值 初始化
            train_score=0
            val_score=0

            #批量训练
            for n,data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                optimizer.zero_grad()  # Clear gradients.
               # print("\r******************train_epoch:{} batch:{}********************\n".format(epoch,n),end=" ")

                if train_GNN.scaler == "StandardScaler":
                    data = train_GNN.scaler_transform(data)
                data = data.to(net_param["device"])
               # time_step=time.time()
                #out: [N,2]
                if train_param["train_model_select"]=="NsDiff_model":
                    loss = train_GNN.training_step(batch=data)  # diffusion training
                elif train_param["train_model_select"]=="pretrain_f":
                    loss = train_GNN.pretrain_f(batch=data)  # pretrain f
                elif train_param["train_model_select"]=="pretrain_g":
                    loss = train_GNN.pretrain_g(batch=data)  # pretrain g
                else:
                    raise ValueError("train_model_select error")
                if  torch.isnan(loss).any():
                    continue

                # time_train=time.time()
                # print("train time_cost:{}".format(time_train-time_step))

                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                train_score=n*train_score/(n+1)+loss.cpu().detach().item()/(n+1)

            if  torch.isnan(loss).any():
                raise ValueError("loss is None")
            if optimizer_param["scheduler_set"] == True:
                scheduler.step()
            current_step = epoch + 1
            # print("\rtrain_epoch:{}".format(epoch),end=" ")
            # print("train_loss:{}".format(train_score))
            #validation
            with torch.no_grad():
                if  train_param["test_set"]:

                    train_GNN.eval()
                    for n, data in enumerate(val_loader):

                        if train_GNN.scaler == "StandardScaler":
                            data = train_GNN.scaler_transform(data)
                        data = data.to(net_param["device"])
                        if train_param["train_model_select"] == "NsDiff_model":
                            loss = train_GNN.training_step(batch=data)  # diffusion training
                        elif train_param["train_model_select"] == "pretrain_f":
                            loss = train_GNN.pretrain_f(batch=data)  # pretrain f
                        elif train_param["train_model_select"] == "pretrain_g":
                            loss = train_GNN.pretrain_g(batch=data)  # pretrain g
                        else:
                            raise ValueError("train_model_select error")


                        val_score = n * val_score / (n + 1) + loss.cpu().detach().item() / (n + 1)


            print("val_loss:{}".format(val_score))
            record_scores["epoch"].append(epoch)

            record_scores["train_scores"].append(train_score)

            record_scores["val_scores"].append(val_score)
            #save model parameters in training process
            if epoch%train_param["ckpt_period"]==0 and epoch!=0 and train_param["ckpt"]:
                sckpt_path =os.path.join(records_path,"ckpt")
                if os.path.exists(sckpt_path):
                    print("ckpt文件夹目录已存在")
                    pass
                else:
                    os.mkdir(sckpt_path)
                if train_param.get("train_model_select") is None or train_param["train_model_select"] == "NsDiff_model":
                    save_checkpoint(path=sckpt_path, model_name="tmpt_model_{}iter".format(epoch), model=train_GNN,
                                    net_param=net_param)
                elif train_param["train_model_select"] == "pretrain_f":
                    save_checkpoint(path=sckpt_path, model_name="tmpt_model_{}iter".format(epoch),
                                    model=train_GNN.cond_pred_model, net_param=net_param)
                elif train_param["train_model_select"] == "pretrain_g":
                    save_checkpoint(path=sckpt_path, model_name="tmpt_model_{}iter".format(epoch),
                                    model=train_GNN.cond_pred_model_g, net_param=net_param)
                else:
                    raise ValueError("train_model_select error")
    except Exception as e:

        if "CUDA out of memory" in str(e):
            # 首先，清理当前可能持有引用的计算图
            loss = None
            # 强制Python的垃圾回收
            import gc
            gc.collect()
            # 清空PyTorch的CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        emergency_checkpoint(train_GNN, net_param, optimizer, scheduler, current_step,
                             record_scores=record_scores,
                             checkpoint_path=records_path)
        data_path = os.path.join(records_path, "train_trace")
        if os.path.exists(data_path):
            print("warning:train_trace文件夹目录已存在")
        else:
            os.mkdir(data_path)
        with open(data_path + '/record_scores.json', 'w') as f:
            json.dump(record_scores, f, indent=4, separators=(',', ':'))
        print(e)
        raise
    #save fininal trainde model 文件夹可略去
    model_path = os.path.join(records_path,"trained_model")
    if os.path.exists(model_path):
        print("trained_model文件夹目录已存在")
    else:
        os.mkdir(model_path)
    if train_param.get("train_model_select") is None or train_param["train_model_select"] == "NsDiff_model":
        save_checkpoint(path=model_path, model_name="model_trained", model=train_GNN,
                        net_param=net_param)
    elif train_param["train_model_select"] == "pretrain_f":
        save_checkpoint(path=model_path, model_name="model_trained", model=train_GNN.cond_pred_model,
                        net_param=net_param)
    elif train_param["train_model_select"] == "pretrain_g":
        save_checkpoint(path=model_path, model_name="model_trained", model=train_GNN.cond_pred_model_g,
                        net_param=net_param)
    else:
        raise ValueError("train_model_select error")
    #保存训练模型性能指标数据 文件夹可略去
    data_path = os.path.join(records_path, "train_trace")
    if os.path.exists(data_path):
        print("warning:train_trace文件夹目录已存在")
    else:
        os.mkdir(data_path)
    with open(data_path+'/record_scores.json', 'w') as f:
        json.dump(record_scores, f, indent=4, separators=(',', ':'))
    return record_scores









