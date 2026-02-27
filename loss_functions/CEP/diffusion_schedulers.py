"""Schedulers for Denoising Diffusion Probabilistic Models"""
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F


def log_norm(x, mu=0.0, sigma=1.0):
  """
  对数正态分布概率密度函数

  参数：
  x    : 输入值或数组（必须>0）
  mu   : 对数均值（默认0）
  sigma: 对数标准差（必须>0，默认1）

  返回：
  概率密度值，形状与x相同
  """
  # 数值稳定性处理
  x = np.asarray(x, dtype=np.float64)
  sigma = np.maximum(sigma, 1e-8)  # 防止sigma=0
  mask = x > 0  # 仅处理正数

  # 初始化结果数组
  pdf = np.zeros_like(x)

  # 核心计算公式
  log_x = np.log(x[mask])
  exponent = -0.5 * ((log_x - mu) / sigma)** 2
  denominator = sigma * np.sqrt(2 * np.pi) * x[mask]

  pdf[mask] = np.exp(exponent) / denominator
  return pdf

class GaussianDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule="linear",loss_weight_schedule="constant",**kwargs):
    # Diffusion steps
    self.T = T
    self.loss_weight_schedule = loss_weight_schedule#"constant" or "logNormal"

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    self.betabar = np.cumprod(self.beta)
    self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
    self.alphabar = np.cumprod(self.alpha)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0, t):
    # Select noise scales
    x0=2*x0-1#将[0,1]映射到[-1,1]
    x0=x0.unsqueeze(1).unsqueeze(1)#[B*nodes,1,1,1]
    noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
    atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
    assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

    # Sample noise and add to x0
    epsilon = torch.randn_like(x0)
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
    return xt

  def get_loss_weights(self,time_step):
    if self.loss_weight_schedule == "constant":
      time_step = torch.from_numpy(time_step)
      return torch.ones_like(time_step)
    elif self.loss_weight_schedule == "logNormal":
      mu = 0.0
      sigma = 0.5
      atbar=self.alphabar[time_step]
      snr=atbar/(1-atbar)
     # print("snr",snr)
      loss_weight = log_norm(snr, mu, sigma)
     # print("loss_weight", loss_weight)
      return torch.from_numpy(loss_weight)

class CategoricalDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule="linear",loss_weight_schedule="constant",**kwargs):
    # Diffusion steps
    self.T = T
    self.loss_weight_schedule = loss_weight_schedule  # "constant" or "logNormal"
    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones
    #[2,2]
    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    #[T,2,2]
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0, t):
    # Select noise scales
    x0=torch.round(x0).long()
    x0_onehot = F.one_hot(x0, num_classes=2)  # (B*nodes,2)
    x0_onehot=x0_onehot.reshape((x0.shape[0],1,1,2))#[B*nodes,1,1,2]
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
 #   print("Q_bar.shape[0]",Q_bar.shape[0])
    # [节点数,1,1,2]
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
  #  print("sampel xt shape:{}".format(xt.shape))
    return xt[..., 1].clamp(0, 1)
  def get_loss_weights(self,time_step):
  #  print(self.loss_weight_schedule)
    if self.loss_weight_schedule == "constant":
      time_step=torch.from_numpy(time_step)
      return torch.ones_like(time_step)
    elif self.loss_weight_schedule == "logNormal":
      mu = 0.0
      sigma = 0.5
      atbar=self.Q_bar[time_step][0][0]
      snr=atbar/(1-atbar)

      loss_weight = log_norm(snr, mu, sigma)
     # print("loss_weight",loss_weight)
      return torch.from_numpy(loss_weight)

class DiscreteFlowDiffusion(object):
  def __init__(self, T,  loss_weight_schedule="constant",discrete_classes=2,**kwargs):
    self.T = T
    self.loss_weight_schedule = loss_weight_schedule  # "constant" or "logNormal"
    self.S = discrete_classes
  def sample(self,x1, t):
    # x1 (B*nodes,): input orignal data
    # t (B*nodes,)
    # Returns xt (B*nodes, )
    x1=torch.round(x1).long()#将[0,1]映射到{0,1}
    # uniform
    t=1-t/self.T
    xt = x1.clone()
    uniform_noise = torch.randint(0, self.S, size=xt.shape, device=xt.device)
    corrupt_mask = torch.rand((xt.shape[0])) < (1 - t).to(xt.device)
    xt[corrupt_mask] = uniform_noise[corrupt_mask]

    # masking
    # xt = x1.clone()
    # xt[torch.rand((B,D)) < (1 - t[:, None])] = S-1

    return xt
  def discreteflow_time(self,t):
    return 1-t/self.T
  def get_loss_weights(self,time_step):
    if self.loss_weight_schedule == "constant":
      time_step = torch.from_numpy(time_step)
      return torch.ones_like(time_step)
    elif self.loss_weight_schedule == "logNormal":
      mu = 0.0
      sigma = 0.5
      atbar=time_step/(self.T)+1e-5
      snr=atbar**2/(1-atbar)**2
     # print("snr",snr)
      loss_weight = log_norm(snr, mu, sigma)
     # print("loss_weight", loss_weight)
      return torch.from_numpy(loss_weight)
  def get_rate_matrix(self,x1,xt,t):
    # Calculate R_t^*
    # For p(x_t | x_1) > 0 and p(j | x_1) > 0
    # R_t^*(x_t, j | x_1) = Relu( dtp(j | x_1) - dtp(x_t | x_1)) / (Z_t * p(x_t | x_1))
    # For p(x_t | x_1) = 0 or p(j | x_1) = 0 we have R_t^* = 0

    # We will ignore issues with diagnoal entries as later on we will set
    # diagnoal probabilities such that the row sums to one later on.
    S=2# number of classes
    dt_p_vals = dt_p_xt_g_xt(x1, t)  # (B*D, S)
    dt_p_vals_at_xt = dt_p_vals.gather(-1, xt[:, None]).squeeze(-1)  # (B*D)

    # Numerator of R_t^*
    R_t_numer = F.relu(dt_p_vals - dt_p_vals_at_xt[:, None])  # (B*D, S)

    pt_vals = p_xt_g_x1(x1, t)  # (B*D, S)
    Z_t = torch.count_nonzero(pt_vals, dim=-1)  # (B*D)
    pt_vals_at_xt = pt_vals.gather(-1, xt[:, None]).squeeze(-1)  # (B*D)

    # Denominator of R_t^*
    R_t_denom = Z_t * pt_vals_at_xt  # (B*D)

    R_t = R_t_numer / R_t_denom[:, None]  # (B*D, S)

    # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero
    R_t[(pt_vals_at_xt == 0.0)[:, None].repeat(1, S)] = 0.0
    R_t[pt_vals == 0.0] = 0.0
    return R_t #[B*D,S]




def dt_p_xt_g_xt(x1, t):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    S=2
    # uniform
    x1_onehot = F.one_hot(x1, num_classes=S)  # (B*D, S)
    return x1_onehot - (1 / S)

    # masking
    # x1_onehot = F.one_hot(x1, num_classes=S) # (B, D, S)
    # M_onehot = F.one_hot(torch.tensor([S-1]), num_classes=S)[None, :, :] # (1, 1, S)
    # return x1_onehot - M_onehot


def p_xt_g_x1(x1, t):
  # x1 (B, D)
  # t float
  # returns (B, D, S) for varying x_t value
  S=2
  # uniform
  x1_onehot = F.one_hot(x1, num_classes=S)  # (B*D, S)
  return t * x1_onehot + (1 - t) * (1 / S)

  # masking
  # x1_onehot = F.one_hot(x1, num_classes=S) # (B, D, S)
  # M_onehot = F.one_hot(torch.tensor([S-1]), num_classes=S)[None, :, :] # (1, 1, S)
  # return t * x1_onehot + (1-t) * M_onehot
