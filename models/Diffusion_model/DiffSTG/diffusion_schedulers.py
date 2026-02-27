"""Schedulers for Denoising Diffusion Probabilistic Models"""
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform

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

  def __init__(self, T, schedule,loss_weight_schedule="constant"):
    # Diffusion steps
    self.T = T
    self.loss_weight_schedule = loss_weight_schedule#"constant" or "logNormal"

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'quad':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0**0.5, bT**5, T)**2

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

    noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
    atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
    assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

    # Sample noise and add to x0
    epsilon = torch.randn_like(x0)
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
    return xt, epsilon

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




class InferenceSchedule(object):
  def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
    self.inference_schedule = inference_schedule
    self.T = T
    self.inference_T = inference_T

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))



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
