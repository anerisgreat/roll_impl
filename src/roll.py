import torch
import logging
from functools import partial
from .beta_dist import Beta
import numpy as np
def _torch_mean_std(yh):
    yhmean = torch.mean(yh)
    yhstd = torch.std(yh)
    return yhmean, yhstd

def _torch_normal_fit(yh):
    yhmean, yhstd = _torch_mean_std(yh)
    return torch.distributions.normal.Normal(yhmean, yhstd)

def split_true_false(yh, y):
    true_indeces = torch.argwhere(y)[:,0]
    false_indeces = torch.argwhere(torch.logical_not(y))[:,0]

    true_yh = yh[true_indeces]
    false_yh = yh[false_indeces]
    return true_yh, false_yh

def _roll_loss_from_fpr_internal(yh, y, fpr):
    true_yh, false_yh = split_true_false(yh, y)
    true_normal = _torch_normal_fit(true_yh)
    false_normal = _torch_normal_fit(false_yh)
    return true_normal.cdf(false_normal.icdf(torch.tensor(1 - fpr)))

def roll_loss_from_fpr(fpr):
    return partial(_roll_loss_from_fpr_internal, fpr = fpr)

def _roll_beta_loss_from_fpr_internal(yh, y, fpr):
    yh = torch.nn.functional.sigmoid(yh)
    true_yh, false_yh = split_true_false(yh, y)
    true_beta = Beta.from_sample(true_yh)
    false_beta = Beta.from_sample(false_yh)
    return true_beta.cdf(false_beta.icdf(torch.tensor(1 - fpr)))

def roll_beta_loss_from_fpr(fpr):
    return partial(_roll_beta_loss_from_fpr_internal, fpr = fpr)

def roll_beta_aoc_loss(yh, y):
    yh = torch.nn.functional.sigmoid(yh)
    true_yh, false_yh = split_true_false(yh, y)
    true_beta = Beta.from_sample(true_yh)
    false_beta = Beta.from_sample(false_yh)
    r = 0
    for fpr in np.arange(0.001, 1.0, 0.001):
        return true_beta.cdf(false_beta.icdf(torch.tensor(1 - fpr)))
    return r/1000

##This part is non used
#Calc moments
#https://discuss.pytorch.org/t/statistics-for-whole-dataset/74511/2
def _calc_moments(arr):
    mean = torch.mean(array)
    diffs = array - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    return mean, var, skews, kurtoses

#Equations 2.3 in Comparison of parameter estimation methods for normal inverse
#gaussian distribution (Jeongyoen Yoon, Jiyeon Kim, Seongjoo Song)
def _calc_gaussian_norm_inv_params_from_moments(mean, var, skews, kurtoses):
    gamma = 3 / \
        (torch.sqrt(var) * torch.sqrt(3 * kurtoses - 5 * (skews ** 2)))
    beta = (S * torch.sqrt(var) * (gamma**2)) / 3
    alpha = torch.sqrt(gamma**2 + beta**2)
    delta = (var * (gamma**3))/((gamma**2) + (beta**2))
    mu = mean - (beta * delta)/(gamma)

    return (alpha, beta, gamma, delta, mu)

def _gaussian_norm_inv_cdf(theta, x):
    _, _, gamma, delta, mu = theta
    stats.norm.logcdf(-np.sqrt(lambda_/mu) * (x/mu + 1))
