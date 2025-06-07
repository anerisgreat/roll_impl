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

def split_true_false_indeces(yh, y):
    true_indeces = torch.argwhere(y)[:,0]
    false_indeces = torch.argwhere(torch.logical_not(y))[:,0]

    return true_indeces, false_indeces

def split_true_false(yh, y):
    true_indeces, false_indeces = split_true_false_indeces(yh, y)

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

import torch
from torch.autograd import Function

class KernelizedROLLoss(Function):
    @staticmethod
    def _sigma_tag_function(var, x):
        return (var * torch.exp(-var*x))/((1 + torch.exp(-var*x))**2)

    @staticmethod
    def _sigma_function(var, x):
        return 1/(1 + torch.exp(-var*x))

    @staticmethod
    def _calc_icdf(theta, var, x):
        #Theta is all the Xs to be kernelized
        #var is the variance of the gaussian kernel
        #x is the pointn at which to calculate
        yt = torch.quantile(theta, x)
        grad = 0

        run_flag = True
        n_iters = 0
        alpha = 1
        while(run_flag and n_iters < 1000):
            diff = x - KernelizedROLLoss._calc_cdf(theta, var, yt)
            yt = yt + alpha * diff
            n_iters += 1

            run_flag = not (torch.abs(diff) < 1e-3)
        if(n_iters == 1000):
            logging.warning(f'Errors reached, diff is still {diff}')
        return yt

    def _calc_true_deriv(vart, a, xt):
        return -(1/torch.tensor(xt.shape[0]))*KernelizedROLLoss._sigma_tag_function(vart, xt - a)

    def _calc_thresh_deriv(vart, a, xt):
        return -np.sum(KernelizedROLLoss._calc_true_deriv(vart, a, xt))

    #Note need to multiply by threshold deriv for full effect
    def _calc_false_deriv(varf, a, xf):

        # dfleftsingle = (1/xf.shape[0])*varf*(torch.exp(varf*(a - xf))/((torch.exp(varf*(a - xf)) + 1)**2))
        dfleftsingle = varf*(torch.exp(varf*(xf - a))/((torch.exp(varf*(xf - a)) + 1)**2))
        dfleft = torch.sum(dfleftsingle) - dfleftsingle

        sigmoids = KernelizedROLLoss._sigma_function(varf, xf - a)
        rightgroup = 1/(sigmoids - sigmoids**2)
        return (1/((dfleft*rightgroup)/varf + 1))

    @staticmethod
    def _calc_cdf(theta, var, x):
        #Theta is all the Xs to be kernelized
        #var is the variance of the gaussian kernel
        #x is the pointn at which to calculate

        return torch.mean(
            KernelizedROLLoss._sigma_function(var, x - theta)
        )

    @staticmethod
    def forward(ctx, target_fpr, yh, y):
        # Compute the loss
        true_indeces, false_indeces = split_true_false_indeces(yh, y)
        true_yh = yh[true_indeces]
        false_yh = yh[false_indeces]
        vart = torch.minimum(1/(torch.var(true_yh)), torch.tensor(10))
        varf = torch.minimum(1/(torch.var(false_yh)), torch.tensor(10))
        # vart = 1/torch.sqrt(torch.var(true_yh))
        # varf = 1/torch.sqrt(torch.var(false_yh))

        # varf = torch.tensor(1)
        # vart = torch.tensor(1)

        # vart = torch.sqrt(torch.tensor(len(true_yh)))
        # varf = torch.sqrt(torch.tensor(len(false_yh)))
        icdf_false = KernelizedROLLoss._calc_icdf(false_yh, varf, target_fpr)
        loss = KernelizedROLLoss._calc_cdf(true_yh, vart, icdf_false)

        ctx.save_for_backward(y, yh, true_yh, false_yh, \
                              true_indeces, false_indeces, icdf_false, loss,
                              vart, varf)

        return loss

    @staticmethod
    def backward(ctx, grad_output) :
        y, yh, true_yh, false_yh, true_indeces, false_indeces, \
            icdf_false, loss, vart, varf = ctx.saved_tensors

        true_grads = KernelizedROLLoss._calc_true_deriv(
            vart, icdf_false, true_yh)
        thresh_grad = -torch.sum(true_grads)
        false_grads = thresh_grad * KernelizedROLLoss._calc_false_deriv(
            varf, icdf_false, false_yh)
        grad_input = torch.zeros(yh.shape[0])
        grad_input[true_indeces] = true_grads
        grad_input[false_indeces] = false_grads * grad_output
        grad_target = None
        grad_fpr = None
        if(torch.any(torch.isnan(grad_input))):
            logging.error('NAN GRADIENT')
        return grad_fpr, grad_input, grad_target

def _kernelized_roll_fpr_internal(yh, y, fpr):
    return KernelizedROLLoss.apply(fpr, yh, y)

def kernelized_roll_fpr(fpr):
    return partial(_kernelized_roll_fpr_internal,
                   fpr = 1 - fpr)
