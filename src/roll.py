import torch
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

def roll_loss_from_fpr(fpr):
    def _partial(yh, y):
        true_yh, false_yh = split_true_false(yh, y)
        true_normal = _torch_normal_fit(true_yh)
        false_normal = _torch_normal_fit(false_yh)
        return true_normal.cdf(false_normal.icdf(torch.tensor(1 - fpr)))
    return _partial
