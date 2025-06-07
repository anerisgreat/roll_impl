import torch
import numpy as np
from copy import deepcopy, copy
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
import logging
from sklearn.metrics import roc_curve
from functools import partial
import torch.multiprocessing as mp
import time
from scipy.stats import beta

from .utils import joinmakedir
from .summary import summarize_episode, summarize_all_episodes, summarize_all_configurations

N_MP_WORKERS = 4

@dataclass
class ModelDataResult:
    criteria : pd.DataFrame
    y : np.array
    yh : np.array

@dataclass
class EpisodeResult:
    best_model : torch.nn.Module
    split_results : Dict[str, ModelDataResult] = field(default_factory = {})
    split_criteria_epochs : Dict[str, pd.DataFrame] = field(default_factory = {})

@dataclass
class MultiEpisodeResult:
    episode_reults : list

@dataclass
class MultiEpisodeResult:
    episode_results : list

    def get_split_scores(self):
        split_names = self.episode_results[0].split_results.keys()
        return {split_name : tuple(zip(*[
                    (ep_res.split_results[split_name].y, ep_res.split_results[split_name].yh) \
                    for ep_res in self.episode_results])) \
                            for split_name in split_names}

class Criteriorator(ABC):
    @abstractmethod
    def init_episode(self):
        raise NotImplemented

    @abstractmethod
    def get_stop_best_flags(self, train_crit, val_crit):
        raise NotImplemented

    def gen_criteria(self, yh, y):
        ret = self._gen_criteria_func(yh, y)
        logging.debug(', '.join([f'{c}: {ret[c][0]}' for c in ret.columns]))
        return ret

    @abstractmethod
    def _gen_criteria_func(self, yh, y):
        raise NotImplemented

    @property
    @abstractmethod
    def loss_func(self):
        return self._loss_func

class BasicCriteriorator(Criteriorator):
    def __init__(self, loss_func, max_iters):
        self._loss_func = loss_func
        self._max_iters = max_iters

    def init_episode(self):
        self._n_iters = 0
        self._best_loss = np.inf

    def loss_func(self, yh, y):
        return self._loss_func(yh, y)

    def _get_loss(self, yh, y):
        loss = self._loss_func(yh, y).detach().numpy()
        if(isinstance(loss, np.ndarray)):
            loss = np.mean(loss)
        return loss

    def _gen_criteria_func(self, yh, y):
        loss = self._get_loss(yh, y)
        return pd.DataFrame({'loss' : [loss]})

    def get_stop_best_flags(self, train_crit, val_crit):
        self._n_iters += 1

        newloss = val_crit.at[0, 'loss']
        best_flag = newloss < self._best_loss
        if(best_flag):
            self._best_loss = newloss
        stop_flag = self._n_iters >= self._max_iters

        return stop_flag, best_flag

class CRBasedCriteriorator(Criteriorator):
    def __init__(self, loss_func, max_iters, fprs):
        self._loss_func = loss_func
        self._max_iters = max_iters
        self._fprs = fprs

    def init_episode(self):
        self._n_iters = 0
        self._best_loss = np.inf
        self._best_crs = [0 for _ in self._fprs]

    def loss_func(self, yh, y):
        return self._loss_func(yh, y)

    def _get_loss(self, yh, y):
        loss = self._loss_func(yh, y).detach().numpy()
        if(isinstance(loss, np.ndarray)):
            loss = np.mean(loss)
        return loss

    def _gen_criteria_func(self, yh, y):
        loss = self._get_loss(yh, y)
        ret = {'loss' : [loss]}
        #TODO HERE
        # tprs = get_tpr_at_fprs(yh, y, fprs)
        tprs = dict(zip([f'tpr@{fpr:0.2f}' for fpr in self._fprs],
                        get_tpr_at_fprs(yh, y, self._fprs)))
        tfrs = dict(zip([f'tfr@{fpr:0.2f}' for fpr in self._fprs],
                        [1 - x for x in get_tpr_at_fprs(yh, y, self._fprs)]))
        # beta_tfrs = dict(zip([f'beta-tfr@{fpr:0.2f}' for fpr in self._fprs],
        #                 [x for x in get_beta_fpr_at_fprs(yh, y, self._fprs)]))

        # ret_df = pd.DataFrame({**ret, **tprs, **tfrs, **beta_tfrs})
        ret_df = pd.DataFrame({**ret, **tprs, **tfrs})
        logging.debug(ret_df)
        return ret_df

    def get_stop_best_flags(self, train_crit, val_crit):
        self._n_iters += 1

        newloss = val_crit.at[0, 'loss']
        best_flag = newloss < self._best_loss
        if(best_flag):
            self._best_loss = newloss
        stop_flag = self._n_iters >= self._max_iters

        return stop_flag, best_flag

def split_indeces(indeces, frac_train, frac_val):
    train_val_split = int(len(indeces)*frac_train)
    val_test_split = int(len(indeces)*(frac_train + frac_val))

    return indeces[:train_val_split], \
        indeces[train_val_split:val_test_split], \
        indeces[val_test_split:]

def split_dataset_indeces(dset, frac_train, frac_val):
    y = dset.y

    true_indeces = torch.argwhere(y)[:,0]
    false_indeces = torch.argwhere(torch.logical_not(y))[:,0]

    true_train, true_val, true_test = \
        split_indeces(true_indeces, frac_train, frac_val)
    false_train, false_val, false_test = \
        split_indeces(false_indeces, frac_train, frac_val)

    return \
        torch.cat((false_train, true_train)), \
        torch.cat((false_val, true_val)), \
        torch.cat((false_test, true_test))

def _get_tpr_at_fpr_internal(fpr, roc_fprs, roc_tprs):
    ind = np.searchsorted(roc_fprs, fpr, side = 'left')
    return roc_tprs[ind]

def get_tpr_at_fprs(yh, y, fprs):
    yh = np.squeeze(yh.detach().numpy())
    y = np.squeeze((y > 0.).detach().numpy())

    yh_false = yh[np.argwhere(y == 0)]
    yh_true = yh[np.argwhere(y > 0)]
    yh_false_sorted = np.sort(yh_false, axis = None)

    ret = []
    for fpr in fprs:
        thresh = yh_false_sorted[int(len(yh_false_sorted) * (1 - fpr))]
        ret.append(np.sum(yh_true >= thresh) / len(yh_true))

    return ret

    # roc_fprs, roc_tprs, _ = roc_curve((y > 0.).detach().numpy(), yh.detach().numpy())
    # return [_get_tpr_at_fpr_internal(fpr, roc_fprs, roc_tprs) \
    #         for fpr in fprs]

def get_beta_fpr_at_fprs(yh, y, fprs):
    yh = np.squeeze(yh.detach().numpy())

    yh_sigm = 1/(1 + np.exp(-yh))


    y = np.squeeze((y > 0.).detach().numpy())

    yh_false = yh_sigm[np.argwhere(y == 0)]
    yh_true = yh_sigm[np.argwhere(y > 0)]

    try:
        b_false = beta.fit(yh_false)
        b_true = beta.fit(yh_true)

        _get_val = lambda fpr: beta.cdf(beta.isf(1-fpr, *b_false), *b_true)
        return [_get_val(fpr) for fpr in fprs]
    except:
        logging.info('Non convergence of fit')
        return [0 for _ in fprs]

def get_tpr_at_fpr(yh, y, fpr):
    return get_tpr_at_fprs(yh, y, [fpr])[0]

class ExperimentDataLoader:
    def __init__(
            self,
            dset, indeces,
            batch_size = 16, is_balanced = False, is_oneshot = False,
            is_shuffle = True):
        self._dset = dset
        self._indeces = indeces
        self._batch_size = batch_size
        self._is_balanced = is_balanced
        self._is_oneshot = is_oneshot
        self._is_shuffle = is_shuffle

        self._true_indeces = torch.argwhere(
                self._dset[self._indeces][1]
            ).flatten()
        self._false_indeces = torch.argwhere(
                torch.logical_not(self._dset[self._indeces][1])
            ).flatten()

    def __iter__(self):
        if(self._is_oneshot):
            yield self._dset[self._indeces]

            return StopIteration

        if(self._is_shuffle):
            self._indeces = self._indeces[torch.randperm(len(self._indeces))]
            self._true_indeces = self._indeces[torch.argwhere(self._dset[self._indeces][1]).flatten()]
            self._false_indeces = self._indeces[torch.argwhere(
                torch.logical_not(self._dset[self._indeces][1])).flatten()]

        self._true_index = 0
        self._false_index = 0
        self._index = 0

        if(self._is_balanced):
            n_true = len(self._true_indeces)
            n_false = len(self._false_indeces)
            n_true_per_batch = int((n_true / (n_true + n_false)) * self._batch_size)
            n_false_per_batch = self._batch_size - n_true_per_batch

            while(self._true_index <= n_true - n_true_per_batch and \
                  self._false_index <= n_false - n_false_per_batch):
                false_Xs, false_Ys = self._dset[self._false_indeces[self._false_index:self._false_index + n_false_per_batch]]
                # false_Xs, false_Ys = zip(*[self._dset[self._false_indeces[j]] \
                #            for j in range(self._false_index,
                #                           self._false_index + n_false_per_batch) \
                #            if not j >= len(self._false_indeces)])
                self._false_index += n_false_per_batch

                
                true_Xs, true_Ys = self._dset[self._true_indeces[self._true_index:self._true_index + n_true_per_batch]]
                # true_Xs, true_Ys = zip(*[self._dset[self._true_indeces[j]] \
                #            for j in range(self._true_index,
                #                           self._true_index + n_true_per_batch) \
                #            if not j >= len(self._true_indeces)])
                self._true_index += n_true_per_batch

                stacked_Xs = torch.cat((true_Xs, false_Xs))
                stacked_Ys = torch.cat((true_Ys, false_Ys))
                yield stacked_Xs, stacked_Ys
            return StopIteration

        for i in range(0, len(self._indeces), self._batch_size):
            Xs, Ys = zip(*[self._dset[self._indeces[j]] \
                           for j in range(i, i + self._batch_size) \
                           if not j >= len(self._indeces)])
            yield torch.stack(Xs, 0), torch.stack(Ys, 0)

        return StopIteration

def basic_data_splitter(dset, is_oneshot = False, batch_size = 128, is_balanced = True):
    train_indeces, val_indeces, test_indeces = \
        split_dataset_indeces(dset, 0.33, 0.33)
    return \
        ExperimentDataLoader(
            dset, train_indeces, batch_size = batch_size, is_shuffle = True, \
            is_oneshot = is_oneshot, is_balanced = is_balanced), \
        ExperimentDataLoader(
            dset, val_indeces, batch_size = batch_size, is_shuffle = False, \
            is_oneshot = is_oneshot, is_balanced = is_balanced), \
        ExperimentDataLoader(
            dset, test_indeces, batch_size = batch_size, is_shuffle = False, \
            is_oneshot = is_oneshot, is_balanced = is_balanced)

def _single_run_dset(loader, model, optim, criteriorator, device, is_train,
                     return_outputs= False):
    loss_list = []
    yh_list = []
    y_list = []

    for bx, by in loader:
        if(is_train):
            optim.zero_grad()
        bx.to(device)
        byh = model(bx)
        loss = criteriorator.loss_func(byh, by)
        if(is_train):
            loss.backward()
            optim.step()

        loss_list.append(loss.item())
        yh_list.append(byh)
        y_list.append(by)

    all_yh = torch.cat(yh_list)
    all_y = torch.cat(y_list)

    crit = criteriorator.gen_criteria(all_yh, all_y)
    if(return_outputs):
        return crit, all_y, all_yh
    return crit

def _single_epoch(train_loader, val_loader,
                  model, optim, criteriorator, device):
    batch_train_losses = []

    train_crit = _single_run_dset(
        train_loader, model, optim, criteriorator, device, is_train = True)
    val_crit = _single_run_dset(
        train_loader, model, optim, criteriorator, device, is_train = False)

    stop_flag, best_flag = criteriorator.get_stop_best_flags(
        train_crit = train_crit, val_crit = val_crit)
    return stop_flag, best_flag, train_crit, val_crit

def _get_model_data_result(loader, model, criteriorator, device):
    crit, y, yh = _single_run_dset(
        loader, model, optim = None, criteriorator = criteriorator,
        device = device,
        is_train = False, return_outputs = True)
    return ModelDataResult(criteria = crit,
                           y = np.squeeze(y.detach().numpy()),
                           yh = np.squeeze(yh.detach().numpy()))

def _perform_episode(
        summary_dir,
        data_loaders,
        logger, device, config):
    train_losses = []
    val_losses = []

    model = config.model_creator_func()
    model.to(device)
    optim = config.optim_class(params = model.parameters(), **config.optim_args)
    epoch_num = 0

    run_flag = True
    best_model = deepcopy(model)
    criteriorator = copy(config.criteriorator)
    criteriorator.init_episode()
    train_crits = []
    val_crits = []
    while(run_flag):
        stop_flag, best_flag, train_crit, val_crit = \
            _single_epoch(data_loaders['train'],
                        data_loaders['val'], model,
                        optim, criteriorator, device)
        train_crits.append(train_crit)
        val_crits.append(val_crit)
        run_flag = not stop_flag
        if(best_flag):
            best_model = deepcopy(model)

    train_crits = pd.concat(train_crits, ignore_index = True)
    val_crits = pd.concat(val_crits, ignore_index = True)
    results = {
        name: _get_model_data_result(loader, best_model, criteriorator, device) \
        for name, loader in data_loaders.items()}

    ep_res = EpisodeResult(
        split_results = results,
        split_criteria_epochs = {'train' : train_crits, 'val' : val_crits},
        best_model = best_model)

    summarize_episode(summary_dir, ep_res, config)

    return ep_res

def _perform_multiple_episodes(
        summary_dir, dataset, device, config, is_mp):
    episode_results = []
    def _gen_episodes():
        for episode_index in range(config.n_episodes):
            logging.info(f'{episode_index:3d} - {config.name}')
            train_loader, val_loader, test_loader = config.data_splitter(dataset)
            yield [
                joinmakedir(summary_dir, f'{episode_index}'), #summary_dir
                {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}, #data_loaders
                None, #logger
                device, #device
                config] #config

    episode_results = None
    if(is_mp):
        with mp.Pool(N_MP_WORKERS) as mppool:
            result = mppool.starmap_async(
                _perform_episode, _gen_episodes())
            while not result.ready():
                time.sleep(1)
            episode_results = result.get()
    else:
        episode_results = []
        for episode_params in _gen_episodes():
            episode_results.append(_perform_episode(*episode_params))

    multi_ep_result = MultiEpisodeResult(episode_results)
    summarize_all_episodes(summary_dir, multi_ep_result, config)
    return multi_ep_result

@dataclass
class ExperimentConfiguration:
    name : str
    model_creator_func : callable
    optim_class : torch.optim.Optimizer = torch.optim.Adam
    optim_args : Dict = field(default_factory = lambda : {'lr' : 0.001})
    criteriorator : Criteriorator = BasicCriteriorator(
        torch.nn.BCEWithLogitsLoss(), 100)
    data_splitter : callable = basic_data_splitter
    n_episodes : int = 5
    is_mp : bool = False

def run_configurations(summary_dir, conf_list, dataset, device = 'cpu', is_mp = False):
    logging.info('Starting running configurations!')
    conf_res = \
        [_perform_multiple_episodes(
            summary_dir = joinmakedir(summary_dir, c.name),
            dataset = dataset,
            device = device,
            config = c,
            is_mp = is_mp) \
            for c in conf_list]
    summarize_all_configurations(summary_dir, conf_res, conf_list)

oneshot_datasplitter = partial(basic_data_splitter, is_oneshot = True)
