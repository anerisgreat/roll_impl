import torch
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
import logging
from sklearn.metrics import roc_curve

from .utils import joinmakedir
from .summary import summarize_episode, summarize_all_episodes, summarize_all_configurations

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
        logging.debug(''.join([f'{c}: {ret[c][0]}' for c in ret.columns]))
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
        self._best_loss = np.infty

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
        self._best_loss = np.infty
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
        tprs = get_tpr_at_fprs(yh, y, fprs)
        return pd.DataFrame()

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
    ind = np.searchsorted(roc_fprs, fpr, side = 'right')
    return roc_tprs[ind]

def get_tpr_at_fprs(yh, y, fprs):
    fprs, tprs, _ = zip(*[roc_curve(y > 0., yh) \
                                        for y, yh in zip(y_list, yh_list)])
    return [_get_tpr_at_fpr_internal(fpr, roc_roc_fprs, roc_tprs) \
            for fpr in fprs]

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

        self._true_indeces = torch.argwhere(self._dset[self._indeces][1])
        self._false_indeces = torch.argwhere(
            torch.logical_not(self._dset[self._indeces][1]))

    def __iter__(self):
        if(self._is_oneshot):
            yield self._dset[self._indeces]
            return StopIteration

        if(self._is_shuffle):
            self._indeces = self._indeces[torch.randperm(len(self._indeces))]
            self._true_indeces = torch.argwhere(self._dset[self._indeces][1])
            self._false_indeces = torch.argwhere(
                torch.logical_not(self._dset[self._indeces][1]))

        self._true_index = 0
        self._false_index = 0
        self._index = 0

        if(self._is_balanced):
            return NotImplemented

        for i in range(0, len(self._indeces), self._batch_size):
            Xs, Ys = zip(*[self._dset[self._indeces[j]] \
                           for j in range(i, i + self._batch_size) \
                           if not j >= len(self._indeces)])
            yield torch.stack(Xs, 0), torch.stack(Ys, 0)

        return StopIteration

def basic_data_splitter(dset, is_oneshot = False):
    train_indeces, val_indeces, test_indeces = \
        split_dataset_indeces(dset, 0.33, 0.33)
    return \
        ExperimentDataLoader(
            dset, train_indeces, batch_size = 128, is_shuffle = True, \
            is_oneshot = is_oneshot), \
        ExperimentDataLoader(
            dset, val_indeces, batch_size = 128, is_shuffle = False, \
            is_oneshot = is_oneshot), \
        ExperimentDataLoader(
            dset, test_indeces, batch_size = 128, is_shuffle = False, \
            is_oneshot = is_oneshot)

def _single_run_dset(loader, model, optim, criteriorator, device, is_train,
                     return_outputs= False):
    loss_list = []
    yh_list = []
    y_list = []

    for bx, by in loader:
        if(is_train):
            optim.zero_grad()
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
        train_loader, val_loader, test_loader,
        logger, device, config):

    data_loaders = {
        'train' : train_loader,
        'val' : val_loader,
        'test' : test_loader}
    train_losses = []
    val_losses = []

    model = config.model_creator_func()
    optim = config.optim_class(params = model.parameters(), **config.optim_args)
    epoch_num = 0

    run_flag = True
    best_model = deepcopy(model)
    criteriorator = deepcopy(config.criteriorator)
    criteriorator.init_episode()
    train_crits = []
    val_crits = []
    while(run_flag):
        stop_flag, best_flag, train_crit, val_crit = \
            _single_epoch(train_loader, val_loader, model,
                          optim, criteriorator, device)
        train_crits.append(train_crit)
        val_crits.append(val_crit)
        run_flag = not stop_flag
        if(best_flag):
            best_model = deepcopy(model)

    train_crits = pd.concat(train_crits)
    val_crits = pd.concat(val_crits)
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
        summary_dir, dataset, device, config):
    episode_results = []
    for episode_index in range(config.n_episodes):
        logging.info(f'{episode_index:3d} - {config.name}')
        train_loader, val_loader, test_loader = config.data_splitter(dataset)
        episode_results.append(_perform_episode(
            summary_dir = joinmakedir(summary_dir, f'{episode_index}'),
            train_loader = train_loader,
            val_loader = val_loader,
            test_loader = test_loader,
            logger = None, device = device,
            config = config))
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

def run_configurations(summary_dir, conf_list, dataset, device = 'cpu'):
    logging.info('Starting running configurations!')
    conf_res = \
        [_perform_multiple_episodes(
            summary_dir = joinmakedir(summary_dir, c.name),
            dataset = dataset,
            device = device,
            config = c) \
            for c in conf_list]
    summarize_all_configurations(summary_dir, conf_res, conf_list)
