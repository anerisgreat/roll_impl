import torch
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd

from .utils import joinmakedir
from .summary import summarize_episode, summarize_all_episodes

@dataclass
class ModelDataResult:
    criteria : pd.DataFrame
    y : np.array
    yh : np.array

@dataclass
class EpisodeResult:
    train_result : ModelDataResult
    val_result : ModelDataResult
    test_result : ModelDataResult

    train_criteria_epochs : pd.DataFrame
    val_criteria_epochs : pd.DataFrame

@dataclass
class MultiEpisodeResult:
    episode_reults : list

@dataclass
class MultiEpisodeResult:
    episodeResults : list

class Criteriorator(ABC):
    @abstractmethod
    def init_episode(self):
        return NotImplemented

    @abstractmethod
    def get_stop_best_flags(self, train_crit, val_crit):
        return NotImplemented

    @abstractmethod
    def gen_criteria(self, yh, y):
        return NotImplemented

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

    def gen_criteria(self, yh, y):
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

def split_indeces(indeces, frac_train, frac_val):
    train_val_split = int(len(indeces)*frac_train)
    val_test_split = int(len(indeces)*(frac_train + frac_val))

    return indeces[:train_val_split], \
        indeces[train_val_split:val_test_split], \
        indeces[val_test_split:]

def split_dataset_indeces(dset, frac_train, frac_val):
    y = dset.y

    true_indeces = torch.argwhere(y)
    false_indeces = torch.argwhere(torch.logical_not(y))

    true_train, true_val, true_test = \
        split_indeces(true_indeces, frac_train, frac_val)
    false_train, false_val, false_test = \
        split_indeces(false_indeces, frac_train, frac_val)

    return \
        torch.cat((false_train, true_train)), \
        torch.cat((false_val, true_val)), \
        torch.cat((false_test, true_test))

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

def basic_data_splitter(dset):
    train_indeces, val_indeces, test_indeces = \
        split_dataset_indeces(dset, 0.33, 0.33)
    return \
        ExperimentDataLoader(
            dset, train_indeces, batch_size = 32, is_shuffle = True), \
        ExperimentDataLoader(
            dset, val_indeces, batch_size = 32, is_shuffle = False), \
        ExperimentDataLoader(
            dset, test_indeces, batch_size = 32, is_shuffle = False)

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
        return crit, all_yh, all_y
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
    return ModelDataResult(criteria = crit, y = y, yh = yh)

def _perform_episode(
        summary_dir,
        model_creator_func,
        train_loader, val_loader, test_loader,
        optim_class, optim_args, criteriorator, logger,
        device):

    train_losses = []
    val_losses = []

    model = model_creator_func()
    optim = optim_class(model.parameters(), **optim_args)
    epoch_num = 0

    run_flag = True
    best_model = deepcopy(model)
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
    train_result = _get_model_data_result(
        train_loader, best_model, criteriorator, device)
    val_result = _get_model_data_result(
        val_loader, best_model, criteriorator, device)
    test_result = _get_model_data_result(
        test_loader, best_model, criteriorator, device)

    ep_res = EpisodeResult(
        train_result = train_result,
        val_result = val_result,
        test_result = test_result,
        train_criteria_epochs = train_crits,
        val_criteria_epochs = val_crits)

    summarize_episode(summary_dir, best_model, ep_res)

    return ep_res

def _perform_multiple_episodes(
        summary_dir, model_creator_func,
        dataset, data_splitter,
        optim_class, optim_args, criteriorator,
        device, n_episodes):
    episode_results = []
    for episode_index in range(n_episodes):
        train_loader, val_loader, test_loader = data_splitter(dataset)
        episode_results.append(_perform_episode(
            summary_dir = joinmakedir(summary_dir, f'{episode_index}'),
            model_creator_func = model_creator_func,
            train_loader = train_loader,
            val_loader = val_loader,
            test_loader = test_loader,
            optim_class = optim_class,
            optim_args = optim_args,
            criteriorator = criteriorator,
            logger = None, device = device))
    summarize_all_episodes(summary_dir, episode_results)
