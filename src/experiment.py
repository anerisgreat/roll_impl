import os
import sys
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
from numpy.typing import NDArray

from .utils import joinmakedir, get_device
from .summary import summarize_episode, summarize_all_episodes, summarize_all_configurations
from .mlflow_tracker import make_mlflow_run

N_MP_WORKERS = 4

@dataclass
class ModelDataResult:
    criteria : pd.DataFrame
    y : NDArray[np.int32]
    yh : NDArray[np.float32]

@dataclass
class EpisodeResult:
    best_model : torch.nn.Module
    split_results : dict[str, ModelDataResult] = field(default_factory = dict)
    split_criteria_epochs : dict[str, pd.DataFrame] = field(default_factory = dict)

@dataclass
class MultiEpisodeResult:
    episode_results : list

    def get_split_scores(self):
        split_names = self.episode_results[0].split_results.keys()
        return {split_name : tuple(zip(*[
                    (ep_res.split_results[split_name].y, ep_res.split_results[split_name].yh) \
                    for ep_res in self.episode_results])) \
                            for split_name in split_names}

class KernelScheduler:
    """Anneals ROLL kernel bandwidth during training.

    Starts kernels at `initial_gamma` × the ISJ estimate (wider = smoother
    loss surface), then halves every `decay_every` epochs until gamma ≤ 1.

    Pass an instance as `kernel_scheduler` to a Criteriorator to enable.
    Leave `kernel_scheduler=None` (default) to disable.
    """
    def __init__(self, initial_gamma: float = 100.0,
                 decay: float = 0.5, decay_every: int = 100):
        self.initial_gamma: float = float(initial_gamma)
        self.decay: float = decay
        self.decay_every: int = decay_every
        self._epoch: int = 0
        self._gamma: float = self.initial_gamma

    def reset(self):
        self._epoch = 0
        self._gamma = self.initial_gamma

    def step(self):
        self._epoch += 1
        if self._gamma > 1.0 and self._epoch % self.decay_every == 0:
            self._gamma = max(1.0, self._gamma * self.decay)
            logging.info(f'KernelScheduler step: epoch={self._epoch}, gamma={self._gamma:.4f}')

    @property
    def gamma(self) -> float:
        return self._gamma


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
    def __init__(self, loss_func, max_iters, max_grad_norm=None,
                 kernel_scheduler: KernelScheduler = None, patience: int = 500,
                 grace_period: int = 0):
        self._loss_func = loss_func
        self._max_iters = max_iters
        self.max_grad_norm = max_grad_norm
        self._kernel_scheduler = kernel_scheduler
        self._patience = patience
        self._grace_period = grace_period

    def init_episode(self):
        self._n_iters = 0
        self._best_loss = np.inf
        self._epochs_without_improvement = 0
        if self._kernel_scheduler is not None:
            self._kernel_scheduler.reset()

    def loss_func(self, yh, y):
        if hasattr(self._loss_func, 'to'):
            self._loss_func.to(yh.device)
        if self._kernel_scheduler is not None:
            return self._loss_func(yh, y, gamma=self._kernel_scheduler.gamma)
        return self._loss_func(yh, y)

    def _get_loss(self, yh, y):
        loss = self._loss_func(yh, y).detach().cpu().numpy()
        if(isinstance(loss, np.ndarray)):
            loss = np.mean(loss)
        return loss

    def _gen_criteria_func(self, yh, y):
        loss = self._get_loss(yh, y)
        return pd.DataFrame({'loss' : [loss]})

    def get_stop_best_flags(self, train_crit, val_crit):
        self._n_iters += 1

        newloss = val_crit.at[0, 'loss']
        if self._n_iters > self._grace_period:
            best_flag = newloss < self._best_loss
            if best_flag:
                self._best_loss = newloss
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1
        else:
            best_flag = False

        stop_flag = (self._n_iters >= self._max_iters or
                     self._epochs_without_improvement >= self._patience)

        if self._kernel_scheduler is not None:
            self._kernel_scheduler.step()

        return stop_flag, best_flag

class CRBasedCriteriorator(Criteriorator):
    def __init__(self, loss_func, max_iters, fprs,
                 kernel_scheduler: KernelScheduler = None, patience: int = 500,
                 grace_period: int = 0):
        self._loss_func = loss_func
        self._max_iters = max_iters
        self._fprs = fprs
        self._kernel_scheduler = kernel_scheduler
        self._patience = patience
        self._grace_period = grace_period

    def init_episode(self):
        self._n_iters = 0
        self._best_loss = np.inf
        self._epochs_without_improvement = 0
        self._best_crs = [0 for _ in self._fprs]
        if self._kernel_scheduler is not None:
            self._kernel_scheduler.reset()

    def loss_func(self, yh, y):
        if hasattr(self._loss_func, 'to'):
            self._loss_func.to(yh.device)
        if self._kernel_scheduler is not None:
            return self._loss_func(yh, y, gamma=self._kernel_scheduler.gamma)
        return self._loss_func(yh, y)

    def _get_loss(self, yh, y):
        loss = self._loss_func(yh, y).detach().cpu().numpy()
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
        if self._n_iters > self._grace_period:
            best_flag = newloss < self._best_loss
            if best_flag:
                self._best_loss = newloss
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1
        else:
            best_flag = False

        stop_flag = (self._n_iters >= self._max_iters or
                     self._epochs_without_improvement >= self._patience)

        if self._kernel_scheduler is not None:
            self._kernel_scheduler.step()

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
    yh = np.squeeze(yh.detach().cpu().numpy())
    y = np.squeeze((y > 0.).detach().cpu().numpy())

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
    yh = np.squeeze(yh.detach().cpu().numpy())

    yh_sigm = 1/(1 + np.exp(-yh))


    y = np.squeeze((y > 0.).detach().cpu().numpy())

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

def get_fpr_at_tpr(yh, y, tpr):
    yh = np.squeeze(yh if isinstance(yh, np.ndarray) else yh.detach().cpu().numpy())
    y  = np.squeeze(y  if isinstance(y,  np.ndarray) else y.detach().cpu().numpy())
    thresh = np.quantile(yh[y > 0], 1 - tpr)
    return float(np.mean(yh[y == 0] >= thresh))

def write_tpr_summary_csv(summary_dir, conf_list, conf_results, tpr):
    """Write per-episode FPR@TPR for each config to summary_dir/tpr_summary.csv."""
    import os
    rows = []
    for config, multi_ep in zip(conf_list, conf_results):
        for ep_idx, ep in enumerate(multi_ep.episode_results):
            test = ep.split_results.get('test')
            if test is None:
                continue
            fpr = get_fpr_at_tpr(test.yh, test.y, tpr)
            rows.append({'config': config.name, 'episode': ep_idx, f'fpr_at_tpr{tpr:.2f}': fpr})
    df = pd.DataFrame(rows)
    path = os.path.join(summary_dir, 'tpr_summary.csv')
    df.to_csv(path, index=False)
    logging.info(f'TPR summary written to {path}')

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

        y_split = self._dset.y[self._indeces]
        self._true_indeces = self._indeces[torch.argwhere(y_split).flatten()]
        self._false_indeces = self._indeces[torch.argwhere(
                torch.logical_not(y_split)).flatten()]

    def __iter__(self):
        if(self._is_oneshot):
            yield self._dset[self._indeces]

            return

        if(self._is_shuffle):
            self._indeces = self._indeces[torch.randperm(len(self._indeces))]
            y_split = self._dset.y[self._indeces]
            self._true_indeces = self._indeces[torch.argwhere(y_split).flatten()]
            self._false_indeces = self._indeces[torch.argwhere(
                torch.logical_not(y_split)).flatten()]

        self._true_index = 0
        self._false_index = 0
        self._index = 0

        if(self._is_balanced):
            n_true = len(self._true_indeces)
            n_false = len(self._false_indeces)
            n_true_per_batch = max(1,
                                   int((n_true / (n_true + n_false)) \
                                       * self._batch_size))
            n_false_per_batch = self._batch_size - n_true_per_batch
            # clamp so a split smaller than batch_size still yields one batch
            n_true_per_batch = min(n_true_per_batch, n_true)
            n_false_per_batch = min(n_false_per_batch, n_false)

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

class ShuffledSplitter:
    """Stratified splitter that shuffles indices before splitting, seeded by call count.

    Each call corresponds to one episode. The seed equals the call count, so
    episode N always produces the same shuffle regardless of which config calls
    it — as long as every config starts from its own fresh instance (call_count=0).
    Create one instance per config via make_shuffled_splitter().
    """
    def __init__(self, is_oneshot=False, batch_size=128, is_balanced=True):
        self.is_oneshot = is_oneshot
        self.batch_size = batch_size
        self.is_balanced = is_balanced
        self._call_count = 0

    def __call__(self, dset):
        seed = self._call_count
        self._call_count += 1

        y = dset.y
        gen = torch.Generator()
        gen.manual_seed(seed)

        true_indeces = torch.argwhere(y)[:, 0]
        false_indeces = torch.argwhere(torch.logical_not(y))[:, 0]

        true_indeces = true_indeces[torch.randperm(len(true_indeces), generator=gen)]
        false_indeces = false_indeces[torch.randperm(len(false_indeces), generator=gen)]

        true_train, true_val, true_test = split_indeces(true_indeces, 0.33, 0.33)
        false_train, false_val, false_test = split_indeces(false_indeces, 0.33, 0.33)

        train_idx = torch.cat((false_train, true_train))
        val_idx   = torch.cat((false_val,   true_val))
        test_idx  = torch.cat((false_test,  true_test))

        kw = dict(batch_size=self.batch_size, is_oneshot=self.is_oneshot,
                  is_balanced=self.is_balanced)
        return (
            ExperimentDataLoader(dset, train_idx, is_shuffle=True,  **kw),
            ExperimentDataLoader(dset, val_idx,   is_shuffle=False, **kw),
            ExperimentDataLoader(dset, test_idx,  is_shuffle=False, **kw),
        )


def make_shuffled_splitter(is_oneshot=False, batch_size=128, is_balanced=True):
    """Return a fresh ShuffledSplitter instance. Call once per config."""
    return ShuffledSplitter(is_oneshot, batch_size, is_balanced)


class LabelNoisyDataset:
    """Wraps a dataset with randomly flipped labels for the given indices.

    Only the specified indices have their labels potentially flipped; all others
    (e.g. test indices) retain original labels.
    """
    def __init__(self, base_dataset, noisy_indices, noise_rate: float, generator: torch.Generator):
        self.x = base_dataset.x
        y = base_dataset.y.clone()
        flip_mask = torch.bernoulli(
            torch.full((len(noisy_indices),), noise_rate), generator=generator
        ).bool()
        y[noisy_indices[flip_mask]] = 1 - y[noisy_indices[flip_mask]]
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NoisySplitter:
    """Picklable data_splitter that flips noise_rate fraction of train and val labels.

    Noise is seeded by episode index (call counter) for per-episode variation while
    remaining consistent across configs. Each config should get its own instance.
    Test split always sees the original clean labels.
    """
    def __init__(self, noise_rate: float = 0.05, is_oneshot=False, batch_size=128, is_balanced=True):
        self.noise_rate = noise_rate
        self.is_oneshot = is_oneshot
        self.batch_size = batch_size
        self.is_balanced = is_balanced
        self._split_cache = None  # (train_idx, val_idx, test_idx)
        self._call_count = 0

    def __call__(self, dset):
        if self._split_cache is None:
            self._split_cache = split_dataset_indeces(dset, 0.33, 0.33)

        train_idx, val_idx, test_idx = self._split_cache
        gen = torch.Generator()
        gen.manual_seed(self._call_count)
        self._call_count += 1

        noisy_indices = torch.cat([train_idx, val_idx])
        noisy_dset = LabelNoisyDataset(dset, noisy_indices, self.noise_rate, gen)

        kw = dict(batch_size=self.batch_size, is_oneshot=self.is_oneshot,
                  is_balanced=self.is_balanced)
        return (
            ExperimentDataLoader(noisy_dset, train_idx, is_shuffle=True, **kw),
            ExperimentDataLoader(noisy_dset, val_idx, is_shuffle=False, **kw),
            ExperimentDataLoader(dset, test_idx, is_shuffle=False, **kw),
        )


def make_noisy_splitter(noise_rate: float = 0.05, is_oneshot=False,
                        batch_size=128, is_balanced=True):
    return NoisySplitter(noise_rate, is_oneshot, batch_size, is_balanced)


class LabelPoisonedDataset:
    """Wraps a dataset, appending N mislabeled copies of positives as negatives.

    Only the positives from `pos_train_indices` are duplicated; the rest of the
    dataset (including val/test indices) is untouched at the tensor level.
    """

    def __init__(self, base_dataset, pos_train_indices, counts: torch.Tensor):
        base_x = base_dataset.x
        base_y = base_dataset.y
        pos_x = base_x[pos_train_indices]
        repeated_x = pos_x.repeat_interleave(counts, dim=0)
        repeated_y = torch.zeros(int(counts.sum().item()), dtype=base_y.dtype)
        self.x = torch.cat([base_x, repeated_x], dim=0)
        self.y = torch.cat([base_y, repeated_y], dim=0)
        self._n_base = len(base_dataset)

    @property
    def _poison_indices(self):
        n_poison = len(self.y) - self._n_base
        return torch.arange(self._n_base, self._n_base + n_poison)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PoisonedSplitter:
    """Picklable data_splitter that injects N mislabeled copies of train and val positives.

    The poisoned copies receive label=0 (false). Test split sees the original dataset only.
    """
    def __init__(self, n_duplicates: int, is_oneshot=False, batch_size=128, is_balanced=True):
        self.n_duplicates = n_duplicates
        self.is_oneshot = is_oneshot
        self.batch_size = batch_size
        self.is_balanced = is_balanced

    def __call__(self, dset):
        train_idx, val_idx, test_idx = split_dataset_indeces(dset, 0.33, 0.33)
        train_true_idx = train_idx[dset.y[train_idx].bool()]
        val_true_idx = val_idx[dset.y[val_idx].bool()]
        train_counts = torch.full((len(train_true_idx),), self.n_duplicates, dtype=torch.long)
        val_counts = torch.full((len(val_true_idx),), self.n_duplicates, dtype=torch.long)
        poisoned = LabelPoisonedDataset(dset, torch.cat([train_true_idx, val_true_idx]),
                                        torch.cat([train_counts, val_counts]))
        n_base = len(dset)
        n_train_poison = int(train_counts.sum().item())
        n_val_poison = int(val_counts.sum().item())
        aug_train_idx = torch.cat([train_idx, torch.arange(n_base, n_base + n_train_poison)])
        aug_val_idx = torch.cat([val_idx, torch.arange(n_base + n_train_poison,
                                                        n_base + n_train_poison + n_val_poison)])
        loader_kw = dict(batch_size=self.batch_size, is_oneshot=self.is_oneshot,
                         is_balanced=self.is_balanced)
        return (
            ExperimentDataLoader(poisoned, aug_train_idx, is_shuffle=True, **loader_kw),
            ExperimentDataLoader(poisoned, aug_val_idx, is_shuffle=False, **loader_kw),
            ExperimentDataLoader(poisoned, test_idx, is_shuffle=False, **loader_kw),
        )


def make_poisoned_splitter(n_duplicates: int, is_oneshot=False,
                           batch_size=128, is_balanced=True):
    return PoisonedSplitter(n_duplicates, is_oneshot, batch_size, is_balanced)


class RandomPoisonedSplitter:
    """Picklable data_splitter with per-episode random poison counts, consistent across configs.

    Each train and val positive receives a count drawn from Uniform(0, n_max) inclusive.
    Counts are seeded by the episode index (call counter), so episode N always produces
    the same counts regardless of which config calls it. Since splits are deterministic,
    two instances of this class will produce identical data for the same episode.
    Each config should get its own instance so the counter starts at 0.
    Test split sees the original dataset only.
    """
    def __init__(self, n_max: int, is_oneshot=False, batch_size=128, is_balanced=True):
        self.n_max = n_max
        self.is_oneshot = is_oneshot
        self.batch_size = batch_size
        self.is_balanced = is_balanced
        self._split_cache = None  # (train_idx, val_idx, test_idx, train_true_idx, val_true_idx)
        self._call_count = 0

    def __call__(self, dset):
        if self._split_cache is None:
            train_idx, val_idx, test_idx = split_dataset_indeces(dset, 0.33, 0.33)
            train_true_idx = train_idx[dset.y[train_idx].bool()]
            val_true_idx = val_idx[dset.y[val_idx].bool()]
            self._split_cache = (train_idx, val_idx, test_idx, train_true_idx, val_true_idx)

        train_idx, val_idx, test_idx, train_true_idx, val_true_idx = self._split_cache
        gen = torch.Generator()
        gen.manual_seed(self._call_count)
        self._call_count += 1

        train_counts = torch.randint(0, self.n_max + 1, (len(train_true_idx),), generator=gen)
        val_counts = torch.randint(0, self.n_max + 1, (len(val_true_idx),), generator=gen)
        poisoned = LabelPoisonedDataset(dset, torch.cat([train_true_idx, val_true_idx]),
                                        torch.cat([train_counts, val_counts]))
        n_base = len(dset)
        n_train_poison = int(train_counts.sum().item())
        n_val_poison = int(val_counts.sum().item())
        aug_train_idx = torch.cat([train_idx, torch.arange(n_base, n_base + n_train_poison)])
        aug_val_idx = torch.cat([val_idx, torch.arange(n_base + n_train_poison,
                                                        n_base + n_train_poison + n_val_poison)])
        kw = dict(batch_size=self.batch_size, is_oneshot=self.is_oneshot,
                  is_balanced=self.is_balanced)
        return (
            ExperimentDataLoader(poisoned, aug_train_idx, is_shuffle=True, **kw),
            ExperimentDataLoader(poisoned, aug_val_idx, is_shuffle=False, **kw),
            ExperimentDataLoader(poisoned, test_idx, is_shuffle=False, **kw),
        )


def make_random_poisoned_splitter(n_max: int, is_oneshot=False,
                                   batch_size=128, is_balanced=True):
    return RandomPoisonedSplitter(n_max, is_oneshot, batch_size, is_balanced)


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)   # L2‑norm
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def _single_run_dset(loader, model, optim, criteriorator, device, is_train,
                     return_outputs= False):
    loss_list = []
    yh_list = []
    y_list = []

    def _run_batch(bx, by):
        bx = bx.to(device)
        by = by.to(device)
        byh = model(bx)
        loss = criteriorator.loss_func(byh, by)
        if is_train:
            loss.backward()
            if criteriorator.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               criteriorator.max_grad_norm)
            optim.step()
        loss_list.append(loss.item())
        yh_list.append(byh.detach())
        y_list.append(by.detach())

    if is_train:
        for bx, by in loader:
            optim.zero_grad()
            _run_batch(bx, by)
    else:
        with torch.no_grad():
            for bx, by in loader:
                _run_batch(bx, by)

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
        val_loader, model, optim, criteriorator, device, is_train = False)

    stop_flag, best_flag = criteriorator.get_stop_best_flags(
        train_crit = train_crit, val_crit = val_crit)
    return stop_flag, best_flag, train_crit, val_crit

def _get_model_data_result(loader, model, criteriorator, device):
    crit, y, yh = _single_run_dset(
        loader, model, optim = None, criteriorator = criteriorator,
        device = device,
        is_train = False, return_outputs = True)
    return ModelDataResult(criteria = crit,
                           y = np.squeeze(y.detach().cpu().numpy()),
                           yh = np.squeeze(yh.detach().cpu().numpy()))

def _perform_episode(
        summary_dir,
        data_loaders,
        logger, device, config):
    # Derive dataset name and episode index from the directory structure:
    # summary_dir = .../results-final/<dataset>/<config>/<episode>
    _parts = os.path.normpath(summary_dir).split(os.sep)
    _episode_idx = int(_parts[-1]) if _parts[-1].isdigit() else 0
    _dataset_name = _parts[-3] if len(_parts) >= 3 else 'unknown'

    _tracker = make_mlflow_run(
        experiment_name=_dataset_name,
        config=config.name,
        dataset=_dataset_name,
        episode=_episode_idx,
        device=str(device),
    )
    _hparams = {}
    if config.opt_factory is None and config.optim_class is not None:
        _hparams['optim'] = config.optim_class.__name__
        _hparams.update(config.optim_args)
    if hasattr(config.criteriorator, '_max_iters'):
        _hparams['max_iters'] = config.criteriorator._max_iters
    if hasattr(config.criteriorator, '_patience'):
        _hparams['patience'] = config.criteriorator._patience
    _tracker['hparams'] = _hparams

    train_losses = []
    val_losses = []

    model = config.model_creator_func()
    model.to(device)
    if config.opt_factory is not None:
        optim = config.opt_factory(model)
    else:
        optim = config.optim_class(params = model.parameters(), **config.optim_args)
    lr_scheduler = (config.lr_scheduler_class(optim, **config.lr_scheduler_args)
                    if config.lr_scheduler_class is not None else None)
    epoch_num = 0

    run_flag = True
    best_model = deepcopy(model)
    criteriorator = copy(config.criteriorator)
    criteriorator.init_episode()
    train_crits = []
    val_crits = []
    best_epoch = 0
    best_val_crit = None
    while(run_flag):
        stop_flag, best_flag, train_crit, val_crit = \
            _single_epoch(data_loaders['train'],
                        data_loaders['val'], model,
                        optim, criteriorator, device)
        if lr_scheduler is not None:
            lr_scheduler.step()
        train_crits.append(train_crit)
        val_crits.append(val_crit)
        for col in train_crit.columns:
            _tracker.track(float(train_crit.at[0, col]), name=col, step=epoch_num,
                           context={'split': 'train'})
        for col in val_crit.columns:
            _tracker.track(float(val_crit.at[0, col]), name=col, step=epoch_num,
                           context={'split': 'val'})
        epoch_num += 1
        run_flag = not stop_flag
        if(best_flag):
            best_model = deepcopy(model)
            best_epoch = epoch_num
            best_val_crit = val_crit
    if best_val_crit is None:
        logging.warning('No best epoch found (loss may be NaN) — using last epoch as fallback')
        best_val_crit = val_crit
        best_model = model
        best_epoch = epoch_num
    logging.info(
        f'Training complete — best epoch: {best_epoch} / {epoch_num}, '
        f'val metrics: {best_val_crit.iloc[0].to_dict()}'
    )

    train_crits = pd.concat(train_crits, ignore_index = True)
    val_crits = pd.concat(val_crits, ignore_index = True)
    results = {
        name: _get_model_data_result(loader, best_model, criteriorator, device) \
        for name, loader in data_loaders.items()}

    ep_res = EpisodeResult(
        split_results = results,
        split_criteria_epochs = {'train' : train_crits, 'val' : val_crits},
        best_model = best_model)

    try:
        from sklearn.metrics import roc_auc_score
        for split_name, split_res in results.items():
            auc = float(roc_auc_score(split_res.y, split_res.yh))
            _tracker.track(auc, name='auc', step=0, context={'split': split_name})
    except Exception:
        pass
    _tracker.close()

    try:
        summarize_episode(summary_dir, ep_res, config)
    except Exception as e:
        logging.error(f'Summary failed: {e}', exc_info=True)

    return ep_res

def _perform_multiple_episodes_subprocess(summary_dir, dataset, device, config):
    """Run each episode as an independent subprocess — MPS-safe parallelism.

    Pickles config and dataset to temp files, spawns one process per episode,
    waits for all to finish, then reloads EpisodeResults from the saved pkls.
    """
    import tempfile, subprocess, pickle as pkl

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl.dump(config, f)
        config_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pkl.dump(dataset, f)
        dataset_path = f.name

    procs = []
    log_files = []
    try:
        for ep_idx in range(config.n_episodes):
            ep_dir = joinmakedir(summary_dir, str(ep_idx))
            log_path = os.path.join(ep_dir, 'worker.log')
            lf = open(log_path, 'w')
            log_files.append(lf)
            logging.info(f'{ep_idx:3d} - {config.name} [subprocess]')
            p = subprocess.Popen(
                [sys.executable, '-m', 'src._episode_worker',
                 config_path, dataset_path, str(ep_idx), summary_dir, str(device)],
                cwd=os.path.join(os.path.dirname(__file__), '..'),
                stdout=lf, stderr=lf,
            )
            procs.append(p)

        for ep_idx, p in enumerate(procs):
            ret = p.wait()
            log_files[ep_idx].flush()
            if ret != 0:
                log_path = os.path.join(summary_dir, str(ep_idx), 'worker.log')
                logging.error(
                    f'Episode worker {ep_idx} exited with code {ret} '
                    f'(see {log_path})'
                )
    finally:
        for lf in log_files:
            lf.close()
        os.unlink(config_path)
        os.unlink(dataset_path)

    # Reload results saved by each worker
    episode_results = []
    for ep_idx in range(config.n_episodes):
        ep_dir = os.path.join(summary_dir, str(ep_idx))
        pkl_path = os.path.join(ep_dir, 'test-res.pkl')
        with open(pkl_path, 'rb') as f:
            episode_results.append(pkl.load(f))
    return episode_results


def _perform_multiple_episodes(
        summary_dir, dataset, device, config, is_mp, sequential_episodes=False):
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
    if device.type == 'mps' and config.n_episodes > 1 and not sequential_episodes:
        episode_results = _perform_multiple_episodes_subprocess(
            summary_dir, dataset, device, config)
    elif is_mp:
        n_workers = config.n_episodes
        with mp.Pool(n_workers) as mppool:
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
    try:
        summarize_all_episodes(summary_dir, multi_ep_result, config)
    except Exception as e:
        logging.error(f'Episode summary failed for {config.name}: {e}', exc_info=True)
    return multi_ep_result

@dataclass
class ExperimentConfiguration:
    name : str
    model_creator_func : callable
    optim_class : torch.optim.Optimizer = torch.optim.Adam
    optim_args : Dict = field(default_factory = lambda : {'lr' : 0.001})
    criteriorator : Criteriorator = field(
        default_factory=lambda: BasicCriteriorator(
            torch.nn.BCEWithLogitsLoss(), 100))
    data_splitter : callable = basic_data_splitter
    n_episodes : int = 5
    is_mp : bool = False
    lr_scheduler_class : type = None
    lr_scheduler_args : Dict = field(default_factory=dict)
    opt_factory : callable = None  # opt_factory(model) -> optimizer; overrides optim_class/optim_args

def run_configurations(summary_dir, conf_list, dataset, device=None, is_mp=True,
                       parallel_configs=False, tpr_summary=None, sequential_episodes=False):
    if device is None:
        device = get_device()
    if device.type == 'mps' and is_mp:
        logging.info('Disabling multiprocessing: MPS tensors cannot be shared between processes')
        is_mp = False
    logging.info('Starting running configurations!')

    if parallel_configs and len(conf_list) > 1:
        n_workers = min(len(conf_list), os.cpu_count() or 1)
        logging.info(f'Running {len(conf_list)} configs across {n_workers} workers')
        args = [
            (joinmakedir(summary_dir, c.name), dataset, device, c, False, sequential_episodes)
            for c in conf_list
        ]
        with mp.Pool(n_workers) as pool:
            result = pool.starmap_async(_perform_multiple_episodes, args)
            while not result.ready():
                time.sleep(1)
            conf_res = result.get()
    else:
        conf_res = [
            _perform_multiple_episodes(
                summary_dir=joinmakedir(summary_dir, c.name),
                dataset=dataset, device=device, config=c, is_mp=is_mp,
                sequential_episodes=sequential_episodes)
            for c in conf_list
        ]

    summarize_all_configurations(summary_dir, conf_res, conf_list)
    if tpr_summary is not None:
        write_tpr_summary_csv(summary_dir, conf_list, conf_res, tpr_summary)

oneshot_datasplitter = partial(basic_data_splitter, is_oneshot = True)
