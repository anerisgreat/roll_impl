from sklearn.metrics import roc_curve
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import plotly
import plotly.graph_objects as go
import plotly.express as px

from .utils import joinmakedir
from scipy.stats import gamma, skewnorm, genhyperbolic, norminvgauss, nct, beta, norm
from scipy.stats import nakagami, rayleigh, gamma, invgamma, weibull_min
import pickle
import logging

from functools import partial

def np_split_true_false(yh, y):
    true_indeces = np.argwhere(y)
    false_indeces = np.argwhere(np.logical_not(y))

    true_yh = yh[true_indeces]
    false_yh = yh[false_indeces]

    return true_yh.squeeze(), false_yh.squeeze()

def _get_pdf_data_from_dist(data, dist):
    fit_params = dist.fit(data)
    ret = dist.isf(np.arange(0.01, 1, 0.01), *fit_params)
    return ret

def _get_pdf_gamm_from_dist():
    return partial(_get_pdf_data_from_dist, dist = gamma)

def _get_pdf_skewnorm_from_dist():
    return partial(_get_pdf_data_from_dist, dist = skewnorm)

def _get_pdf_from_dist(dist):
    return partial(_get_pdf_data_from_dist, dist = dist)

def _summarize_score_w_transform(
        transform, transform_name, dist_approx_dict, true_yh, false_yh, summary_dir, split):

    true_yh = transform(true_yh)
    false_yh = transform(false_yh)

    true_dist_approx = {
        f'f{k}_true_approx' : v(true_yh) \
        for k, v in dist_approx_dict.items()}
    false_dist_approx = {
        f'f{k}_false_approx' : v(false_yh)\
        for k, v in dist_approx_dict.items()}

    true_df = pd.DataFrame({'label' : 'True', 'score' : true_yh})
    false_df = pd.DataFrame({'label' : 'False', 'score' : false_yh})

    approx_true_dfs = [pd.DataFrame({'label' : k, 'score' : v }) \
                                    for k, v in true_dist_approx.items()]
    approx_false_dfs = [pd.DataFrame({'label' : k, 'score' : v }) \
                                    for k, v in false_dist_approx.items()]

    all_df = pd.concat((true_df, false_df,
                        *approx_true_dfs, *approx_false_dfs),
                        ignore_index = True)
    fig = px.ecdf(all_df, x = 'score', color = 'label')
    fig.write_html(os.path.join(summary_dir, f'{split}-f{transform_name}-scores.html'))

def summarize_episode(summary_dir, ep_res, config):
    criteria_dir = joinmakedir(summary_dir, 'criteria')
    #Summarize criteria
    for split, split_epochs in ep_res.split_criteria_epochs.items():
        fig = go.Figure()
        for c in split_epochs.columns:
            fig.add_traces(go.Scatter(
                x = split_epochs.index,
                y = split_epochs[c],
                mode = 'lines',
                name = c,
                legendgroup = c
            ))
        fig.update_layout(showlegend = True)
        fig.write_html(os.path.join(criteria_dir, f'{split}.html'))

    #Summarize scores
    for split, result in ep_res.split_results.items():
        with open(os.path.join(summary_dir, f'{split}-res.pkl'), 'wb') as ofile:
            pickle.dump(ep_res, ofile)

        #Score distrib graph
        true_yh, false_yh = np_split_true_false(result.yh, result.y)
        # _summarize_score_w_transform(
        #     transform = lambda x: x,
        #     transform_name = 'none',
        #     dist_approx_dict = {
        #         'norm' : _get_pdf_from_dist(norm)},
        #     true_yh = true_yh,
        #     false_yh = false_yh,
        #     summary_dir = summary_dir,
        #     split = split)
        # _summarize_score_w_transform(
        #     transform = lambda x: np.exp(x),
        #     transform_name = 'exp',
        #     #nakagami, rayleigh, gamma, invgamma, weibull_min
        #     dist_approx_dict = {
        #         'beta' : _get_pdf_from_dist(beta),
        #         'nakagami' : _get_pdf_from_dist(nakagami),
        #         'rayleigh' : _get_pdf_from_dist(rayleigh),
        #         'invgamma' : _get_pdf_from_dist(invgamma),
        #         'weibull_min' : _get_pdf_from_dist(weibull_min)},
        #     true_yh = true_yh,
        #     false_yh = false_yh,
        #     summary_dir = summary_dir,
        #     split = split)
        # _summarize_score_w_transform(
        #     transform = lambda x: x + np.min(x),
        #     transform_name = 'pmin',
        #     #nakagami, rayleigh, gamma, invgamma, weibull_min
        #     dist_approx_dict = {
        #         'beta' : _get_pdf_from_dist(beta),
        #         'nakagami' : _get_pdf_from_dist(nakagami),
        #         'rayleigh' : _get_pdf_from_dist(rayleigh),
        #         'invgamma' : _get_pdf_from_dist(invgamma),
        #         'weibull_min' : _get_pdf_from_dist(weibull_min)},
        #     true_yh = true_yh,
        #     false_yh = false_yh,
        #     summary_dir = summary_dir,
        #     split = split)
        # _summarize_score_w_transform(
        #     transform = lambda x: 1/(1 + np.exp(-x)),
        #     transform_name = 'sigm',
        #     #nakagami, rayleigh, gamma, invgamma, weibull_min
        #     dist_approx_dict = {
        #         'beta' : _get_pdf_from_dist(beta),
        #         'nakagami' : _get_pdf_from_dist(nakagami),
        #         'rayleigh' : _get_pdf_from_dist(rayleigh),
        #         'invgamma' : _get_pdf_from_dist(invgamma),
        #         'weibull_min' : _get_pdf_from_dist(weibull_min)},
        #     true_yh = true_yh,
        #     false_yh = false_yh,
        #     summary_dir = summary_dir,
        #     split = split)


@dataclass
class CombinedRocResult:
    fpr : np.ndarray
    tpr_mean : np.ndarray
    tpr_max : np.ndarray
    tpr_min : np.ndarray
    tpr_75 : np.ndarray
    tpr_25 : np.ndarray
    tpr_percentiles : np.ndarray
    num_episodes : int

    def __init__(self, y_list, yh_list):
        roc_curves = fprs, tprs, _ = zip(*[roc_curve(y > 0., yh) \
                                        for y, yh in zip(y_list, yh_list)])
        fpr = np.unique(np.concatenate(fprs))
        self.fpr = fpr
        tpr_interp = np.array([np.interp(fpr, tpri, npri) \
                            for tpri, npri in zip(fprs, tprs)])

        self.tpr_percentiles = np.percentile(
            tpr_interp, q = [0, 25, 50, 75, 100],
            axis = 0)

        # self.tpr_mean = np.mean(tpr_interp, axis = 0)
        self.tpr_min = self.tpr_percentiles[0]
        self.tpr_25 = self.tpr_percentiles[1]
        self.tpr_mean = self.tpr_percentiles[2]
        self.tpr_75 = self.tpr_percentiles[3]
        self.tpr_max = self.tpr_percentiles[4]

        self.num_episodes = tpr_interp.shape[0]

def _get_roc_curves_from_episode(res):
    all_split_scores = res.get_split_scores()

    return {name: CombinedRocResult(*score_tuple) \
                for name, score_tuple in all_split_scores.items()}

def _color_hex_str_to_rgb(s):
    return [int(s[i:i+2], 16) for i in range(1, len(s), 2)]

def _color_str_to_rgb(s):
    if s[0] == '#':
        return _color_hex_str_to_rgb(s)
    return list(map(int, s[5:-1].split(',')))

_color_str_to_rgb_str = lambda s: _color_rgb_to_str(_color_str_to_rgb(s))

def _color_change_brightness(c, a):
    if(a == 0):
        return c
    brighten = a > 0
    target = 255 if a > 0 else 0
    _interp = lambda x, y, z: x + (y - x)*z
    _interp_to_target = lambda x: int(_interp(x, target, a))
    return [_interp_to_target(ci) for ci in c[:3]] + c[3:]

def _color_change_opacity(c, a):
    return c[:3] + [a]

def _color_rgb_to_hex_str(c):
    return '#' + ''.join([f'{i:2x}' for i in c])

def _color_rgb_to_str(c):
    if(len(c) == 3):
        c = c + [1]
    return f'rgba({", ".join([str(ci) for ci in c])})'

_color_str_change_brightness= lambda s, a: \
    _color_rgb_to_str(
        _color_change_brightness(
            _color_str_to_rgb(s), a))

_color_str_change_opacity= lambda s, a: \
    _color_rgb_to_str(
        _color_change_opacity(
            _color_str_to_rgb(s), a))

def _gen_roc_to_file(fname, multi_ep_results, names,
                    disabled_modes = ['Train']):
    color_palette = plotly.colors.qualitative.Dark24
    color_iter = iter(color_palette)
    fig = go.Figure()
    for ep_results, name in zip(multi_ep_results, names):
        rocs = _get_roc_curves_from_episode(ep_results)
        base_color = _color_str_to_rgb_str(next(color_iter))
        colors = [
            base_color, \
            _color_str_change_brightness(base_color, 0.3),
            _color_str_change_brightness(base_color, -0.3)]
        colors_iter = iter(colors)
        for res_name, roc in rocs.items():
            c = next(colors_iter)
            roc_name = f'{name}-{res_name}'
            fig.add_trace(go.Scatter(
                x = roc.fpr,
                y = roc.tpr_mean,
                mode = 'lines',
                name = roc_name,
                legendgroup = roc_name,
                line = dict(color = c)))
            fig.add_trace(
                go.Scatter(
                    x = np.concatenate((roc.fpr, roc.fpr[::-1])),
                    y = np.concatenate((roc.tpr_75, roc.tpr_25[::-1])),
                    fill = 'toself',
                    fillcolor = _color_str_change_opacity(c, 0.1),
                    mode = 'lines',
                    line = dict(color = 'rgba(0,0,0,0)'),
                    hoverinfo = 'skip',
                    legendgroup = roc_name,
                    showlegend = False))
            fig.add_trace(
                go.Scatter(
                    x = np.concatenate((roc.fpr, roc.fpr[::-1])),
                    y = np.concatenate((roc.tpr_max, roc.tpr_min[::-1])),
                    fill = 'toself',
                    fillcolor = _color_str_change_opacity(c, 0.1),
                    mode = 'lines',
                    line = dict(color = 'rgba(0,0,0,0)'),
                    hoverinfo = 'skip',
                    legendgroup = roc_name,
                    showlegend = False))

        fig.update_layout(showlegend = True)
        fig.update_traces(
            visible = 'legendonly',
            selector = lambda x: \
                any((not x.name is None and d.lower() in x.name.lower()) or \
                (not x.legendgroup is None and d.lower() in x.legendgroup.lower()) \
                for d in disabled_modes))
        fig.write_html(fname)

def summarize_all_episodes(summary_dir, episode_results, config):
    #train, val, test
    _gen_roc_to_file(
        fname = os.path.join(summary_dir, 'graph.html'),
        multi_ep_results = [episode_results],
        names = [config.name],
        disabled_modes = ['Train'])

def summarize_all_configurations(summary_dir, multi_ep_results, configs):
    #train, val, test
    _gen_roc_to_file(
        fname=os.path.join(summary_dir, 'graph.html'),
        multi_ep_results = multi_ep_results,
        names = [c.name for c in configs],
        disabled_modes = ['Train', 'Val'])

