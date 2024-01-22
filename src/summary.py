from sklearn.metrics import roc_curve
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os

import plotly
import plotly.graph_objects as go
import plotly.express as px

def summarize_episode(summary_dir, best_model, ep_res):
    pass
    #raise NotImplementedError()

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

        self.tpr_mean = np.mean(tpr_interp, axis = 0)
        self.tpr_min = self.tpr_percentiles[0]
        self.tpr_max = self.tpr_percentiles[4]
        self.tpr_25 = self.tpr_percentiles[1]
        self.tpr_75 = self.tpr_percentiles[3]

        self.num_episodes = tpr_interp.shape[0]

def _get_roc_curves_from_episode(res):
    train_roc = CombinedRocResult(*zip(*[
        (ep_res.train_result.y, ep_res.train_result.yh) \
        for ep_res in res.episode_results]))

    val_roc = CombinedRocResult(*zip(*[
        (ep_res.val_result.y, ep_res.val_result.yh) \
        for ep_res in res.episode_results]))

    test_roc = CombinedRocResult(*zip(*[
        (ep_res.test_result.y, ep_res.test_result.yh) \
        for ep_res in res.episode_results]))

    return \
        train_roc, \
        val_roc, \
        test_roc

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
        for roc, res_name, c in zip(
                rocs, ['Train', 'Validation', 'Test'], colors):
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
                    fillcolor = _color_str_change_opacity(c, 0.2),
                    mode = 'lines',
                    line = dict(color = 'rgba(0,0,0,0)'),
                    hoverinfo = 'skip',
                    legendgroup = roc_name,
                    showlegend = False))
        fig.update_layout(showlegend = True)
        fig.update_traces(
            visible = 'legendonly',
            selector = lambda x: \
                any((not x.name is None and d in x.name) or \
                (not x.legendgroup is None and d in x.legendgroup) \
                for d in disabled_modes))
        fig.write_html(fname)

def summarize_all_episodes(summary_dir, episode_results):
    #train, val, test
    _gen_roc_to_file(
        fname = os.path.join(summary_dir, 'graph.html'),
        multi_ep_results = [episode_results],
        names = ['model'],
        disabled_modes = ['Train'])

def summarize_all_configurations(summary_dir, multi_ep_results):
    #train, val, test
    _gen_roc_to_file(
        fname=os.path.join(summary_dir, 'graph.html'),
        multi_ep_results = multi_ep_results,
        #TODO names from nowhere, restructure
        names = ['fpr0.4', 'fpr0.2', 'bce'],
        disabled_modes = ['Train', 'Validation'])





