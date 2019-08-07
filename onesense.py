from itertools import compress
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from statsmodels.nonparametric.api import KDEUnivariate
from scipy.signal import argrelextrema
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from umap import UMAP
from joblib import Parallel, delayed
from numpy.random import randint
from dcor import distance_correlation
import os
from tqdm import tqdm, tqdm_notebook
import six
from adjustText import adjust_text

def _type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'
    
def progress(it):
    env = _type_of_script()
    if hasattr(it, '__len__'):
        total = len(it)
    else:
        total = None
    if env == 'jupyter':
        return tqdm_notebook(it, total=total)
    else:
        return tqdm(it, total=total)

def _test_umap(xdata, random_state, **kwargs):
    transformer = UMAP(random_state=random_state, **kwargs)
    x = transformer.fit_transform(xdata)
    return x, random_state, distance_correlation(xdata, x)

# def _get_umap(xdata, random_state, **kwargs):
#     transformer = UMAP(random_state=random_state, **kwargs)
#     x = transformer.fit_transform(xdata)
#     return x

def optimal_umap(xdata, n, n_jobs=1, **kwargs):
    rnds = randint(0, 1000*n, n)
    
    # written in a slightly wonky way on purpose... as soon as UMAP is run outside joblib child processes,
    # they silently hang afterward. Perhaps some bad interaction with numba
    if n_jobs > 1:    
        out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_test_umap)(xdata=xdata, random_state=r, **kwargs) for r in rnds)
        x, r, scores = zip(*out)
    else:
        r = np.zeros(rnds.shape)
        scores = np.zeros(rnds.shape)
        x = dict()
        for i, random_state in progress(list(enumerate(rnds))):
            x[i], r[i], scores[i] = _test_umap(xdata=xdata, random_state=random_state, **kwargs)
        
    return x[np.argmax(scores)], int(r[np.argmax(scores)]), zip(x, r, scores)

def minimize_clusters(run_data, cluster_num, skip=0):
    xs, rs, scores = zip(*run_data)
    scores = np.array(scores)
    
    sortind = np.argsort(scores)[::-1]
    
    for i, ind in enumerate(sortind):
        n_clusters = len(_kde_bins(xs[ind].astype(float))[0])/2
        if n_clusters == cluster_num:
            if skip == 0:
                print('{0}th highest scoring iteration has desired number of clusters (score: {1})'.format(i, scores[ind]))
                return xs[ind], rs[ind]
            else:
                print('Skipping match...')
                skip -= 1
    else:
        print('Did not find solution with desired number of clusters')
        return xs[sortind[0]], rs[sortind[0]]

def _kde_bins(data, threshold=None):
    kde = KDEUnivariate(data)
    kde.fit("gau", 0.25, gridsize=1000)
    grid, z = kde.support, kde.density
    
    if threshold is None:
        min_pos = argrelextrema(kde.density, np.less)[0]
        # add in the two end points
        min_pos = np.append(min_pos, [0, len(kde.density) - 1])
        min_pos = np.unique(min_pos)
        minima = kde.density[min_pos]
        maxima = kde.density[argrelextrema(kde.density, np.greater)[0]]
        # find the highest peak, and then look for the largest
        # local minimum that is less than 20% its height
        # this ensures dips in density in tightly packed regions
        # are respected
        threshold = 1.2*np.max(minima[minima < 0.2*maxima.max()])

    zero_crossings = np.where(np.diff(np.sign(z - threshold)))[0]
    return grid[zero_crossings], grid, z

def _get_bin_patch(bin_left, width, height, color, bottom=0, angle=0):
    return patches.Rectangle(
        (bin_left, bottom),
        width,
        height,
        zorder=0,
        color=color,
        angle=angle
    )

def bins_to_patches(bins, height, c, bottom=0, orientation='vertical'):
    if orientation == 'vertical':
        lefts = bins[:-1]
        rights = bins[1:]
        widths = rights - lefts     
    else:
        lefts = [bottom,]*(len(bins) - 1)
        bottoms = bins[1:]
        widths = bins[1:] - bins[:-1]
    
    if isinstance(c, basestring):
        cs = [c,]*(len(bins) - 1)
    else:
        cs = c
    
    if orientation == 'vertical':
        if hasattr(height, '__len__'):
            return [_get_bin_patch(lefts[i], widths[i], height[i], cs[i], bottom) for i, _ in enumerate(lefts)]
        else:
            return [_get_bin_patch(lefts[i], widths[i], height, cs[i], bottom) for i, _ in enumerate(lefts)]        
    else:
        if hasattr(height, '__len__'):
            return [_get_bin_patch(lefts[i], widths[i], height[i], cs[i], bottoms[i], angle=-90) for i, _ in enumerate(lefts)]
        else:
            return [_get_bin_patch(lefts[i], widths[i], height, cs[i], bottoms[i], angle=-90) for i, _ in enumerate(lefts)]
        
def _normalized_stat(stat, quantiles=None, vmin=None, vmax=None, symmetric=False):
    if quantiles is not None:
        if symmetric is False:
            low_quantile = stat.quantile(quantiles[0])
            high_quantile = stat.quantile(quantiles[1])
        else:
            abs_quantile = np.min([np.abs(stat.quantile(quantiles[0])), np.abs(stat.quantile(quantiles[1]))])
            low_quantile = -abs_quantile
            high_quantile = abs_quantile
        norm = matplotlib.colors.Normalize(vmin=low_quantile, vmax=high_quantile, clip=True)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    stat = stat.map(lambda x: norm(x))
    return stat

def heat_scatter(x, y, statistic, threshold=None, quantiles=[0, 0.95],
                 vmin=None, vmax=None, figsize=None, summary='heatmap', plot_midpoint=False, plot_scatter=True,
                 ax=None, orientation='vertical', cmap='Blues', ymin=None, ymax=None):
    bins, _, _ = _kde_bins(x, threshold=threshold)
    binned_stat = binned_statistic(x, y, statistic, bins=bins)
    midpoints = (bins[1:] + bins[:-1])/2.0
    binned_stat = pd.Series(binned_stat.statistic)
    make_patch = ~binned_stat.isnull()
    binned_stat = binned_stat.fillna(0)
    
    if orientation == 'horizontal':
        w = x.copy()
        x = y.copy()
        y = w

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
    plt.sca(ax)
    
    if summary == "heatmap":
        cm = plt.cm.get_cmap(cmap)
        if vmin is None:
            cs = _normalized_stat(binned_stat, quantiles=quantiles)
        else:
            cs = _normalized_stat(binned_stat, vmin=vmin, vmax=vmax)
        cs = cm(cs)
        if plot_scatter:
            plt.scatter(x, y, s=5, color='lightgray')
            if ymin is None:
                if orientation == 'vertical':
                    ymin, ymax = plt.ylim()
                else:
                    ymin, ymax = plt.xlim()
        else:
            ymin, ymax = 0, 1
            ax.set_xlim(np.min(bins)*1.05,np.max(bins)*1.05)

        ymin = np.min([ymin, 0])
        ps = bins_to_patches(bins, ymax - ymin, cs, orientation=orientation, bottom=ymin)
        ps = compress(ps, make_patch)
        for p in ps:
            ax.add_patch(p)
        if orientation == 'vertical':
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_xlim(ymin, ymax)
    elif summary == "bar":
        if plot_scatter:
            plt.scatter(x, y, s=5, color='SteelBlue')
        else:
            ax.set_ylim(np.min(binned_stat)*0.95, np.max(binned_stat)*1.1)
            ax.set_xlim(np.min(bins)*1.05,np.max(bins)*1.05)
        ymin, ymax = plt.ylim()
        ymin = np.min([ymin, 0])
        ps = bins_to_patches(bins, binned_stat, 'lightgray', orientation=orientation, bottom=ymin)
        ps = compress(ps, make_patch)
        for p in ps:
            ax.add_patch(p)
            
    if plot_midpoint:
        plt.scatter(list(compress(midpoints, make_patch)),
                list(compress(binned_stat, make_patch)),
                marker='_', color='orangered')        
    
    if orientation == 'vertical':
        plt.xticks([]);
    else:
        plt.yticks([]);
        for tick in ax.get_xticklabels():
            tick.set_rotation(-90)
    
    return binned_stat, bins    
    
    
def onesense(x, y, c, xs, ys, 
             xlabels=None, ylabels=None, left=0.10, width=0.7, bottom=0.1, height=0.7,
                              scatter_pad=0.02, marginal_pad = 0.01, figsize=[7,7],
                              scatter_cmap='RdBu_r', cmaps=None, annotations=None,
                              label=False, xlims=None, ylims=None,
                              s=25):

    bottom_h = left_h = left + width + scatter_pad

    nx = len(xs)
    ny = len(ys)

    bottom_steps = np.linspace(bottom_h, 1, nx + 1)
    dbottom = np.diff(bottom_steps)[0] - marginal_pad
    bottom_steps = bottom_steps[:-1]
    left_steps = np.linspace(left_h, 1, ny + 1)
    dleft = np.diff(left_steps)[0] - marginal_pad
    left_steps = left_steps[:-1]

    rect_scatter = [left, bottom, width, height]

    x_rects = [[left, b, width, dbottom] for b in bottom_steps]
    y_rects = [[l, bottom, dleft, height] for l in left_steps]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=figsize)
    nullfmt = NullFormatter() 
    axScatter = plt.axes(rect_scatter)
    axScatter.xaxis.set_major_formatter(nullfmt)
    axScatter.yaxis.set_major_formatter(nullfmt)
    plt.xticks([])
    plt.yticks([])

    axxs = dict()
    axys = dict()
    
    for i, rect in enumerate(x_rects):
        
        axxs[i] = plt.axes(rect)
        axxs[i].xaxis.set_major_formatter(nullfmt)
        if xlabels is not None:
            axxs[i].text(-0.08, 0.5, xlabels[i],
            horizontalalignment='right',
            verticalalignment='center',
            transform=axxs[i].transAxes)
    for i, rect in enumerate(y_rects):
        axys[i] = plt.axes(rect)
        axys[i].yaxis.set_major_formatter(nullfmt)
        if ylabels is not None:
            axys[i].text(0.3, 1.02, ylabels[i],
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=60,
            transform=axys[i].transAxes)
        
    fig.text(0,0, ' ')

    # the scatter plot:
    quant_min = np.min([np.abs(c.quantile(0.02)), c.quantile(0.98)])
    
    norm = matplotlib.colors.Normalize(vmin=-quant_min, vmax=quant_min, clip=True)
    c = c.map(lambda x: norm(1.1*x))
    axScatter.scatter(x, y, c=c, s=s, cmap=plt.cm.get_cmap(scatter_cmap), edgecolor='lightgray')

    if label:
        texts = list()
        for name, xpos in x.iteritems():
            ypos = y.loc[name]
            texts.append(axScatter.text(xpos, ypos, name, fontsize=7))
        adjust_text(texts, ax=axScatter, lim=5)
    
    if cmaps is None:
        cmaps = ['Blues',]*np.max([nx, ny])
    
    if xlims is None:
        xlims = ((None, None),)*len(xs)
    if ylims is None:
        ylims = ((None, None),)*len(ys)

    for i, data in enumerate(xs):
        statx, binsx = heat_scatter(x, data, 'median', ax=axxs[i], cmap=cmaps[i], ymin=xlims[i][0], ymax=xlims[i][1])
    for i, data in enumerate(ys):
        staty, binsy = heat_scatter(y, data, 'median', ax=axys[i], cmap=cmaps[i], orientation='horizontal', ymin=ylims[i][0], ymax=ylims[i][1])
        
    midpointsx = (binsx[1:] + binsx[:-1])/2.0
    make_patchx = ~(statx == 0)
    midpointsx = list(compress(midpointsx, make_patchx))
    for j, mid in enumerate(midpointsx):
        axxs[len(axxs) - 1].text(mid, 1, j, horizontalalignment='center')
    clustersx = x.map(lambda x: find_nearest(midpointsx, x))

    midpointsy = (binsy[1:] + binsy[:-1])/2.0
    make_patchy = ~(staty == 0)
    midpointsy = list(compress(midpointsy, make_patchy))
    for j, mid in enumerate(midpointsy):
        axys[len(axys) - 1].text(1, mid, j, verticalalignment='center')
    clustersy = y.map(lambda y: find_nearest(midpointsy, y))
    
    if annotations is not None:
        topsy = list(compress(binsy[1:], make_patchy))
        bottomsy = list(compress(binsy[:-1], make_patchy))
        leftsx = list(compress(binsx[:-1], make_patchx))
        rightsx = list(compress(binsx[1:], make_patchx))
        
        ind = np.lexsort(zip(*annotations))
        annotations = annotations[ind]
        ax, ay = zip(*annotations)
        ax = np.array(ax)
        ay = np.array(ay)   
        
        for i in np.unique(ay):
            x_offset = 0.25
            for j in ax[ay == i]:
                ann = '\n'.join(clustersx[(clustersy == i) & (clustersx == j)].index)
                axys[len(axys) - 1].text(1 + x_offset, topsy[i], ann, verticalalignment='top', fontsize=7)
                x_offset = x_offset + 1.5
                axScatter.add_patch(patches.Rectangle((leftsx[j], bottomsy[i]), rightsx[j] - leftsx[j], topsy[i] - bottomsy[i],
                                   fill=False, edgecolor='lightgrey'))
                
    return clustersx, clustersy
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
