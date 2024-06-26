# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:35:49 2023

@author: YUW1SI
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import warnings
from matplotlib.path import Path
import matplotlib.patches as patches

from datasurfer.lib_plots.plot_utils import trigrid

#%%

def plot_violin(ax, df, ldata=None, rdata=None, **kwargs):
    def calc_violin_data(w, y0):
        from scipy import interpolate
        w0, inv_idx = np.unique(w, return_inverse=True)  
                    
        fct = (w0-w0.min()) / w0.ptp()            
        x1 = x0.ptp() * fct + x0.min()                       
        x2 = x1[inv_idx]            
        
        f = interpolate.interp1d(x0, y0, kind='linear', bounds_error=False, fill_value='extrapolate')
        y2 = f(x2)
        
        return y2

    labels = df.columns
    arr = df.dropna().to_numpy()  
    
    facecolor = kwargs.pop('facecolor', 'green') 
    edgecolor = kwargs.pop('edgecolor', 'black')
    alpha = kwargs.pop('alpha', 0.5)  
    levels = kwargs.pop('levels', 10)
    cmap = kwargs.pop('cmap', 'viridis')
    
    vits = ax.violinplot(arr, **kwargs)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels)    
    ax.spines['bottom'].set_visible(False) 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.grid(visible=True, which='major', linestyle='--', axis='y')   

    cbs = []
    for body, x0 in zip(vits['bodies'], arr.T):

        body.set_facecolor(facecolor)
        body.set_edgecolor(edgecolor)
        body.set_alpha(alpha)  
        
        p, = body.get_paths()
        arr = p.vertices
        u, w = arr.T
        
        if ldata is not None: 
            
            u0 = u.clip(u.min(), u.mean())
            tris = trigrid(u0, w)    
            y2 = calc_violin_data(w, ldata)
            cb = ax.tricontourf(tris, y2, levels=levels, cmap=cmap)
            cbs.append(cb)                        

        if rdata is not None:
            u0 = u.clip(u.mean(), u.max())
            tris = trigrid(u0, w)    
            y2 = calc_violin_data(w, rdata)
            cb = ax.tricontourf(tris, y2, levels=levels, cmap=cmap)
            cbs.append(cb)            
            
        
    
    return vits, cbs

    
    
#%%

def plot_dendrogram(ax, df, method='centroid'):
    """
    Plots a dendrogram using the given axis and dataframe.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis on which to plot the dendrogram.
    - df (pandas.DataFrame): The dataframe containing the data to be plotted.
    - method (str, optional): The linkage method to be used. Defaults to 'centroid'.

    Returns:
    - Z (numpy.ndarray): The linkage matrix.
    - dn (dict): The dendrogram dictionary.

    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(df.values.T, method=method)
    dn = dendrogram(Z, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.spines['left'].set_visible(False) 
    ax.axes.get_yaxis().set_visible(False)
    
    xlbls = [int(t.get_text()) for t in ax.get_xticklabels()]
    xlbltxts = [df.columns[idx] for idx in xlbls]    
    ax.set_xticklabels(xlbltxts, rotation=45, ha="right", fontsize=10)
    
    return Z, dn



    
#%%
def plot_parallel_coordinate(host, df, **kwargs):
    """
    Plots parallel coordinates for a given DataFrame.

    Parameters:
    - host: The main axis to plot on.
    - df: The DataFrame containing the data to be plotted.
    - **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
    - axes: A list of axes used for the plot.

    """
    ynames = list(df.columns)
    ys = np.array(df)
    
    ymins = kwargs.pop('ymins', ys.min(axis=0))    
    ymaxs = kwargs.pop('ymaxs', ys.max(axis=0))
    
    axes = kwargs.pop('axes', [host] + [host.twinx() for i in range(ys.shape[1] - 1)])

    # Rest of the code...
def plot_parallel_coordinate(host, df, **kwargs):
      
    ynames = list(df.columns)
    ys = np.array(df)
    
    ymins = kwargs.pop('ymins', ys.min(axis=0))    
    ymaxs = kwargs.pop('ymaxs', ys.max(axis=0))
    
    axes = kwargs.pop('axes', [host] + [host.twinx() for i in range(ys.shape[1] - 1)])

        
    # ymins = ys.min(axis=0)
    # ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins
    
    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    
    
    
    # axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    
    for i, ax in enumerate(axes):
        #ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.set_ylim(ymins[i], ymaxs[i])
        #ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='minor', length=0)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
    
    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=10)
    host.tick_params(axis='x', which='major', pad=7)
    
    host.spines['right'].set_visible(False)
    
    # colors = plt.cm.Set2.colors
    # legend_handles = [None for _ in ynames]
    for j in range(ys.shape[0]):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, **kwargs)
        #legend_handles[iris.target[j]] = patch
        host.add_patch(patch)
    #host.legend(legend_handles, iris.target_names,
                #loc='lower center', bbox_to_anchor=(0.5, -0.18),
                #ncol=len(iris.target_names), fancybox=True, shadow=True)
    #plt.tight_layout()
    return axes



#%%
def plot_xybar(ax, data, bins, width=None, labels=None, title=None, xlabel=None, 
              ylabel=None, yfun=None, pct=True, colors=None, rebuildx=False):
    """
    Plot a bar chart with multiple bars for each x value.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the bar chart.
    - data (list): A list of arrays containing the y values for each bar.
    - bins (array-like): The x values for the bars.
    - width (float, optional): The width of each bar. If not provided, it is automatically calculated.
    - labels (list, optional): The labels for each group of bars.
    - title (str, optional): The title of the plot.
    - xlabel (str, optional): The label for the x-axis.
    - ylabel (str, optional): The label for the y-axis.
    - yfun (function, optional): A function to apply to the y values before plotting.
    - pct (bool, optional): Whether to display the percentage of each bar.
    - colors (list, optional): The colors for each group of bars.
    - rebuildx (bool, optional): Whether to rebuild the x values as a range from 0 to the number of bins.

    Returns:
    - bs (list): The bar containers for each group of bars.
    """
    n = len(data)
    
    if width is None:
        width = min(np.diff(np.asarray(bins))) * 0.5 / n
        
    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
    
    for idx, dat in enumerate(data):
        y = dat
        x = bins
        
        if yfun:
            y = yfun(y)

        if rebuildx:
            x = np.arange(0, len(x))
            
        xloc = (x[:-1] + x[1:]) / 2   
        
        if colors:            
            color=colors[idx]
        else:
            color=None
        
        if labels:
            bs = ax.bar(xloc+offsets[idx], y, width=width, label=labels[idx], color=color)
        else:
            bs = ax.bar(xloc+offsets[idx], y, width=width, color=color)
        
        if pct:
            sum_ = np.abs(y).sum()
            pcs = np.abs(y) / sum_ * 100
            for rect, pc in zip(bs, pcs):
                height = rect.get_height()
        
                if pc > 1:
                    txt = f'{pc:.0f}%'
                else:
                    txt = f'{pc:.1f}%'
                
                if pc >= 0:
                    ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{txt}', 
                        ha='center', va='bottom', fontsize=8, color='grey')    
                else:
                    ax.text(rect.get_x(), height, f'{txt}', 
                        ha='center', va='bottom', fontsize=8, color='grey')                     
    
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.grid(b=True, which='major', linestyle='--', axis='y')
    ax.xaxis.set_tick_params(gridOn=False) 
    
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    
    if title:
        ax.set_title(title)
        
    if xlabel:
        ax.set_xlabel(xlabel)
        
    if ylabel:
        ax.set_ylabel(ylabel)
    
    return bs

#%%
def plot_xybarh(ax, data, bins, width=None, labels=None, title=None, xlabel=None, 
              ylabel=None, yfun=None, pct=True, colors=None):

    n = len(data)
    
    if width is None:
        
        width = min(np.diff(np.asarray(bins))) * 0.5 / n
        
        
    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)  
    
    
    for idx, dat in enumerate(data):
        
        y = dat
        x = bins

        if colors:            
            color=colors[idx]
        else:
            color=None
        
        if yfun:
            y = yfun(y)
        x = (x[:-1] + x[1:]) / 2
        
        if labels:
            bs = ax.barh(x+offsets[idx], y, height=width, label=labels[idx], color=color)
        else:
            bs = ax.barh(x+offsets[idx], y, height=width, color=color)        

        if pct:
            sum_ = np.abs(y).sum()
            pcs = np.abs(y) / sum_ * 100
            for rect, pc in zip(bs, pcs):
                
                width = rect.get_width()
        
                if pc > 1:
                    txt = f'{pc:.0f}%'
                else:
                    txt = f'{pc:.1f}%'
                
                if pc >= 0:
                    ax.text(width+np.diff(ax.get_xlim())*0.1, rect.get_y() + rect.get_height()*0 / 2.0, f'{txt}', 
                        ha='center', va='bottom', fontsize=8, color='grey')
                    
    ax.set_yticks(bins)
    ax.grid(b=True, which='major', linestyle='--', axis='x')
    ax.yaxis.set_tick_params(gridOn=False) 
    
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
        
    
    return bs                    
        

#%%


def plot_histh(ax, data, bins, width=None, labels=None, title=None, xlabel=None, 
              ylabel=None, yfun=None, pct=True, rebuildx=False):
    '''
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    bins : TYPE
        DESCRIPTION.
    width : TYPE, optional
        DESCRIPTION. The default is None.
    labels : TYPE, optional
        DESCRIPTION. The default is None.
    title : TYPE, optional
        DESCRIPTION. The default is None.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.
    yfun : TYPE, optional
        DESCRIPTION. The default is None.
    pct : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    '''
    
    #assert data.ndim == 2, 'Data should be a 2-dimentional array.'
    
    
    n = len(data)
    
    if width is None:
        
        width = min(np.diff(np.asarray(bins))) * 0.5 / n
        
        
    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
    

    
    for idx, dat in enumerate(data):
        
        y, x = np.histogram(dat, bins)    
        
        if yfun:
            y = yfun(y)
            
        if rebuildx:
            x = np.arange(0, len(x))
            
        xloc = (x[:-1] + x[1:]) / 2     

        
        if labels:
            bs = ax.barh(xloc+offsets[idx], y, height=width, label=labels[idx])
        else:
            bs = ax.barh(xloc+offsets[idx], y, height=width)
        
    
        if pct:
            sum_ = np.abs(y).sum()
            pcs = np.abs(y) / sum_ * 100
            for rect, pc in zip(bs, pcs):
                
                width = rect.get_width()
        
                if pc > 1:
                    txt = f'{pc:.0f}%'
                else:
                    txt = f'{pc:.1f}%'
                
                if pc > 0:
                    ax.text(width+np.diff(ax.get_xlim())*0.1, rect.get_y() + rect.get_height()*0 / 2.0, f'{txt}', 
                        ha='center', va='bottom', fontsize=8, color='grey')    


    
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.grid(b=True, which='major', linestyle='--', axis='x')
    ax.yaxis.set_tick_params(gridOn=False) 
    
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
        
    
    return bs

#%%

def plot_hist(ax, data, bins, width=None, labels=None, title=None, xlabel=None, 
              ylabel=None, yfun=None, pct=True, rebuildx=False):
    """
    Plot histogram(s) on the given axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the histogram(s).
    - data (list or array-like): The data to be plotted as histogram(s).
    - bins (int or sequence): The number of bins or the bin edges.
    - width (float, optional): The width of each bar in the histogram. If not provided, it is calculated automatically.
    - labels (list, optional): The labels for each histogram. If not provided, no labels will be shown.
    - title (str, optional): The title of the plot.
    - xlabel (str, optional): The label for the x-axis.
    - ylabel (str, optional): The label for the y-axis.
    - yfun (function, optional): A function to apply to the y-values of the histogram(s).
    - pct (bool, optional): Whether to show the percentage values on top of each bar. Default is True.
    - rebuildx (bool, optional): Whether to rebuild the x-axis values. Default is False.

    Returns:
    - bs (matplotlib.container.BarContainer): The bar containers representing the histogram(s).
    """
    
    #assert data.ndim == 2, 'Data should be a 2-dimentional array.'
    
    
    n = len(data)
    
    if width is None:
        
        width = min(np.diff(np.asarray(bins))) * 0.5 / n
        
        
    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
    
   
    for idx, dat in enumerate(data):
        
        y, x = np.histogram(dat, bins)    
        
        if yfun:
            y = yfun(y)
        

        if rebuildx:
            x = np.arange(0, len(x))
            
        xloc = (x[:-1] + x[1:]) / 2        

        
        if labels:
            bs = ax.bar(xloc+offsets[idx], y, width=width, label=labels[idx])
        else:
            bs = ax.bar(xloc+offsets[idx], y, width=width)
        
            
        if pct:
            sum_ = np.abs(y).sum()
            pcs = np.abs(y) / sum_ * 100
            for rect, pc in zip(bs, pcs):
                
                height = rect.get_height()
        
                if pc > 1:
                    txt = f'{pc:.0f}%'
                else:
                    txt = f'{pc:.1f}%'
                
                if pc >= 0:
                    ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{txt}', 
                        ha='center', va='bottom', fontsize=8, color='grey') 
                else:
                    ax.text(rect.get_x(), height, f'{txt}', 
                        ha='center', va='bottom', fontsize=8, color='grey')   
    

    
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.grid(b=True, which='major', linestyle='--', axis='y')
    ax.xaxis.set_tick_params(gridOn=False) 
    
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    
    if title:
        ax.set_title(title)
        
    if xlabel:
        ax.set_xlabel(xlabel)
        
    if ylabel:
        ax.set_ylabel(ylabel)
    
    
    return bs

#%%
def plot_hist_stacked(ax, data, bins, width=None, labels=None, title=None, xlabel=None, 
              ylabel=None, yfun=None, pct=True):
    

    if width is None:
        
        width = min(np.diff(np.asarray(bins))) * 0.5   
    
    bottom = np.zeros(len(bins)-1)
    
    heights = np.zeros(len(bins)-1)
    
    for idx, dat in enumerate(data):
        
        y, x = np.histogram(dat, bins)
               
        
        if yfun:
            y = yfun(y)
            
        x = (x[:-1] + x[1:]) / 2
        
        
        if labels:
            bs = ax.bar(x, y, bottom=bottom, width=width, label=labels[idx])
        else:
            bs = ax.bar(x, y, bottom=bottom, width=width)   
            
        
        bottom = bottom + y
        heights = heights + np.array([rect.get_height() for rect in bs])
        
        
        
        
    if pct:
        sum_ = np.abs(bottom).sum()
        pcs = np.abs(bottom) / sum_ * 100
        for height, rect, pc in zip(heights, bs, pcs):
            
            #height = rect.get_height()
    
            if pc > 1:
                txt = f'{pc:.0f}%'
            else:
                txt = f'{pc:.1f}%'
            
            if pc > 0:
                ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{txt}', 
                    ha='center', va='bottom', fontsize=8, color='grey')

    ax.set_xticks(bins)
    ax.grid(b=True, which='major', linestyle='--', axis='y')
    ax.xaxis.set_tick_params(gridOn=False) 
    
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    
    if title:
        ax.set_title(title)
        
    if xlabel:
        ax.set_xlabel(xlabel)
        
    if ylabel:
        ax.set_ylabel(ylabel)
    
    
    return ax

#%%

def get_maxmin_labels(dat, prefix=None, rnd=1):
    """
    Returns a list of labels containing the maximum, minimum, and average values of the input data.

    Parameters:
    - dat: list of numpy arrays, the input data
    - prefix: list of strings, optional prefix for each label
    - rnd: int, optional number of decimal places to round the values

    Returns:
    - list of strings, the labels

    Example usage:
    >>> data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> labels = get_maxmin_labels(data, prefix=['A', 'B'], rnd=2)
    >>> print(labels)
    ['A: max 3.0; min 1.0; avg 2.0', 'B: max 6.0; min 4.0; avg 5.0']
    """

    if prefix:
        return [f'{p}: max {round(d.max(), rnd)}; min {round(d.min(), rnd)}; avg {round(d.mean(), rnd)}' for d, p in zip(dat, prefix)]
    else:
        return [f'max {round(d.max(), rnd)}; min {round(d.min(), rnd)}; avg {round(d.mean(), rnd)}' for d in dat]



#%%

def plot_histogram(ax, data, bins, width=None, labels=None, yfun=None, 
                   pct=True, pctfs=8, pctcolor='grey', colors=None, 
                   rebuildx=True, horizontal=False, pctoffset=0, pctfct=0.5,
                   hide_bottom=True, stacked=True, plot_kwargs=None):
    """
    Plot a histogram on the given axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the histogram.
    - data (list): The data to be plotted as a histogram.
    - bins (int or sequence): The number of bins or the bin edges.
    - width (float, optional): The width of each bar. If not provided, it is calculated automatically.
    - labels (list, optional): The labels for each data set.
    - yfun (function, optional): A function to apply to the y-values of each data set.
    - pct (bool, optional): Whether to display the percentage on top of each bar.
    - pctfs (int, optional): The font size of the percentage text.
    - pctcolor (str, optional): The color of the percentage text.
    - colors (list, optional): The colors for each data set.
    - rebuildx (bool, optional): Whether to rebuild the x-axis ticks.
    - horizontal (bool, optional): Whether to plot the histogram horizontally.
    - pctoffset (float, optional): The offset of the percentage text from the bar.
    - pctfct (float, optional): The factor to adjust the position of the percentage text.
    - hide_bottom (bool, optional): Whether to hide the bottom spine.
    - stacked (bool, optional): Whether to stack the bars.
    - plot_kwargs (dict, optional): Additional keyword arguments to pass to the bar plot function.

    Returns:
    - bs (matplotlib.container.BarContainer): The bar container object.
    # Example 1
    data1 = [np.random.normal(0, 1, 1000)]
    bins1 = np.linspace(-3, 3, 11)
    labels1 = ['Data 1']
    fig1, ax1 = plt.subplots()
    plot_histogram(ax1, data1, bins1, labels=labels1)
    ax1.legend()
    ax1.set_title('Histogram Example 1')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')

    # Example 2
    data2 = [np.random.normal(0, 1, 1000), np.random.normal(2, 1, 1000)]
    bins2 = np.linspace(-3, 5, 16)
    labels2 = ['Data 1', 'Data 2']
    fig2, ax2 = plt.subplots()
    plot_histogram(ax2, data2, bins2, labels=labels2, stacked=False)
    ax2.legend()
    ax2.set_title('Histogram Example 2')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')

    # Example 3
    data3 = [np.random.normal(0, 1, 1000), np.random.normal(2, 1, 1000)]
    bins3 = np.linspace(-3, 5, 16)
    labels3 = ['Data 1', 'Data 2']
    fig3, ax3 = plt.subplots()
    plot_histogram(ax3, data3, bins3, labels=labels3, horizontal=True)
    ax3.legend()
    ax3.set_title('Histogram Example 3')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Value')
    
    # Example 3
    import datastructure as ds
    
    dp = ds.DataPool(['test_files/eATS16_2023-07-03_11-43-42_019.mf4'])
    obj, = dp
    dat0 = obj.load("TG202_GetriebedeckelOF").values
    dat1 = obj.load("TE203_AC1_Busbar_Schraubpunkt").values + 1
    dat2 = obj.load("TE204_AC2_Bussbar_Mitte").values + 1
       
    bins = [45, 45.3, 45.7, 45.9, 46, 46.5]
    # dat0 = [10, -15, 12, -4, 5]

    fig, ax = plt.subplots()
    
    #ax.barh(dat0, bins[:-1])
    
    plot_histogram(ax, [[dat0, dat1], [dat0, dat2]], bins,
                    colors = ['b', 'g', 'b', 'r'],
                    labels=['parm1', 'parm0','parm1', 'parm2'], yfun=lambda y:y/600, pctfs=8,
                    rebuildx=True,  pctoffset=0.2, pctfct=0.5, hide_bottom=False, stacked=False)
    
    
    x = np.asarray(ax.get_xticks())
    xloc = (x[:-1] + x[1:]) / 2
    ax.set_xticks(xloc)
    
    ax.set_xticklabels(list('abcde'))
    ax.legend()
    """
 

    
    n = len(data)
    x = np.asarray(bins)
    
    plot_kwargs = plot_kwargs or {} 
    
    
    if rebuildx:
        x = np.arange(0, len(bins))
        
    if width is None:        
        width = min(np.diff(x)) * 0.5 / n   

    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
    bottom  = np.zeros([n, len(bins)-1], dtype=float)    

    xlocs = (x[:-1] + x[1:]) / 2 + offsets.reshape(-1, 1) 
    
    count = 0
    used_labels = []
    for idx0, dat in enumerate(data):
        try:
            arr2d = np.asarray(dat).squeeze()
            
            if arr2d.ndim == 1:   
                
                arr2d = arr2d.reshape(1, -1)
                
        except ValueError:
            
            arr2d = [np.asarray(d) for d in dat]
        
        for arr1d in arr2d:
            
            if arr1d.size == len(bins) - 1:            
                y = arr1d
                
            else:               
                y, _ = np.histogram(arr1d, bins)
                
            if yfun:
                y = yfun(y)
                
            if colors:     
                
                color = colors[count]
                                
            else:
                color = None
                
            if labels:
                label = labels[count]
                
                if label in used_labels:
                    flag_label = False
                    warnings.warn(f'Label "{label}" exists already.')
                else:
                    flag_label = True
                    used_labels.append(label)
            
            xloc = xlocs[idx0, :]
            
            if stacked:
                
                bttm = bottom[idx0, :]
            else:
                bttm = 0
                        
            if not horizontal:
                                
                if labels and flag_label:

                    bs = ax.bar(xloc, y, width=width, bottom=bttm, label=label, color=color, **plot_kwargs)
                else:
                    bs = ax.bar(xloc, y, width=width, bottom=bttm, color=color, **plot_kwargs)   

            else:

                if labels and flag_label:
                    bs = ax.barh(xloc, y, height=width, left=bttm, label=label, color=color, **plot_kwargs)
                else:
                    bs = ax.barh(xloc, y, height=width, left=bttm, color=color, **plot_kwargs)
            
            if stacked:
                bottom[idx0, :] = bottom[idx0, :] + y
            else:
                
                bottom[idx0, :] = np.max(np.vstack((bottom[idx0, :], y)), axis=0)
                                
            count += 1

    if pct:
        sums = np.abs(bottom).sum(axis=1, keepdims=True)
        pcs  = np.abs(bottom) / sums * 100
        

        for height, xloc, pct in zip(bottom, xlocs, pcs):
                        
            for h, xx, pc in zip(height, xloc, pct):
                
                if pc > 1:
                    txt = f'{pc:.0f}%'
                else:
                    txt = f'{pc:.1f}%'  

                if pc > 0.0999:
                    
                    if not horizontal:
                        if h >= 0: 
                            ax.text(xx, h, f'{txt}', 
                                ha='center', va='bottom', fontsize=pctfs, color=pctcolor)   
                        else:
                            
                            ax.text(xx, h-pctoffset, f'{txt}', 
                                ha='center', va='bottom', fontsize=pctfs, color=pctcolor)     
                    else:
                        
                        if h >= 0: 
                            ax.text(h+pctoffset, xx-width*pctfct, f'{txt}', 
                                ha='center', va='bottom', fontsize=pctfs, color=pctcolor)                           
                        else:
                            
                            ax.text(h-pctoffset, xx-width*pctfct, f'{txt}', 
                                ha='center', va='bottom', fontsize=pctfs, color=pctcolor)                         
                        
                    
                
    if not horizontal:    
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.grid(visible=True, which='major', linestyle='--', axis='y')
        ax.xaxis.set_tick_params(gridOn=False) 
        ax.xaxis.set_tick_params(which='minor', bottom=False)
        if hide_bottom:
            ax.spines['bottom'].set_visible(False)
    else:
        ax.set_yticks(x)
        ax.set_yticklabels(bins)            
        ax.grid(visible=True, which='major', linestyle='--', axis='x')
        ax.yaxis.set_tick_params(gridOn=False) 
        ax.yaxis.set_tick_params(which='minor', bottom=False)
        if hide_bottom:
            ax.spines['left'].set_visible(False)
    
    ax.set_axisbelow(True)
    ax.minorticks_on()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return bs


#%%---------------------------------------------------------------------------#
def scale_radar(array, n=6, kind='max'):
    
    from matplotlib.ticker import MaxNLocator, LinearLocator
    
    
    if array.ndim == 1:
        
        array = np.atleast_2d(array).T
        
    if kind == 'linear':
        locator = LinearLocator(n)
    else:
        locator = MaxNLocator(n)
        
    tvalues = ([locator.tick_values(*aminmax) for aminmax
                        in zip(array.min(axis=0), array.max(axis=0))])
        
    aminmax = np.vstack((arr.min(), arr.max()) for arr in tvalues)
    amins = aminmax[:, 0]
    aptps = aminmax.ptp(axis=1)

    
    ticks = np.vstack(tvalue[:n] for tvalue in tvalues)
    
    arrays = (array - amins) / aptps
    
    return arrays, ticks

#%%---------------------------------------------------------------------------#
def radar(titles, **title_opts):

    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.collections import LineCollection


    def unit_poly_verts(theta):
        """Return vertices of polygon for subplot axes.
    
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def draw_poly_patch(self):
        
        verts = unit_poly_verts(THETA)
        
        return plt.Polygon(verts, closed=True, edgecolor='k')
    
    TITLES = list(titles)   
    THETA = np.linspace(0, 2*np.pi, len(TITLES), endpoint=False) + np.pi/2 
    FRAME = title_opts.pop('frame', 'polygon')
    
    class RadarAxes(PolarAxes):
    
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            
            self.dim = len(TITLES)
            
            self.theta = np.linspace(0, 2*np.pi, self.dim+1) + np.pi/2
            
            super().__init__(*args, **kwargs)
                        
            super().grid(False)

            self.rgrid() 
            
            title_opts_ = dict(fontsize=12, weight="bold", color="black")
            
            title_opts_.update(title_opts)
            self.set_thetagrids(np.degrees(THETA).astype(np.float32), 
                                labels=TITLES, **title_opts_) 

            self.set_xlim(np.pi/2, np.pi*2 + np.pi/2)
            self.set_ylim(0, 1)


            self.set_yticklabels([])
            for tick, d in zip(self.xaxis.get_ticklabels(), THETA):

                if np.pi / 2 < d < np.pi * 1.5:
                    tick.set_ha('right')
                elif np.pi * 1.5 < d:
                    tick.set_ha('left')

        def _gen_axes_patch(self):
            
            return draw_poly_patch(self)
        
    
        def _gen_axes_spines(self):
            if FRAME == 'circle':
                return super()._gen_axes_spines()
            else:
                # The following is a hack to get the spines (i.e. the axes frame)
                # to draw correctly for a polygon frame.
        
                # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
                spine_type = 'circle'
                verts = unit_poly_verts(THETA)
                # close off polygon by repeating first vertex
                verts.append(verts[0])
                path = Path(verts)
        
                spine = Spine(self, spine_type, path)
                spine.set_transform(self.transAxes)
                
                return {'polar': spine}
            
        def rgrid(self, b=True, count=5, **kwargs):
            
            if hasattr(self, '_rgrids'):
                for col in self._rgrids:
                    
                    col.remove()
            
            if b:
            
                defaults = dict(color='grey', lw=0.5)       
                defaults.update(kwargs)
                
                dy = 1 / count
                            
                ys = np.ones_like(self.theta) * np.arange(dy, 1+dy, dy).reshape(-1, 1)
                
                xs = np.tile(self.theta, (count, 1))
                
                xys = np.dstack([xs, ys])
        
                line_segments = LineCollection(xys, **defaults)
                
                line_colls1 = self.add_collection(line_segments)
                
                xs = np.tile(THETA.reshape(-1, 1), 2)
                ys = np.tile([0, 1], (self.dim, 1))
                
                xys = np.dstack([xs, ys])
                line_segments = LineCollection(xys, **defaults)
                
                line_colls2 = self.add_collection(line_segments)
                                
                self._rgrids = (line_colls1, line_colls2)

        def set_zebra(self, **kwargs):
            
            assert hasattr(self, '_rgrids')
            
            defaults = dict(color='grey', alpha=0.2)
            
            defaults.update(kwargs)
            arr = np.dstack([p.vertices for p in self._rgrids[0].get_paths()])
            self._rgrids[0].set_visible(defaults.pop('edge', False))
            xs, ys = arr[:, 0, :], arr[:, 1, :]
            
            xs = xs.T
            ys = ys.T
            

            return [self.fill_between(xs[2*i], ys[2*i], ys[2*i+1], 
                        zorder=0, **defaults) for i in range(len(xs) // 2)]
            
        def get_theta(self, title):
            
            return THETA[TITLES.index(title)]
        
        def scale(self, array, *arg,  **kwargs):
            
            assert hasattr(self, '_rgrids')

            origin = kwargs.pop('include_origin', False)
            apply = kwargs.pop('apply_tick', False)
            kind = kwargs.pop('kind', 'max')
            
            arr = np.dstack(p.vertices for p in self._rgrids[0].get_paths())

            _, _, n = arr.shape
            

            arrays, ticks = scale_radar(array, n=n+1, kind=kind)
            
            if not origin:
                
                ticks = ticks[:, 1:]

            
            if apply:
                
                for title, labels in zip(TITLES, ticks):
                    
                    
                    self.set_rlabel(title, labels, include_origin=origin, **kwargs)
            
            return arrays, ticks
            
        def set_rlabel(self, title, labels, **kwargs):
            
            def get_txt():
                
                for xy, lbl in zip(xys, labels):
                    
                    yield self.text(*xy, fmt(lbl), **kwargs)
                    
            assert hasattr(self, '_rgrids')
            assert hasattr(labels, '__iter__')
            
            fmt = kwargs.pop('fmt', str)
            origin = kwargs.pop('include_origin', False)
            
            array = np.dstack([p.vertices for p in self._rgrids[0].get_paths()])
            

            idx = TITLES.index(title)
            xys = array[idx].T

            if origin:

                xys = np.concatenate([[[0, 0]], xys], axis=0)            
            
            return list(get_txt())
        
        def plot(self, array, *args, **kwargs):
            
            clustered = kwargs.pop('clustered', False)            
            yarray = np.atleast_2d(np.asarray(array))
            
            assert np.all((yarray >= 0) & (yarray <= 1))
            num, dim = yarray.shape
            
            if dim != self.dim and num == self.dim:
                
                yarray = yarray.T
            
                num, dim = dim, num
            
            assert dim == self.dim
            
            yarray = np.concatenate([yarray, yarray[:, :1]], axis=1)

            
            if clustered:
                
                xys = np.dstack([np.tile(self.theta, (num, 1)), yarray])
                line_segments = LineCollection(xys, *args, **kwargs)
                
                lines = self.add_collection(line_segments)
            
            else:
                lines = super().plot(self.theta, yarray.T, *args, **kwargs)
            
            return lines
        
        def fill_lines(self, lines, **kwargs):
            
            colors = kwargs.pop('colors', [None]*len(lines))
            
            assert len(colors) == len(colors)
            
            for line, c in zip(lines, colors):
                
                color = line.get_color() if c is None else c
                super().fill(*line.get_xydata().T, color=color, **kwargs)
                
            return self
        
    register_projection(RadarAxes)
    
    return RadarAxes.name 

def plot_wordcloud(ax, text, **kwargs):
    """
    Plot a word cloud on the given axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the word cloud.
    - text (str): The text to generate the word cloud from.
    - **kwargs: Additional keyword arguments to customize the word cloud appearance.


                kwargs
                ----------
                font_path : string
                    Font path to the font that will be used (OTF or TTF).
                    Defaults to DroidSansMono path on a Linux machine. If you are on
                    another OS or don't have this font, you need to adjust this path.

                width : int (default=400)
                    Width of the canvas.

                height : int (default=200)
                    Height of the canvas.

                prefer_horizontal : float (default=0.90)
                    The ratio of times to try horizontal fitting as opposed to vertical.
                    If prefer_horizontal < 1, the algorithm will try rotating the word
                    if it doesn't fit. (There is currently no built-in way to get only
                    vertical words.)

                mask : nd-array or None (default=None)
                    If not None, gives a binary mask on where to draw words. If mask is not
                    None, width and height will be ignored and the shape of mask will be
                    used instead. All white (#FF or #FFFFFF) entries will be considerd
                    "masked out" while other entries will be free to draw on. [This
                    changed in the most recent version!]

                contour_width: float (default=0)
                    If mask is not None and contour_width > 0, draw the mask contour.

                contour_color: color value (default="black")
                    Mask contour color.

                scale : float (default=1)
                    Scaling between computation and drawing. For large word-cloud images,
                    using scale instead of larger canvas size is significantly faster, but
                    might lead to a coarser fit for the words.

                min_font_size : int (default=4)
                    Smallest font size to use. Will stop when there is no more room in this
                    size.

                font_step : int (default=1)
                    Step size for the font. font_step > 1 might speed up computation but
                    give a worse fit.

                max_words : number (default=200)
                    The maximum number of words.

                stopwords : set of strings or None
                    The words that will be eliminated. If None, the build-in STOPWORDS
                    list will be used. Ignored if using generate_from_frequencies.

                background_color : color value (default="black")
                    Background color for the word cloud image.

                max_font_size : int or None (default=None)
                    Maximum font size for the largest word. If None, height of the image is
                    used.

                mode : string (default="RGB")
                    Transparent background will be generated when mode is "RGBA" and
                    background_color is None.

                relative_scaling : float (default='auto')
                    Importance of relative word frequencies for font-size.  With
                    relative_scaling=0, only word-ranks are considered.  With
                    relative_scaling=1, a word that is twice as frequent will have twice
                    the size.  If you want to consider the word frequencies and not only
                    their rank, relative_scaling around .5 often looks good.
                    If 'auto' it will be set to 0.5 unless repeat is true, in which
                    case it will be set to 0.

                    .. versionchanged: 2.0
                        Default is now 'auto'.

                color_func : callable, default=None
                    Callable with parameters word, font_size, position, orientation,
                    font_path, random_state that returns a PIL color for each word.
                    Overwrites "colormap".
                    See colormap for specifying a matplotlib colormap instead.
                    To create a word cloud with a single color, use
                    ``color_func=lambda *args, **kwargs: "white"``.
                    The single color can also be specified using RGB code. For example
                    ``color_func=lambda *args, **kwargs: (255,0,0)`` sets color to red.

                regexp : string or None (optional)
                    Regular expression to split the input text into tokens in process_text.
                    If None is specified, ``r"\w[\w']+"`` is used. Ignored if using
                    generate_from_frequencies.

                collocations : bool, default=True
                    Whether to include collocations (bigrams) of two words. Ignored if using
                    generate_from_frequencies.


                    .. versionadded: 2.0

                colormap : string or matplotlib colormap, default="viridis"
                    Matplotlib colormap to randomly draw colors from for each word.
                    Ignored if "color_func" is specified.

                    .. versionadded: 2.0

                normalize_plurals : bool, default=True
                    Whether to remove trailing 's' from words. If True and a word
                    appears with and without a trailing 's', the one with trailing 's'
                    is removed and its counts are added to the version without
                    trailing 's' -- unless the word ends with 'ss'. Ignored if using
                    generate_from_frequencies.

                repeat : bool, default=False
                    Whether to repeat words and phrases until max_words or min_font_size
                    is reached.

                include_numbers : bool, default=False
                    Whether to include numbers as phrases or not.

                min_word_length : int, default=0
                    Minimum number of letters a word must have to be included.

                collocation_threshold: int, default=30
                    Bigrams must have a Dunning likelihood collocation score greater than this
                    parameter to be counted as bigrams. Default of 30 is arbitrary.

                    See Manning, C.D., Manning, C.D. and Schütze, H., 1999. Foundations of
                    Statistical Natural Language Processing. MIT press, p. 162
                    https://nlp.stanford.edu/fsnlp/promo/colloc.pdf#page=22

                Attributes
                ----------
                ``words_`` : dict of string to float
                    Word tokens with associated frequency.

                    .. versionchanged: 2.0
                        ``words_`` is now a dictionary

                ``layout_`` : list of tuples ((string, float), int, (int, int), int, color))
                    Encodes the fitted word cloud. For each word, it encodes the string,
                    normalized frequency, font size, position, orientation, and color.
                    The frequencies are normalized by the most commonly occurring word.
                    The color is in the format of 'rgb(R, G, B).'

                Notes
                -----
                Larger canvases make the code significantly slower. If you need a
                large word cloud, try a lower canvas size, and set the scale parameter.

                The algorithm might give more weight to the ranking of the words
                than their actual frequencies, depending on the ``max_font_size`` and the
                scaling heuristic.


    Returns:
    - matplotlib.axes.Axes: The axes with the word cloud plotted.
    """
    from wordcloud import WordCloud
    
    default = dict(background_color='black', colormap='viridis', height=600, width=1000,
                   contour_width=2, contour_color='steelblue')
    
    default.update(kwargs)
    wc = WordCloud(**default)
    wc.generate(text)
    
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    return ax
    
    
#%% main

if __name__ == '__main__':
    
    fig = plt.figure(dpi=120)
    
    ax = fig.add_subplot(111, projection=radar(['A', 'B', 'C', 'DYE', 'E', 'FICE']))
    ax.rgrid(lw=1, linestyle='--',  alpha=0.4)
    
    lines = ax.plot([0.5, 0.4, 0.6, 0.8, 0.6, 0.3], color='black')
    lines = ax.plot([0.5, 0.4, 0.8, 0.2, 0.6, 0.3], color='red')
    ax.fill_lines(lines, alpha=0.3, label='boo')
    ax.set_zebra( lw= 1)
    ax.set_rlabel('E', list('abcde'), color='b')
    ax.set_rlabel('A', list('cdefg'), color='b')
    ax.legend()
    plt.show()