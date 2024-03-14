import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from collections import abc
from functools import wraps
from datasurfer.lib_plots.plot_collection import plot_histogram, plot_dendrogram, plot_parallel_coordinate
from datasurfer.lib_plots.plot_utils import parallel_coordis

figparams = {'figsize': (8, 6), 
             'dpi': 120,}

def set_ax(ax):
    
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(which='major', ls='--')
    ax.grid(which='minor', ls=':')
    return ax
    
axisfunc = set_ax

def define_ax(func):
    
    @wraps(func)
    def wrapper(self, *keys, **kwargs): 
          
        ax = kwargs.pop('ax', None)
        setax = kwargs.pop('setax', False)
        
        if not ax:
            _, ax = plt.subplots(**figparams)        
        
        ax = func(self, *keys, ax=ax, **kwargs)
        if setax:
            axisfunc(ax)
        
        return ax
    return wrapper
    
def parse_data(func):   
    @wraps(func)
    def wrapper(self, *keys, **kwargs):
        
        def get(keys):
            out = []
            lbls = []
            for key in keys:
                if isinstance(key, str):
                    out.append(self.dp[[key]].dropna().to_numpy().ravel())
                    lbls.append(key)
                elif isinstance(key, pd.Series):
                    out.append(key.dropna().to_numpy())
                    lbls.append(key.name)
                elif isinstance(key, pd.DataFrame):
                    out.extend(key.dropna().to_numpy().T)
                    lbls.extend(key.columns)
                elif isinstance(key, np.ndarray):
                    out.append(key)
                    lbls.append(None)
                elif isinstance(key, abc.Sequence):
                    o, ls = get(key)
                    out.append(o)
                    lbls.extend(ls)
                else:
                    raise ValueError('keys must be strings or numpy arrays')
                
            return out, lbls
        
        if all(isinstance(key, str) for key in keys):
            out = self.dp[keys].dropna().to_numpy().T    
            lbls = keys
        else:        
            out, lbls = get(keys)
        
        if ('labels' not in kwargs) and all(lbl is not None for lbl in lbls) :
            kwargs['labels'] = lbls     

        return func(self, *out, **kwargs)
    
    return wrapper
    

class Plots(object):
    """
    A class for generating statistical plots.
    
    Parameters:
    - dp: A pandas DataFrame containing the data.
    """
    
    def __init__(self, dp=None):
        """
        Initialize the Stat_Plots object.
        
        Parameters:
        - dp: A pandas DataFrame containing the data.
        """
        self.dp = dp
    
    def set_figparam(self, **kwargs):
        """
        Set the figure parameters for the plots.
        
        Parameters:
        - **kwargs: The keyword arguments to be passed to the matplotlib figure function.
        """
        global figparams
        figparams = kwargs
        return self
    
    def set_axisfunc(self, func):
        """
        Set the function to be used for formatting the axes of the plots.
        
        Parameters:
        - func: The function to be used for formatting the axes of the plots.
        """
        assert callable(func), 'func must be a callable function'
        global axisfunc
        axisfunc = func
        return self
        
    @define_ax   
    @parse_data
    def histogram(self, *keys, ax=None, bins=None, **kwargs):
        """
        Generate a histogram plot.
        
        Parameters:
        - keys: The column names of the data to plot. If keys are strings, the corresponding columns will be used. 
                If keys are arrays, the arrays will be used directly.
        - ax: The matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        - bins: The number of bins to use in the histogram. If None, a default of 10 bins will be used.
                If bins is an int, it specifies the number of equal-width bins in the given range.
                If bins is a sequence, it defines the bin edges, including the rightmost edge.
        - **kwargs: Additional keyword arguments to be passed to the plot_histogram function.
        
        Returns:
        - ax: The matplotlib Axes object containing the histogram plot.
        """
        if all(isinstance(key, str) for key in keys):
            data = self.dp[keys].dropna().to_numpy().T
        else:
            data = keys

        
        if ax is None:           
            _, ax = plt.subplots()
            
            
        if bins is None:
            bins = np.linspace(np.min(data), np.max(data), 10)
        
        elif isinstance(bins, (abc.Sequence, np.ndarray)):
            bins = np.asarray(bins)
        
        elif isinstance(bins, int):
            bins = np.linspace(np.min(data), np.max(data), bins) 
              
        else:
            raise ValueError('bins must be an int or a sequence of values')

        
        plot_histogram(ax, keys, bins, **kwargs)
               
        return ax
    
    @define_ax
    @parse_data
    def scatter(self, *keys, ax=None, setax=True, **kwargs):
        """
        Create a scatter plot.

        Parameters:
        - keys: The data to be plotted. Must contain 2-4 elements.
        - ax: The matplotlib Axes object to plot on. If not provided, a new figure and axes will be created.
        - setax: A boolean indicating whether to set the x and y axis labels. Default is True.
        - labels: A list of labels for the x and y axis. If provided, the x and y axis labels will be set accordingly.
        - kwargs: Additional keyword arguments to be passed to the scatter function.

        Returns:
        - ax: The matplotlib Axes object containing the scatter plot.
        """
        labels = kwargs.pop('labels', None) 
        
        if len(keys) == 2:
            ax.scatter(keys[0], keys[1], **kwargs)
        elif len(keys) == 3:
            ax.scatter(keys[0], keys[1], c=keys[2], **kwargs)   
        elif len(keys) == 4:
            ax.scatter(keys[0], keys[1], c=keys[2], s=keys[3], **kwargs)    
        else:
            raise ValueError('keys must contain 2-4 elements')
        
        if labels:
            ax.set_xlabel(labels[0])   
            ax.set_ylabel(labels[1])
        
        return ax
    
    @define_ax
    @parse_data
    def line(self, *keys, ax=None, **kwargs):
        """
        Plot a line graph.

        Parameters:
        - keys: Tuple of two elements representing the x and y values.
        - ax: Optional matplotlib Axes object to plot on.
        - labels: Optional tuple of x and y axis labels.

        Returns:
        - ax: The matplotlib Axes object with the line graph plotted.
        """
        labels = kwargs.pop('labels', None)
        
        if len(keys) == 2:
            ax.plot(keys[0], keys[1], **kwargs)           
        else:
            raise ValueError('keys must contain 2 elements')
        
        if labels:
            ax.set_xlabel(labels[0])   
            ax.set_ylabel(labels[1])        
        return ax
     
    @define_ax
    @parse_data
    def dendrogram(self, *keys, ax=None, **kwargs):
        """
        Generate a dendrogram plot.

        Parameters:
            *keys: Variable length argument list of keys.
            ax: Optional matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments to pass to the plot_dendrogram function.

        Returns:
            ax: The matplotlib Axes object containing the dendrogram plot.
        """
        labels = kwargs.pop('labels')
        df = pd.DataFrame(dict(zip(labels, keys)))        
        plot_dendrogram(ax, df.dropna(), **kwargs)       
        return ax
    
    @define_ax
    @parse_data
    def parallel_coordinate(self, *keys, ax=None, **kwargs):
        """
        Plots a parallel coordinate plot based on the given keys.

        Parameters:
            *keys: The keys used to create the parallel coordinate plot.
            ax (optional): The matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments to customize the plot.

        Returns:
            The matplotlib Axes object containing the parallel coordinate plot.
        """
        labels = kwargs.pop('labels')
        df = pd.DataFrame(dict(zip(labels, keys)))            
        default = dict(facecolor='none', lw=0.3, alpha=0.5, edgecolor='g')
        default.update(kwargs)        
        plot_parallel_coordinate(host=ax, df=df.dropna(), **default)
        
        return ax
    
    @define_ax
    @parse_data
    def parallel_coordis(self, *keys, **kwargs):
        labels = kwargs.pop('labels')
        df = pd.DataFrame(dict(zip(labels, keys)))               
        parallel_coordis(df.values.T, **kwargs)
        
        return self
    
    
       
if __name__ == '__main__':
    
    pass