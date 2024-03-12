import numpy as np
import matplotlib.pyplot as plt
from collections import abc
from datasurfer.lib_plots.plot_collection import plot_histogram

class Plots(object):
    """
    A class for generating statistical plots.
    
    Parameters:
    - dp: A pandas DataFrame containing the data.
    """
    
    def __init__(self, dp):
        """
        Initialize the Stat_Plots object.
        
        Parameters:
        - dp: A pandas DataFrame containing the data.
        """
        self.dp = dp
        
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
            data = np.array(keys)
        
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

        
        plot_histogram(ax, data, bins, **kwargs)
        
        return ax
    
if __name__ == '__main__':
    
    pass