import numpy as np
import matplotlib.pyplot as plt
from collections import abc
from datasurfer.lib_plots.plot_collection import plot_histogram

class Stat_Plots(object):
    
    def __init__(self, dp):
        
        self.dp = dp
        
    def histogram(self, *keys, ax=None, bins=None, **kwargs):
        
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