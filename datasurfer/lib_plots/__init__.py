import numpy as np
import matplotlib.pyplot as plt
from collections import abc
from datasurfer.lib_plots.plot_collection import plot_histogram



class Stat_Plots(object):
    
    def __init__(self, dp):
        
        self.dp = dp
        
    def histogram(self, *keys, ax=None, bins=None, **kwargs):
        
        data = self.dp[keys].to_numpy().T
        
        if ax is None:           
            _, ax = plt.subplots()
            
        if bins is None:
            bins = np.linspace(0, 9, 10)
        
        plot_histogram(ax, data, bins, **kwargs)
        
        return ax
    
if __name__ == '__main__':
    
    pass