
import numpy as np
from datasurfer.datautils import arghisto, parse_data

#%%

class Stats(object):
    """
    A class for performing statistical computations.

    Attributes:
    dp: The data object.

    """

    def __init__(self, dp) -> None:
        self.dp = dp

    @parse_data
    def arghisto(self, val, *, bins, **kwargs):
        """
        Compute the histogram of a given value.

        Parameters:
        val (array-like): The input values.
        bins (int): The number of bins to use for the histogram.

        Returns:
        numpy.ndarray: The computed histogram.

        """
        return arghisto(val, bins)
    
    def corr(self, *keys, **kwargs):   
        assert len(keys) >= 2, 'At least two keys are required for correlation computation.'
        return self.dp[keys].corr(**kwargs)
    
    @parse_data
    def interp_linearND(self, *vals, **kwargs):
   
        from scipy.interpolate import LinearNDInterpolator

        assert len(vals) >= 3, 'At least three keys are required for interpolation.'
        
        X = np.vstack(vals[:-1]).T
        y = np.atleast_2d(vals[-1].ravel()).T
               
        mask = np.all(~np.isnan(X), axis=1).ravel() & (~np.isnan(y).ravel())
        
        X = X[mask]
        y = y[mask]
           
        f = LinearNDInterpolator(X, y)
  
        return lambda arr: f(arr).ravel()
    
    
            
            
            
            
            
        
 
        
    
# %%
