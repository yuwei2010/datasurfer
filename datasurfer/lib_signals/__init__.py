import pandas as pd
import numpy as np
from datasurfer.datautils import parse_data, show_pool_progress

#%%

class Signal(object):
    """
    A class for performing statistical computations.

    Attributes:
    dp: The data object.

    """

    def __init__(self, dp) -> None:
        self.dp = dp
        
    def __call__(self, *keys, **kwargs):
        
        return self.dp[keys].describe(**kwargs)

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
        from datasurfer.lib_signals.distrib_methods import arghisto
        return arghisto(val, bins)
    
    @parse_data
    def kde(self, key, **kwargs):
        """
        Calculate the kernel density estimate (KDE) for a given key or array.

        Parameters:
        - key: The key for which to calculate the KDE.
        - **kwargs: Additional keyword arguments to be passed to the `get_kde` function.

        Returns:
        - The kernel density estimate for the given key.

        """
        from datasurfer.lib_signals.distrib_methods import get_kde

        kwargs.pop('labels', None)

        return get_kde(key, **kwargs)
    
    def corr(self, *keys, **kwargs):
        """
        Compute the correlation between two or more keys in the dataset.

        Parameters:
        - keys: Two or more keys to compute the correlation for.
        - kwargs: Additional keyword arguments to be passed to the `corr` method.

        Returns:
        - The correlation matrix between the specified keys.

        Raises:
        - AssertionError: If less than two keys are provided.

        Example usage:
        >>> dataset.corr('key1', 'key2', method='pearson')
        """
        assert len(keys) >= 2, 'At least two keys are required for correlation computation.'
        return self.dp[keys].corr(**kwargs)
    
    @parse_data
    def interp_linear(self, *vals, **kwargs):
        """
        Perform linear interpolation in N-dimensional space.

        Parameters:
        *vals: tuple
            Tuple of N arrays representing the coordinates of the points to be interpolated.
            The last array in the tuple represents the values at the given coordinates.
        **kwargs: dict, optional
            Additional keyword arguments to be passed to the interpolation method.

        Returns:
        f: object
            The interpolated function.

        Raises:
        AssertionError: If less than three keys are provided for interpolation.

        Example:
        >>> x = [1, 2, 3]
        >>> y = [4, 5, 6]
        >>> z = [7, 8, 9]
        >>> vals = (x, y, z)
        >>> interp_linearND(*vals)
        <interpolated function object>
        """
        from datasurfer.lib_signals.interp_methods import interp_linear1D, interp_linearND

        assert len(vals) >= 2, 'At least two inputs are required for interpolation.'

        kwargs.pop('labels', None)
        
        if len(vals) == 2:
            x, y = vals
            f = interp_linear1D(x, y, **kwargs)
        else:
            X = np.vstack(vals[:-1]).T
            y = np.atleast_2d(vals[-1].ravel()).T

            f = interp_linearND(X, y, **kwargs)

        return f
    
    @parse_data
    def fit_curve(self, *vals, f=None, **kwargs):
        
        from datasurfer.lib_signals.interp_methods import fit_curve
        
        assert len(vals) == 2, 'Only two inputs are required for curve fitting.'
        
        kwargs.pop('labels', None)
        
        if f is None:
            f = lambda x, a, b: a*x + b
            
        return fit_curve(f, *vals, **kwargs)
        
    @parse_data
    def polyfit(self, *vals, degree=1, **kwargs):
        
        
        assert len(vals) == 2, 'Only two inputs are required for curve fitting.'
        
        kwargs.pop('labels', None)
        
        return np.poly1d(np.polyfit(*vals, degree, **kwargs))
            
    def cdist(self, df, axis=0, pbar=True):
        
        from scipy.spatial import distance
        
        assert isinstance(df, pd.DataFrame), "Expect data frame object as input."
        
        XB = np.atleast_2d(df.values)
        keys = df.columns
        
        @show_pool_progress('Caculating', show=pbar)
        def get(self):
            for obj in self.objs:
                try:
                    XA = obj[keys].values
                    dist = distance.cdist(XA, XB).min(axis=axis)
                    yield obj.name, dist
                except (KeyError, RuntimeError):
                    yield
        
        res = dict(x for x in get(self.dp) if x)
        
        return pd.DataFrame.from_dict(res, orient='index').transpose()
    
    @parse_data
    def detect_lags(self, *vals, **kwargs):
        
        from scipy.signal import correlate
        from scipy.signal import correlation_lags
        
        x, y = vals
        
        # Cross-correlate 
        correlation = correlate(x, y, 'full')

        # Get the lag vector that corresponds to the correlation vector
        lags = correlation_lags(x.size,  y.size, mode="full")

        # Find the lag at the peak of the correlation
        lag = lags[np.argmax(correlation)]

        return lag

        
            
            
            
        
 
        
    
# %%
