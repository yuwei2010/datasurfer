
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
    def interp_linearND(self, *vals, **kwargs):
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
        from datasurfer.lib_stats.interp_methods import interp_linearND

        assert len(vals) >= 3, 'At least three keys are required for interpolation.'

        lbls = kwargs.pop('labels', None)

        X = np.vstack(vals[:-1]).T
        y = np.atleast_2d(vals[-1].ravel()).T

        f = interp_linearND(X, y, **kwargs)

        return f
    
    
            
            
            
            
            
        
 
        
    
# %%
