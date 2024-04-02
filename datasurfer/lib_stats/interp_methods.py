import numpy as np
from datasurfer.datautils import arghisto, parse_data

#%%

def interp_linear1D(x0, y0, **kwargs):
    """
    Perform linear interpolation on 1D data.

    Parameters:
    x0 (array-like): The x-coordinates of the data points.
    y0 (array-like): The y-coordinates of the data points.

    Returns:
    callable: A function that performs linear interpolation on an input array.

    """
    from scipy.interpolate import interp1d
    f = interp1d(x0, y0, kind='linear', bounds_error=False, fill_value='extrapolate')

    return lambda arr: f(arr).ravel()

#%%
def interp_linearND(X, y, **kwargs):
    """
    Perform linear interpolation in N-dimensional space.

    Parameters:
    *vals: tuple of arrays
        The input arrays representing the coordinates of the points to be interpolated.
        The last array in *vals represents the values associated with the coordinates.
        At least three keys are required for interpolation.

    Returns:
    function:
        A lambda function that takes an array of coordinates and returns the interpolated values.

    Raises:
    AssertionError:
        If less than three keys are provided for interpolation.

    """
    from scipy.interpolate import LinearNDInterpolator
            
    mask = np.all(~np.isnan(X), axis=1).ravel() & (~np.isnan(y).ravel())
    
    X = X[mask]
    y = y[mask]
        
    f = LinearNDInterpolator(X, y, **kwargs)

    return lambda arr: f(arr).ravel()