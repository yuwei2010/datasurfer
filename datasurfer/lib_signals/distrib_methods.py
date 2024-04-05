

import numpy as np
#%%
def get_kde(x, **kwargs):
    """
    Compute the kernel density estimate (KDE) for a given dataset.

    Parameters:
    x (array-like): The input data.
    **kwargs: Additional keyword arguments to be passed to the `gaussian_kde` function.

    Returns:
    density (gaussian_kde): The kernel density estimate.

    """
    from scipy.stats import gaussian_kde
    
    density = gaussian_kde(x, **kwargs)
    # density.covariance_factor = lambda : .25
    density._compute_covariance()

    return density

#%%
def arghisto(data, bins):
    """
    Compute the histogram of the input data based on the given bins.

    Parameters:
    data (ndarray): Input data array.
    bins (ndarray): Bins for computing the histogram.

    Returns:
    list: List of arrays containing the indices of data points falling into each bin.
    """
    out = []
    dat = data.ravel()
       
    for idx in range(0, len(bins)-1):
        if idx == 0:
            out.append(np.where((bins[idx]<=dat) & (bins[idx+1]>=dat))[0])
        else:
            out.append(np.where((bins[idx]<dat) & (bins[idx+1]>=dat))[0])
        
    return out

#%%
if __name__ == '__main__':
    
    import numpy as np
    
    x = np.random.randn(1000)
    
    print(get_kde(x)(x))