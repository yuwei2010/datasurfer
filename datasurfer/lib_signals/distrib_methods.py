
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
if __name__ == '__main__':
    
    import numpy as np
    
    x = np.random.randn(1000)
    
    print(get_kde(x)(x))