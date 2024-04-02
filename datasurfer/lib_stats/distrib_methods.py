
#%%
def get_kde(x, **kwargs):
    
    from scipy.stats import gaussian_kde
    
    density = gaussian_kde(x, **kwargs)
    # density.covariance_factor = lambda : .25
    density._compute_covariance()

    return density


if __name__ == '__main__':
    
    import numpy as np
    
    x = np.random.randn(1000)
    
    print(get_kde(x)(x))