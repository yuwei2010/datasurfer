from functools import wraps
from datasurfer.lib_mlearn import output_switcher


#%%
def cluster_template(func):
            
    @wraps(func)
    @output_switcher
    def wrapper(X, n_clusters, **kwargs):

        mdl = func(X, n_clusters=n_clusters, **kwargs)
        
        return mdl, lambda :mdl.predict(X)
        
    return wrapper
    

#%%
def kmeans(X, n_clusters=3, **kwargs):
    
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(X)
    
    return kmeans

#%%

def spectral(X, n_clusters=3, **kwargs):
    
    from sklearn.cluster import SpectralClustering
    
    sc = SpectralClustering(n_clusters, **kwargs).fit(X)
    
    return sc

#%%
def DBSCAN(X, **kwargs):
    from sklearn.cluster import DBSCAN
    
    db = DBSCAN(**kwargs).fit(X)
    
    return db

#%%

def gaussian(X, n_clusters=3, **kwargs):
    
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_clusters, **kwargs).fit(X)
    
    return gm

#%%
def bayesian(X, n_clusters=3, **kwargs):
    from sklearn.mixture import BayesianGaussianMixture
    bg = BayesianGaussianMixture(n_components=n_clusters, **kwargs).fit(X)
    
    return bg
    

#%%

__all__ = [kmeans, spectral, gaussian, bayesian]

#%%

class Cluster(object):
   
    def __new__(cls, *args, **kwargs):
        
        instance = super().__new__(cls)
        
        for func in __all__:
            
            setattr(instance, func.__name__, cluster_template(func))
     
        return instance
    
    @output_switcher
    def DBSCAN(self, X, **kwargs):
        
        mdl = DBSCAN(X,  **kwargs)
        
        return mdl, lambda :mdl.labels_
        

#%%
        
    
    
    
    
    
    
    
    

# %%
