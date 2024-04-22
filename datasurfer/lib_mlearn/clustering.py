def kmeans(X, n_clusters=3, **kwargs):
    
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(X)
    
    return kmeans
    
    
    
    
    
