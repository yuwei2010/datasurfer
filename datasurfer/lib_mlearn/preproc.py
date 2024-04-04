import numpy as np
#%%
def detect_outliers(X, **kwargs):
    
    from sklearn.neighbors import LocalOutlierFactor
   
    lof = LocalOutlierFactor(**kwargs)
    y_pred = lof.fit_predict(X)
    
    return y_pred
    