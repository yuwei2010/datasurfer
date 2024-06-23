# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stats import fit_sin

#%%---------------------------------------------------------------------------#
alpha_fun = lambda x, y: np.arctan(np.polyfit(x, y, 1)[0])

origin_fun = lambda x, y: np.array([0, np.polyfit(x, y, 1)[1]]).reshape(-1, 1)

rot_fun = (lambda alpha: np.array([[np.cos(alpha), np.sin(alpha )],
                                   [-np.sin(alpha), np.cos(alpha)]],))

#%%---------------------------------------------------------------------------#
def rot_xy(x, y, alpha=None, origin=None):
    
    alpha = alpha_fun(x, y) if alpha is None else alpha     
    
    origin = origin_fun(x, y) if origin is None else origin
    
    xy = np.vstack([x.ravel(), y.ravel()])
    
    return rot_fun(alpha) @ (xy-origin) + origin

#%%---------------------------------------------------------------------------#
    
def get_ndis(xdata):
    
    pass



#%%---------------------------------------------------------------------------#
if __name__ == '__main__':
    
    
    plt.close('all')
    df = pd.read_csv('_CON.DE', index_col=0, parse_dates=True, header=None, squeeze=True)
    
    xx = np.arange(df.size)
    yy = df.values
    
    sinfit = fit_sin(xx, yy)
    
    sinfun = sinfit['fitfunc']
    
    linfun = np.poly1d(np.polyfit(xx, yy, 1))
    vtan, offset = np.polyfit(xx, yy, 1)
    
#    vtan = sinfit['slope']
    print(vtan)
    
    alpha = np.arctan(vtan)
    
    print(alpha)
    
    xx1, yy1 =  rot_xy(xx, yy)
    
    sinfit = fit_sin(xx1, yy1)
    
    sinfun = sinfit['fitfunc']
    
    print(sinfit)
    

    
    fig, ax = plt.subplots()
    
#    ax.plot(xx, yy)
#    
#    ax.plot(xx, linfun(xx))

    
    ax.plot(xx1, yy1)
    
    ax.plot(xx1, sinfun(xx1))
    
