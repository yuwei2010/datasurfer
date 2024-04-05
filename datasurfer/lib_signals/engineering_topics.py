import json
import numpy as np
from pathlib import Path
from functools import singledispatch


def calc_heatdump_duration(compkey, Tarray, Tfinal, threshold_dT=0.5, 
                  select_method='max', xstar=np.arange(-36000, 36000, 0.1)):
    """
    Calculate heatdump based on the given parameters.

    Parameters:
    compkey (str): The component key.
    Tarray (array-like): The simulation temperature data.
    Tend (float): The end temperature.
    threshold_dT (float, optional): The threshold temperature difference. Defaults to 1e-3.
    select_method (str, optional): The method to select heatdump. Defaults to 'max'.
    xstar (array-like, optional): The x values for fitting. Defaults to np.arange(-1e4, 1e4, 0.1).

    Returns:
    tuple: A tuple containing xstar and ystar arrays.
    """
    
    Tarray = np.asarray(Tarray)  
    Tarray = Tarray - Tfinal 
    
    dict_coeffis = json.load(open(Path(__file__).parent / 'heatdump_fitting_coeff.json'))    
    funfit = lambda x, a, b, c, d: a * np.exp(-b * x + d) + c    
    
    coeffis = dict_coeffis[compkey]   
    ystar = funfit(xstar, *coeffis)[::-1]    
    ystar = ystar - ystar.min()
    
    if select_method == 'max':
        Tmax = Tarray.max()
    elif select_method == 'last':
        Tmax = Tarray[-1]
        
    assert np.abs(ystar-Tmax).min() < 1
    
    idx_start = np.abs(ystar-threshold_dT).argmin()
    idx_end = np.abs(ystar-Tmax).argmin()
    
    assert idx_start < idx_end
    
    time = xstar[idx_start:idx_end]
    time = time-time.min()
    
    Temp = ystar[idx_start:idx_end]
    Temp = Temp + Tfinal
    
    return time, Temp[::-1]
    
    
    
    

