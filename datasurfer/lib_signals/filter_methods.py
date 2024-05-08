

def savgol_filter(data, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):

    from scipy.signal import savgol_filter
    
    return savgol_filter(data, window_length, polyorder, deriv=deriv, delta=delta, axis=axis, mode=mode, cval=cval)