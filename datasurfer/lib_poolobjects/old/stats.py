# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import logging

from functools import wraps
from scipy.integrate import quad
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit, least_squares, fmin, brute

#%%---------------------------------------------------------------------------%
def brute_cache(func):
    
    cache = {}
    
    @wraps(func)
    def wrapper(idx, *args, **kwargs):
        
        idx = np.unique(np.asarray(idx, dtype=int))
        key = tuple(idx)
        
        full_output = kwargs.get('full_output', False)
        
        if full_output:
            
            cache.clear()
            return func(idx, *args, **kwargs)
        
        elif key not in cache:
            cache[key] = func(idx, *args, **kwargs)
        
        return cache[key]
    
    return wrapper

#%%---------------------------------------------------------------------------%
def optfun_lsq(ks, xs, ys, gparams):

    return ((ys - sum(map(lambda n: gauss1d(xs, 
                        *gparams[n], ks[n]), range(len(ks)))))**2).sum()

#%%---------------------------------------------------------------------------%
@brute_cache
def optfun_brute_kfit(idx, xs, ys, chunk_min=2, full_output=False):
    
    
    rngs = np.vstack([(a, b) for a, b in  np.vstack(zip(np.concatenate([[0], idx]), 
                np.concatenate([idx, [len(xs)]]))) if b - a >= max(chunk_min, 2)])
    
    slices = [slice(*rng, 1) for rng in rngs] 
    
    
    gparams = np.vstack([(lambda arr: (arr.mean(), arr.std()))(xs[s]) for s in slices]) 

    
    x0 = 0.5 * np.ones(len(gparams))
    
    res = least_squares(optfun_lsq, x0, bounds=(0, 1), args=(xs, ys, gparams))
    
    if full_output:
    
        return np.hstack([gparams, res['x'].reshape(-1, 1)]), rngs
    else:
        return res['cost']
    
#%%---------------------------------------------------------------------------%
@brute_cache
def optfun_brute(idx, xs, ys, chunk_min=2, full_output=False):

    rngs = np.vstack([(a, b) for a, b in  np.vstack(zip(np.concatenate([[0], idx]), 
                np.concatenate([idx, [len(xs)]]))) if b - a >= max(chunk_min, 2)])
    
    slices = [slice(*rng, 1) for rng in rngs] 
    
    
    gparams = np.vstack([(lambda arr: (arr.mean(), arr.std(), 
                                    arr.size/xs.size))(xs[s]) for s in slices]) 
    
    cost = ((ys - sum_gauss1d(xs, *gparams))**2).sum()
   
    if full_output:
        return gparams, rngs

    else:
        return cost

#%%---------------------------------------------------------------------------%
def opt_ks(xs, ys, gparams):
    
    x0 = 0.5 * np.ones(len(gparams))
    
    res = least_squares(optfun_lsq, x0, bounds=(0, 1), args=(xs, ys, gparams))
    
    return res

#%%---------------------------------------------------------------------------%
def integrate(func, xdata, args=()):
    
    import warnings
    warnings.filterwarnings("ignore")
    
    xdata = np.asarray(xdata)
    
    a = xdata.min()
    b = xdata.max()
    
    return quad(func, a, b, args=tuple(args))[0]
            
#%%---------------------------------------------------------------------------%
def gauss1d(xdata, mu, sig, k=1):
    
    return abs(k) / (np.sqrt(2*np.pi*sig**2)) * np.exp(-(xdata-mu)**2/(2*sig**2))

#%%---------------------------------------------------------------------------%
def sum_gauss1d(xdata, *gparams):
    
    return sum(map(lambda p: gauss1d(xdata, *p), gparams))

#%%---------------------------------------------------------------------------%
def gaussfit_err(xs, gparams, pfun=None):
    
    if pfun is None:
        pfun = gaussian_kde(xs)
    
    s_kde = integrate(pfun, xs)
    delta = integrate(lambda x: abs(pfun(x)-sum_gauss1d(x, *gparams)), xs)  
    
    return delta / s_kde

#%%---------------------------------------------------------------------------%
def split_kde2gauss(xdata, xs=None, peaks=None, tol=0.05, try_max=20,
                    kde_kwargs=None, fit_kwargs=None, noise=1e-5):
    '''
        Split a kernel density estimation (kde) into several gaussian distributions
        
        return kde_func, gaussian parameters matrix (mean, std, factor), gaussian function, area center
        
    ''' 

    
    kde_kwargs = {} if kde_kwargs is None else kde_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    
    xs = xdata if xs is None else xs
    
    xs = np.asarray(xs)
    
    order = xs.argsort()
    
    pfun = gaussian_kde(xdata, **kde_kwargs)
    
    ys = pfun(xs)
    
    if peaks is None:
        
        from scipy.signal import find_peaks
        
        peaks = xs[order][find_peaks(ys[order])[0]]
    
    elif isinstance(peaks, int):
        
        peaks = np.linspace(xdata.min(), xdata.max(), peaks+2)[1:-1]
        
    x0 = np.concatenate([np.atleast_2d(peaks).T, 
                         np.ones((len(peaks), 2))], axis=1).ravel()
    
    optfun = lambda xs, *args: sum(map(
             lambda params: gauss1d(xs, *params), np.array(args).reshape(-1, 3)))
    
    popts, _ = curve_fit(optfun, xs, ys, p0=x0, **fit_kwargs)
    
    popts = popts.reshape(-1, 3)
    
    popts = popts[popts[:, -1] > noise]
    
    if tol :
        
        count = 0 
       

        while gaussfit_err(xs, popts, pfun) > tol and count < try_max:

            try:
                popts_ = split_kde2gauss(xdata, xs=xs, peaks=len(popts)+1, 
                    tol=None, kde_kwargs=kde_kwargs, fit_kwargs=fit_kwargs, noise=noise)
                means = popts_[:, 0]
                if np.any(means<xdata.min()) or np.any(means>xdata.max()):
                    
                    raise ValueError
                
                else:
                    popts = popts_
                    count += 1

                    
            except Exception:
                break                
    
    return popts

#%%---------------------------------------------------------------------------%
def cluster_kde2gauss(xdata, peaks=None, Ns=20, chunk_min=2, split_ranges=None,
                      kde_kwargs=None, fit_kwargs=None, tol=0.1, peak_max=5, kfit=False):
    
    kde_kwargs = {} if kde_kwargs is None else kde_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    
    if peaks is None and split_ranges is None:
        try:
            popts = split_kde2gauss(xdata, **fit_kwargs)
            peaks = len(popts)
        except RuntimeError:
            peaks = 1
    

    pfun = gaussian_kde(xdata, **kde_kwargs)
    
    xs = np.array(xdata)
    ys = pfun(xs)
    
    if split_ranges is None:
    
        rngs = np.tile(np.asarray([0, len(xdata)]), (max(peaks-1, 1), 1))
        
    else:
        
        rngs = np.asarray(split_ranges, dtype=int)
        
    

    if kfit:
        res = brute(optfun_brute_kfit, rngs, args=(xs, ys, chunk_min), 
                    Ns=Ns, full_output=False, finish=fmin) 
    else:

        res = brute(optfun_brute, rngs, args=(xs, ys, chunk_min), 
                    Ns=Ns, full_output=False, finish=fmin)
    
    gparams, slices = optfun_brute(res, xs, ys, chunk_min=chunk_min, full_output=True)
    
    err = gaussfit_err(xs, gparams, pfun)

    if tol and split_ranges is None and err > tol and peaks < peak_max:
        
        logging.warning('Residual {:0.2f} > {}. Using {} distributions to start a new iteration.'.format(
                err, tol, peaks+1))
        return cluster_kde2gauss(xdata, peaks=peaks+1, Ns=Ns, chunk_min=chunk_min, 
                      kde_kwargs=kde_kwargs, fit_kwargs=fit_kwargs, tol=tol, peak_max=peak_max)
    return gparams, slices

#%%---------------------------------------------------------------------------%
def split_cluster(xdata, slices, num=-1, **kwargs):

    s = slice(*slices[num])
    
    idx = np.arange(len(xdata))[s]
        
    xs = np.array(xdata)[s]
    
    _, ss = cluster_kde2gauss(xs, **kwargs)
    
    
    sid = np.concatenate([idx[np.unique(ss)[:-1]], [s.stop]], axis=0)
    
    sids = np.unique(np.concatenate([slices.ravel(), sid]), axis=0)
    
    pfun = gaussian_kde(xdata)
    gparams, slices = optfun_brute(sids[1:-1], xdata, pfun(xdata), full_output=True)


    return gparams, slices

#%%---------------------------------------------------------------------------%
def merge_cluster(xdata, slices, idx):

    idx = np.asarray(idx, dtype=int)   
    
    assert np.all(np.diff(np.sort(idx))==1)

    
    ss = np.unique(np.asarray(slices).ravel())
    
    ss = np.delete(ss, idx[-1])
    pfun = gaussian_kde(xdata)
    gparams, slices = optfun_brute(ss, xdata, pfun(xdata), full_output=True)
    
    return gparams, slices
    
    


#%%---------------------------------------------------------------------------#

def fft(x, y, dt=None):
    from scipy.fftpack import fft
    
    dt = x[1] - x[0] if dt is None else dt
    
    N = x.size
    xfreq = np.linspace(0, 0.5 / dt,  N // 2)
    y0freq = fft(y)
    yfreq = (2.0 / N * np.abs(y0freq[:N//2])) 

    return xfreq, yfreq

#%%---------------------------------------------------------------------------#
def fit_sin(tt, yy, lock_amp=True, **kwargs):
    
    '''Fit sin to the input time sequence, and return fitting parameters 
      "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    
    
#    from scipy.optimize import curve_fit
    
       
    tt = np.array(tt)
    yy = np.array(yy)
    
    if 'freq' in kwargs:
        
        guess_freq = kwargs.pop('freq')
    else:
        # excluding the zero frequency "peak", which is related to offset
        dt = kwargs.pop('dt', tt[1] - tt[0]) # assume uniform spacing
        freq, amp = fft(tt, yy, dt)
        
        guess_freq = freq[np.argmax(amp[1:])+1]   
    
    guess_amp = kwargs.pop('amp', np.std(yy) * 2.**0.5)
    guess_offset = kwargs.pop('offset', np.mean(yy))
    
    
    if lock_amp:
        b=0
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., 0, guess_offset])
                
        optfunc = lambda t, A, w, p, a, c:   (b*t + A) * np.sin(w*t + p) + a*t + c
        popt, pcov = curve_fit(optfunc, tt, yy, p0=guess)
        A, w, p, a, c = popt
    else:
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., 0, guess_offset, 0])
                
        optfunc = lambda t, A, w, p, a, c, b:   (b*t + A) * np.sin(w*t + p) + a*t + c       
        popt, pcov = curve_fit(optfunc, tt, yy, p0=guess)
        A, w, p, a, c, b = popt        


    f = w/(2.*np.pi)
    
    fitfunc = lambda t: (b*t+A) * np.sin(w*t + p) + a*t+ c
    sinfunc = lambda t: A * np.sin(w*t + p) + c
    meanfunc = lambda t: a*t + c
    limfunc = lambda t: b*t + A
    
    return {"amp": A, "omega": w, "phase": p, "offset": c, "slope": a, 'ampramp': b,
            "freq": f, "period": 1./f, "fitfunc": fitfunc, 
            'sinfunc': sinfunc,  
            'meanfunc': meanfunc,
            'limfunc': limfunc,
            "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
    
    



#%%---------------------------------------------------------------------------%

if __name__ == "__main__":
    
    import pandas as pd
    from fiana import Indicator
    from visual import plot_kde2gauss
    
    import matplotlib.cm as cm
    
    DF = pd.read_csv('SHARES.DE', index_col=0, parse_dates=True, header=[0, 1])
    
    
    dc = DF.xs('close', level=1, axis=1)
    dv = DF.xs('volume', level=1, axis=1)
    do = DF.xs('open', level=1, axis=1)
    
    keys = dc.columns[::-1]
    keys = ['CON.DE']
    
    for key in keys:
        
        print(key)
        
        
        dc0 = Indicator(dc[key].dropna()).split_apply('d',)
        dv0 = Indicator(dv[key].dropna()).split_apply('d',)
        do0 = Indicator(do[key].dropna()).split_apply('d',)
        
        print(dc0.size)
#        date = df0.index
        
        xdata = dc0

            
        plt.close('all')
        
        vcolor = cm.bwr(np.sign(dc0-do0))
        
        try:        
            gparams, slices = cluster_kde2gauss(xdata, tol=0.1, Ns=20)
        except ValueError:
            continue

        print(gaussfit_err(xdata, gparams))
            
        ax1, ax2 = plot_kde2gauss(xdata, gparams, volume=dv0, slices=slices, vcolor=vcolor,
                                  figsize=(12, 6), dpi=100)
        

        
#        plt.savefig('_'+key+'.png')
        
#        plt.close()
        
        print(gparams, slices)
        

   
    
    

    
    

    
    
    
    


