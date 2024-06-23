# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from stock.stats import fit_sin, gaussfit_err, sum_gauss1d, gauss1d
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


#%%plot_kde2gauss
def plot_kde2gauss(xdata, gparams=None, slices=None, axs=None, legend=True,
                    volume=None, chunk_min=0.1, title=None, vcolor=None, diff=None, **kwargs):
    
    def set_axes(ax):
        
        ax.minorticks_on()
        ax.grid(ls='--', which='major', lw=0.5, color='g', alpha=1)
        ax.grid(ls=':', which='minor', lw=0.5, color='g', alpha=1)    
        
        
    def plot_ax1():
        
        for idx, (x, y, (a, b)) in enumerate(plot_data):
            
            flag = b-a > chunk_min
            
            mean, std, k = gparams[idx]
            color = colors[idx]
            ax1.plot(x, y, color=color, label='{} - {} = {}'.format(b, a, b-a), **ax1_settings)         
            ax1.plot(xx, mean*np.ones(xx.size), '-.', color=color, lw=1)
            
            ax1.bar(x[1:], np.diff(y), bottom=xarr.mean(), alpha=0.7, color=color)


            
            for n in range(1, 4):
                ax1.fill_between([a, b], 
                                 [mean+std*n]*2, [mean-std*n]*2, alpha=0.08*(4-n), color=color)   
                ax1.plot([a, b], [mean+std*n]*2, alpha=0.2, color=color)
                ax1.plot([a, b], [mean-std*n]*2, alpha=0.2, color=color)
                
            
            ax1.plot([a]*2, [mean-std*3, mean+std*3], color='k', lw=0.5)
            ds = pd.Series(y)
            for n in np.linspace(0, 1, 5):
                v = ds.quantile(n)
                ax1.plot([a, a+1], [v]*2, color='k', lw=1)
                if flag:
                    ax1.text(a+2, v, '{:0.2f}'.format(v), va='center')
                
            try:
             
                sinfit = fit_sin(x, y, lock_amp=True)                
                ax1.plot(x, sinfit['fitfunc'](x), '--', color=color, lw=0.5)
                ax1.plot(x, sinfit['meanfunc'](x), '--', color=color, lw=0.5)
                if flag:
                    ax1.text(a, mean+std*3, 'S{:0.2f}, A{:0.2f}, P{}'.format(
                        sinfit['slope'], 
                        abs(sinfit['amp']), 
                        int(sinfit['period'])), color='k', ha='left')
                
            except (TypeError, RuntimeError, ValueError):
                fitpars = np.polyfit(x, y, 1)
                fitfun = np.poly1d(fitpars)
                ax1.plot(x, fitfun(x), '--', color=color, lw=0.5)
                if flag:
                    ax1.text(a, mean+std*3, 'S{:0.2f}'.format(fitpars[0]), color='k', ha='left')
                
        ax1.plot(x[-1], y[-1], 'o', color=color, alpha=1,
         markeredgecolor='k', markeredgewidth=1)
        set_axes(ax1)
        
        ax1.set_xlim(0, xx.size)
        ax1.xaxis.set_major_locator(MaxNLocator(11))
        

        fig.canvas.draw()

        
        xlocs = ax1.get_xticks().astype(int)
        
        ax1.set_xticklabels([ax1_xticks[n] for n in xlocs if n<=xx.max()])
#        ax1.spines["left"].set_position(("axes", 0.5))
        
        ax = ax1.twiny()
        ax.minorticks_on()        
        ax.set_xlim(0, xx.size)
        
        if diff is not None:
            
            ax1.fill_between(np.arange(diff.size), diff+yy.min(), 
                             np.zeros_like(diff), alpha=0.4, color='grey')
        
        if volume is not None:
            
           
            
            vcol = 'k' if vcolor is None else vcolor
            
#            dv0 = np.asarray(volume)
#            
#            md = np.median(dv0)
            ax = ax1.twinx()
#            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2E'))
            
            ax.bar(np.arange(volume.size), volume, alpha=0.4, color=vcol, zorder=0)
            ax.set_ylim(0, volume.max()*5) 
#            ax.set_yticks([])
            ax.set_yticks([volume[-1]])
#            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                  
        ax = ax1.twinx()
        
#        ax.fill_between(xx[1:], np.diff(xarr)/xarr[:-1], np.zeros_like(xx[1:]), 
#                        lw=0.4, alpha=0.3, zorder=0, color='grey')
        
        yper = np.diff(xarr) / xarr[:-1]
        ax.fill_between(xx[1:], yper, np.zeros_like(yper), 
                lw=0.7, alpha=0.3, zorder=0, color='k')
        
#        yper2 = np.cumsum(np.diff(xarr)/xarr.min())
#        
#        ax.plot(xx[1:], yper2-yper2.min(), 
#                lw=0.7, alpha=0.3, zorder=0, color='k')
        
        ax.set_ylim((yy.min()-xarr.min())/xarr.min(), (yy.max()-xarr.min())/xarr.min())

        
#        ax.set_yticks(np.linspace(0, (yy.max()-xarr.min())/xarr.min(), 10))
        ax.yaxis.set_major_locator(MaxNLocator(11))
#        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in ax.get_yticks()])
        
#        ax.spines["right"].set_position(("axes", 1.1))
        ax.minorticks_on()  
        
    
    def plot_ax2():
        
        ax2.yaxis.tick_right()
        ax2.xaxis.tick_top()
        ax2.invert_xaxis()
        

        ax2.plot(kde_fun(yy), yy, color='black', label='gaussian_kde: {:0.4f}'.format(
                gaussfit_err(xarr, gparams, kde_fun)))  
        
        ax2.plot(kde_fun(xarr), xarr, 'o', color='r', alpha=0.2,)
    
        ax2.plot(kde_fun(xarr[-1]), xarr[-1], 'o', color='r', alpha=1,
                 markeredgecolor='k', markeredgewidth=1)

        ax2.text(kde_fun(xarr[-1]), xarr[-1], '{:0.2f}'.format(xarr[-1]), ha='left')

        ax2.fill(sum_gauss1d(yy, *gparams), yy, color='black', alpha=0.1)
        
        xmin, _ = ax2.get_xlim()
        colors = []
        
        for ii, (mean, std, k) in enumerate(gparams):
            
            
            xx = gauss1d(yy, mean, std, k)
            
            l, = ax2.plot(xx, yy, 
                          label="N({:0.2f}, {:0.2f}) x {:0.2f}".format(mean, std, k), alpha=0.5)
            
            color = l.get_color()
            colors.append(color)
            ax2.fill(xx, yy, alpha=0.3, color=color)
            ax2.plot([xmin, 0], [mean]*2, '-.', color=color, lw=1)
            ax2.text(0, mean, '{:0.2f}'.format(mean), ha='right')
            ax2.text(0, mean, ' {}'.format(str(ii+1)), color=color)
            
            

        
        ax2.set_xlim(xmin, 0)
        ax2.set_ylim(yy.min(), yy.max())
        set_axes(ax2)

        if volume is not None:

            ax = ax2.twiny()
            ax2.xaxis.tick_top()
            ax.xaxis.tick_bottom()
            ax.minorticks_on()
            
            std = volume.std()
            mean = volume.mean()
            
            xxx = np.linspace(mean-3*std, mean+3*std, 200)
            yyy = gaussian_kde(volume)(xxx)
            
            
            ax.plot(xxx, yyy/yyy.max()*(yy.max()-yy.min())*0.1+yy.min(), lw=1, color='k')
            ax.fill(xxx, yyy/yyy.max()*(yy.max()-yy.min())*0.1+yy.min(), alpha=0.1, color='k')
            ax.set_xticks(np.quantile(volume, np.linspace(0.25, 0.75, 3)))
#            ax.set_xticklabels(['{:0.1e}'.format(x) for x in ax.get_xticks()])
            ax.ticklabel_format(style='sci', axis='x',scilimits=(0,0))
            plt.xticks(rotation=30, ha='right')
            ax.invert_xaxis()
            ax.set_xlim(None, 0)
        
        
        return colors
        
        
    
    
    if not isinstance(xdata, pd.Series):
        
        xdata = pd.Series(np.asarray(xdata))

    xx = np.arange(xdata.size)
    xarr = xdata.values   
    if isinstance(xdata.index, pd.DatetimeIndex):
        ax1_xticks = xdata.index.strftime('%y/%m/%d\n%H:%M')
    else:
        ax1_xticks = xx

    
    yy = np.linspace(xarr.min()-xarr.ptp()/2, xarr.max()+xarr.ptp()/2, 200)
    
    kde_fun = gaussian_kde(xarr)
    
    if axs is None:
        
        fig = plt.figure(**kwargs)
        fig.subplots_adjust(wspace=0, left=0.05, right=0.95, bottom=.1, top=.95) #bottom=.05, top=.95, left=0.05, right=0.95,
        
        ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((1, 4), (0, 3), sharey=ax1) 

    else:        
        ax1, ax2 = axs
        
    if gparams is None:
        
        gparams = [[xarr.mean(), xarr.std(), 1]]
    
    if slices is None:
        
        plot_data = [[xx, xarr, (0, xx.size)]]
        
    else:
        
        plot_data = [(xx[slice(*s)], xarr[slice(*s)], s) for s in slices]
        
    ax1_settings = dict(marker='o', alpha=0.7, 
          ls='none', markeredgecolor='k', markeredgewidth=0.5) 
    
    if 0 < chunk_min < 1:
        
        chunk_min = int(chunk_min*xx.size)
    
    
    
    colors = plot_ax2()
    plot_ax1()
    
    if legend:
        
        ax2.legend(loc=1, title=title)
        ax1.legend(loc=9, ncol=5, 
                title='min: {:0.2f}, max: {:0.2f}, mean: {:0.2f}, median: {:0.2f}, last value: {:0.2f}'.format(
                xarr.min(), xarr.max(), xarr.mean(), np.median(xarr), xarr[-1]))
        
    return ax1, ax2



    
#%%---------------------------------------------------------------------------%
if __name__ == '__main__':
    
    from stats import cluster_kde2gauss
    from fiana import Indicator
    import matplotlib.cm as cm
    
    DF = pd.read_csv('SHARES.DE', index_col=0, parse_dates=True, header=[0, 1])
    
    
    do = DF.xs('open', level=1, axis=1)
    dc = DF.xs('close', level=1, axis=1)
    dh = DF.xs('high', level=1, axis=1)
    dl = DF.xs('low', level=1, axis=1)
    dv = DF.xs('volume', level=1, axis=1)
    

    keys = ['DAI.DE']
    
    for key in keys:
        
        print(key)
        

        do0 = do[key].dropna()
        dc0 = dc[key].dropna()
        dh0 = dh[key].dropna()
        dl0 = dl[key].dropna()
        dv0 = dv[key].dropna()   
        vcolor = cm.bwr(np.sign(dc0-do0))

                
        gparams, slices = cluster_kde2gauss(dc0, tol=0.1, Ns=30)


            
        ax1, ax2 = plot_kde2gauss(dc0, gparams, volume=dv0, slices=slices, vcolor=vcolor,
                              figsize=(12, 6), dpi=120, diff=(dh0-dl0).values*1)   
        
#        ax = ax1.inset_axes([0.2, 0.5, 0.2, 0.2])
#        ax.patch.set_alpha(0)
        
#        plt.savefig('_'+key+'.png')
        
#        plt.close()
        
        print(gparams, slices)    
