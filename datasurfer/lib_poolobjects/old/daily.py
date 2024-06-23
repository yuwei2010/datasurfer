# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:38:09 2020

@author: weiyu
"""


# -*- coding: utf-8 -*-
import sys
import io

sys.path.insert(0, r'D:\01_Python')
import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from shares import DAX, MDAX, SDAX, GREEN, GOLD

from fiana import FiAna

from stats import cluster_kde2gauss
from visual import plot_kde2gauss

from datetime import datetime

from translater.util_trans import Translator

from translater.BaiduTranslate import Dict

tr = Translator()
bd = Dict()


redo = False

root = os.path.join(r'Analyse_Data', datetime.today().strftime('%Y%m%d') )

if not os.path.lexists(root):
    os.mkdir(root)

tparams = (
            (60*24,     365,    '1Y',   40),
            (60*24*3,   365*3,  '3Y',   20),
            (60*24*7,   365*10, '10Y',  30),
            (60,        60,     '60D',  20),
            (10,        30,     '30D',  30),
            (10,        10,     '10D',  30),
            (1,         5,      '5D',   30),
            (1,         1,      '1D',   30),
        )

grps = [
        (0, GOLD, 'GOLD'),
        (1, DAX, 'DAX'), 
        (2, MDAX, 'MDAX'),
        (3, SDAX, 'SDAX'),
        (4, GREEN, 'GREEN'),
        ]


        
    
for idx, grp, gname in grps:  
    
    for key, name in grp.items():
        
        print(idx, key)
        
        
        
        
        for freq, days, txt, ns in tparams:
            

            
            try:
                
                
                df_file = os.path.join(root, 'DATA_{}.csv'.format(txt))
                
                if os.path.lexists(df_file):
                    
                    df0 = pd.read_csv(df_file, index_col=0, parse_dates=True, header=[0, 1])
                    

                    if key not in df0.columns.get_level_values(0):

                        DF = FiAna(symbol=key, freq=freq, days=days).dataframe()
                        df0 = pd.concat([df0, DF], axis=1)
                    
                else:
                    DF = FiAna(symbol=key, freq=freq, days=days).dataframe()
                    df0 = DF
                    
                

                    
                df0.to_csv(df_file)
                time.sleep(0.1)
                
#                raise
            except (ValueError, RuntimeError, KeyError) as err:
                
                raise
                print('ERROR: ', str(err))


#%%
                
                
for idx, grp, gname in grps:
    
    subroot = os.path.join(root, gname)
    
    if not os.path.lexists(subroot):
        
        os.mkdir(subroot)
        
    for freq, days, txt, ns in tparams:
        
        filename = os.path.join(subroot, '{}_'.format(idx)+key+'_{}.png'.format(txt))
        
        if not redo and os.path.lexists(filename):
            
            continue
        
        plt.close('all')
        
        try:
            df_file = os.path.join(root, 'DATA_{}.csv'.format(txt))
            
            if os.path.lexists(df_file):
                
                df0 = pd.read_csv(df_file, index_col=0, parse_dates=True, header=[0, 1])
                
                for key, name in grp.items():
                    
                    print(idx, key)
                    
                    DF = df0[key]

                    do = DF['open']
                    dc = DF['close']
                    dh = DF['high']
                    dl = DF['low']
                    dv = DF['volume']
                    
                    do0 = do.dropna()
                    dc0 = dc.dropna()
                    dh0 = dh.dropna()
                    dl0 = dl.dropna()
                    dv0 = dv.dropna()
                
                    print(idx, key, freq, days, dc0.size)
                
    
                    vcolor = cm.bwr(np.sign(dc0-do0))
                    gparams, slices = cluster_kde2gauss(dc0, tol=0.1, Ns=ns)
                    
                    ax1, ax2 = plot_kde2gauss(dc0, gparams, volume=dv0, slices=slices, vcolor=vcolor,
                                          figsize=(12, 6), dpi=120, title=name+' '+txt, diff=(dh0-dl0).values)  
                    
                    plt.savefig(filename)
            
        except (ValueError, RuntimeError, KeyError) as err:
            
            raise
            print('ERROR: ', str(err))