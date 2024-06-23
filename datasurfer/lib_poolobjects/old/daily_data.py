# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:49:24 2020

@author: weiyu
"""


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

from processing import ProcessPool



from multiprocessing import Pool



redo = False

root = os.path.join(r'Analyse_Data', datetime.today().strftime('%Y%m%d') )

if not os.path.lexists(root):
    os.mkdir(root)

tparams = (
            (60*24,     365,    '1Y',   40),
            (60*24*3,   365*3,  '3Y',   20),
            (60*24*7,   365*10, '10Y',  30),
            (60,        60,     '60D',  20),
            (30,        30,     '30D',  30),
            (10,        10,     '10D',  30),
            (1,         5,      '5D',   30),
            (1,         1,      '1D',   30),
        )

grps = [
        (0, GOLD, 'GOLD'),
        (1, DAX, 'DAX'), 
        (2, MDAX, 'MDAX'),
        (3, SDAX, 'SDAX'),
#        (4, GREEN, 'GREEN'),
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
                        time.sleep(0.1)
                    
                else:
                    DF = FiAna(symbol=key, freq=freq, days=days).dataframe()
                    df0 = DF
                    time.sleep(0.1)
                    
                

                    
                df0.to_csv(df_file)
                
                
#                raise
            except (ValueError, RuntimeError, KeyError) as err:
                
#                raise
                print('ERROR: ', str(err))
                continue