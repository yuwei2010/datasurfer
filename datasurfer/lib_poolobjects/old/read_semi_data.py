# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:54:41 2021

@author: weiyu
"""


import pandas as pd
import time
import os

from fiana import FiAna


tparams = (
            (60*24,     365,    '1Y',  'D'),
            (60*24*3,   365*3,  '3Y',  'D'),
            (60*24*7,   365*10, '10Y', 'D'),
            (60,        60,     '60D', 's'),
            (30,        30,     '30D', 's'),
            (10,        10,     '10D', 's'),
            (1,         5,      '5D',  's'),
            (1,         1,      '1D',  's'),
        )


filename = 'Semiconductor Companies.xlsm'

df = pd.read_excel(filename, index_col=0)

mask = df['To be considered'] == 'Y'

df = df[mask]

for freq, days, txt, unit in tparams:


    out = None
    
    for _, (key, name) in df[['Ticker', 'Name']].iterrows():
        
        print(key)
        try:
            DF = FiAna(symbol=key, freq=freq, days=days).dataframe(title=name)
    
            
            DF.index = DF.index.round(unit)
            
    
        except KeyError as err:
            
            print('ERROR: ', str(err))
            continue 
        
        time.sleep(0.1)
        
        if out is None:
            
            out = DF
            
        
        else:
            
            out = out.join(DF, how='left').fillna(method='ffill')
    
            
    out.to_excel(os.path.join('DATA_{}.xlsx'.format(txt)))
    
    
    
        