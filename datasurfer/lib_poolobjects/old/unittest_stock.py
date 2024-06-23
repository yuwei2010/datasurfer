# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:20:36 2019

@author: WEI
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock import fidat


end = (2019, 7, 19)
days = 90
freq = 60
key = 'close'

symbols = [
            'ADS.DE',   # adidas
            'ALV.DE',   # Allianz
            'BAS.DE',   # BASF
            'BAYN.DE',  # Bayer
            'BEI.DE',   # Beiersdorf
            'BMW.DE',   # BMW
            'CON.DE',   # Continental
            '1COV.DE',  # Covestro
            'DAI.DE',   # Daimler
            'DBK.DE',   # Deutsche Bank
            'DB1.DE',   # Deutsche Börse
            'DPW.DE',   # Deutsche Post
            'DTE.DE',   # Deutsche Telekom
            'EOAN.DE',  # E.ON
            'FRE.DE',   # Fresenius
            'FME.DE',   # Fresenius Medical
            'HEI.DE',   # HeidelbergCement
            'HEN.DE',   # Henkel
            'IFX.DE',   # Infineon
            'LIN.DE',   # Linde
            'LHA.DE',   # Lufthansa
            'MRK.DE',   # Merck
            'MUV2.DE',  # Münchener
            'RWE.DE',   # RWE
            'SAP.DE',   # SAP
            'SIE.DE',   # Siemens
            'TKA.DE',   # thyssenkrupp
            'VNA.DE',   # Vonovia
            'WDI.DE',   # Wirecard
          ]

if 0:
    
    shares = [fidat(s, days=days, end=end, freq=freq) for s in symbols]
    
    closes = [getattr(s, key) for s in shares]
    
    df = pd.concat(closes, axis=1)
    
    df.columns = symbols

    df.to_csv('DAX.CSV')
    
df = pd.read_csv('DAX.CSV', index_col=0)

df0 = [d.median() for d in fidat.split(df, 'day')]

df1 = pd.concat(df0, axis=1).T

print(df1.corr())

