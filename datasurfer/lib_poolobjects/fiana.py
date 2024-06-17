# -*- coding: utf-8 -*-

import json
import datetime
import requests
import pandas as pd
import numpy as np


#%%---------------------------------------------------------------------------#
URL_YAHOO = (
            'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?symbol={symbol}'
            '&period1={time_start}&period2={time_end}&interval={frequency}&'
            'includePrePost=true&events=div%7Csplit%7Cearn&lang=en-US&'
            'region=US&crumb=t5QZMhgytYZ&corsDomain=finance.yahoo.com'
            )

FREQ_STRS = ['1m', '2m', '5m', '15m', '30m', '60m', 
             '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

FREQ_ARR = np.array([1, 2, 5, 15, 30, 60, 90, 60, 60*24, 60*24*5, 60*24*7, 
            60*24*7*30, 60*24*7*30*3], dtype=int)



#%%---------------------------------------------------------------------------#
class YahooFinanceError(Exception):

    def __init__(self, message):
        
        self.message = message

#%%---------------------------------------------------------------------------#
class Indicator(pd.Series):
    
    def __init__(self, *arg, **kwargs):
        
        super().__init__(*arg, **kwargs)
        
        if not isinstance(self.index, pd.DatetimeIndex):
            
            self.index = pd.to_datetime(self.index, unit='s')

    #%%-----------------------------------------------------------------------# 

    def ceil(self, freq):
        
        return Indicator(self.values, index=self.index.ceil(freq), name=self.name)
    
    #%%-----------------------------------------------------------------------#     
    def split(self, freq):
        def get():
            
            for d in ds.unique():

                yield self[ds==d]

        ds = self.ceil(freq).index

        
        return list(get())
    
    #%%-----------------------------------------------------------------------#     
    def split_apply(self, freq, method='median'):
        
        def get():
            for s in self.split(freq):

                idx, = s.index.floor(freq).unique()
                
                if isinstance(method, str):
                
                    yield idx, getattr(s, method)()
                
                else:
                    
                    yield idx,  method(s)
                
        
        idx, values = zip(*get())
        
        return Indicator(values, index=idx, name=self.name)
    
    #%%-----------------------------------------------------------------------#     
    def normalize(self):
        
        return Indicator((self-self.min()) / (self.max()-self.min()), name=self.name)

        
#%%---------------------------------------------------------------------------#
class FiAna(object):
    
    def __init__(self, **kwargs):
        
        js = kwargs.pop('js', None)
        
        if js is None:
            
            symbol = kwargs.pop('symbol')
            freq = kwargs.pop('freq')
            start = kwargs.pop('start', None)
            end = kwargs.pop('end', None)
            days = kwargs.pop('days', None)
            
            
            if start is None:
                
                end = datetime.datetime.now() if end is None else end
                
                end = (end if isinstance(end, datetime.datetime) 
                        else datetime.datetime(*end))
                
                start = end - datetime.timedelta(days=days)
                
                
            else:
 
                start = (start if isinstance(start, datetime.datetime) 
                        else datetime.datetime(*start))
                
                if days is None:
                    
                    end = datetime.datetime.now() if end is None else end
                
                else:
                    
                    end = start + datetime.timedelta(days=days)
            
            freq = FREQ_STRS[np.abs(FREQ_ARR - freq).argmin()]
            

            url = URL_YAHOO.format(symbol=symbol, 
                                   time_start=int(start.timestamp()),
                                   time_end=int(end.timestamp()), 
                                   frequency=freq)  
            
            print(url)
            
            js = requests.get(url).json()['chart']

        else:

            with open(js, 'r') as jsonfile:
            
                js = json.load(jsonfile)
                
        
        assert not kwargs, 'Unexpected arguments "{}".'.format(', '.join(kwargs.keys()))
        self.json = js
        
        if self.json['error'] is not None:
            
            code = self.json['error']['code']
            description = self.json['error']['description']
            
            raise YahooFinanceError('{}: {}'.format(code, description))     
            
    #%%-----------------------------------------------------------------------# 
    @property
    def result(self):

        return self.json['result'][0]  

    #%%-----------------------------------------------------------------------# 
    @property
    def meta(self):

        return self.json['result'][0]['meta']
    
    #%%-----------------------------------------------------------------------#        
    @property
    def symbol(self):
        
        return self.result['meta']['symbol']

    #%%-----------------------------------------------------------------------#
    @property
    def open(self):

        return Indicator(pd.Series(self.result['indicators']['quote'][0]['open'], 
                         index=pd.to_datetime(self.result['timestamp'], unit='s'),).dropna(),
                         dtype=float, name=self.symbol)    
        
    #%%-----------------------------------------------------------------------#
    @property
    def close(self):

        return Indicator(pd.Series(self.result['indicators']['quote'][0]['close'], 
                        index=pd.to_datetime(self.result['timestamp'], unit='s'),).dropna(),
                         dtype=float, name=self.symbol)      

    #%%-----------------------------------------------------------------------#
    @property
    def high(self):

        return Indicator(pd.Series(self.result['indicators']['quote'][0]['high'], 
                        index=pd.to_datetime(self.result['timestamp'], unit='s'),).dropna(),
                         dtype=float, name=self.symbol)
    #%%-----------------------------------------------------------------------#
    @property
    def low(self):

        return Indicator(pd.Series(self.result['indicators']['quote'][0]['low'], 
                        index=pd.to_datetime(self.result['timestamp'], unit='s'),).dropna(),
                         dtype=float, name=self.symbol)
    #%%-----------------------------------------------------------------------#
    @property
    def volume(self):

        return Indicator(pd.Series(self.result['indicators']['quote'][0]['volume'], 
                         index=pd.to_datetime(self.result['timestamp'], unit='s'),).dropna(),
                         dtype=int, name=self.symbol)
                
    #%%-----------------------------------------------------------------------#
    def dataframe(self, *keys, transform=None, title=None):
        
        if not keys:
            keys = ['open', 'close', 'high', 'low', 'volume']
            
        if transform is None:
            transform = lambda x:x
            
        title = title or self.symbol
            
        columns = pd.MultiIndex.from_product([[title], keys],
                                    names=['DATA', 'DATE'])
    
        
        df = pd.concat([transform(getattr(self, k)) for k in keys], axis=1)
        
        df.columns = columns
    
        return df
    
    #%%-----------------------------------------------------------------------# 
    def save(self, file):
        
        with open(file, 'w') as outfile:
            
            json.dump(self.json, outfile)
            
        return self

#%%---------------------------------------------------------------------------#
class Corrin(object):
    
    def __init__(self, share, df, rng, drop_exist=True, normalize=True):
                
        if drop_exist and share.name in df:
            
            df.drop(share.name, inplace=True, axis=1)

                        
        self.share = share
        self.df = df     
        self.dict_corr = dict((c, self.calc_count(c)) for c in rng)
        
    #%%-----------------------------------------------------------------------#  
    def calc_count(self, count):
        
        def get():
            
            for col in self.df:
                
                try:
                    ref = self.df[col].dropna()
                    
                    ref = (ref-ref.min()) / (ref.max()-ref.min())
                    
                    index = range(ref.size-count-1)
                    
    
                    
                    yield pd.Series([np.corrcoef(c, ref[n:n+count])[0, 1] for 
                                                 n in index], name=col, index=index)
                except ValueError:
                    
                    continue
                
        
        c = Indicator(self.share[-count:]).normalize()
        
        return pd.concat(get(), axis=1)
    

            
    #%%-----------------------------------------------------------------------#  
    def calc_minmax(self, which='max'):
        
        def get():
            
            for c, df in self.dict_corr.items():
                
                yield pd.Series(getattr(df, which)(), name=c)
                


        
        return pd.concat(get(), axis=1)
    
    #%%-----------------------------------------------------------------------# 
    def select_count(self, top=3, bottom=3, level=None):
        

            
        df_max = self.calc_minmax(which="max")
        df_min = self.calc_minmax(which="min")
            
        ntop = pd.concat([df_max[s].nlargest(top) for s in df_max], axis=1)
        
        nflop = pd.concat([df_min[s].nsmallest(bottom) for s in df_min], axis=1)
        
        merged = pd.concat([ntop, nflop], axis=0)
        if level:
            merged = merged[merged.abs() >= level].dropna(how='all')
        
        return merged[merged.abs().sum().idxmax()].dropna()
    
    #%%-----------------------------------------------------------------------# 
    def get_data(self, *args, **kwargs):
        
        def get():
            
            for k, v in index.items():
                
                s = self.df[k].dropna()[v:]
                
                s.index = range(len(s))
                
                yield s
        
        ds = self.select_count(*args, **kwargs)

        corrs = self.dict_corr[ds.name][ds.index]
        
        share = pd.Series(self.share[-ds.name:].values, 
                          index=range(ds.name), name=self.share.name)
        

        index = dict((k, corrs.index[v][0]) for k, v in 
                     corrs[corrs==ds].notnull().items())
        
        return pd.concat(get(), axis=1), share
            
            

        

        
        
                
#%%---------------------------------------------------------------------------#

if __name__ == "__main__":
    
    share = FiAna(symbol='TSMC34.SA', days=3, freq=1).save('IFX.DE')
    
    # print(share.url)
    
    print(share.dataframe())
    
    print(datetime.datetime.now())



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    