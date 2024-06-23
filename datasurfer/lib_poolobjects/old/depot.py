# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import functools

from datetime import datetime

#%%---------------------------------------------------------------------------#


POSTBANK = (
            np.array([1200, 2600, 5200, 12500, 25000]),
            np.array([9.95, 17.95, 29.95, 39.95, 54.95, 69.95])
            ) 


COLUMNS = ['VOLUME', 'PRICE', 'NET', 'FEE', 'TOTAL']

BROKER = POSTBANK 

#%%---------------------------------------------------------------------------#
def get_fee(volume, broker):
    
    levels, charges = broker
        
    return charges[np.searchsorted(levels, volume)] 

#%%---------------------------------------------------------------------------#
def load_depot(name, *args, **kwargs):
    
    default = dict(header=0, index_col=[0, 1], parse_dates=True)
    
    default.update(kwargs)
    return pd.read_csv(name, *args, **default)

#%%---------------------------------------------------------------------------#
class Depot(pd.DataFrame):
    
    def __init__(self, *args, **kwargs):
        
        if args and isinstance(args[0], str):
            
            name, *args = args
        
            super().__init__(load_depot(name, *args, **kwargs))
        
        else:
            super().__init__(*args, **kwargs)
    #%%-----------------------------------------------------------------------#
    @property
    def shares(self):
        names, _ = self.index.levels
        
        return list(names)
    #%%-----------------------------------------------------------------------#
    def get_noempty(self):
        
        df = pd.concat([self.get_share(s) for s in self.shares if self.get_share(s)['VOLUME'].sum()>0])
        return Depot(df)
    #%%-----------------------------------------------------------------------#
    def get_empty(self):
        
        df = pd.concat([self.get_share(s) for s in self.shares if self.get_share(s)['VOLUME'].sum()==0])
        return Depot(df)
         
    #%%-----------------------------------------------------------------------#
    def trade(kind):
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, symbol, price, volume=None, date=None, broker=BROKER, sort=True):
                if date is None:
                    
                    date = datetime.now()
                
                else:
                    date = (date if isinstance(date, datetime) 
                                    else datetime(*date))
                    
                index = pd.MultiIndex.from_tuples([(symbol, date)], names=['SYMBOL', 'DATE TIME'])
                
                df = pd.DataFrame(np.atleast_2d(np.zeros(5)), columns=COLUMNS, index=index)
                
                if kind == 'SELL': 
                    if volume is None:
                        
                        volume = self.loc[symbol]['VOLUME'].sum()
                        
                    volume = -1 * abs(volume)
                
                else:
                    
                    volume = abs(volume)
                    
                cost = -1 * price * volume
                
                fee = -1 * get_fee(abs(cost), broker=broker)
                                
                df['PRICE'] = abs(price)
                df['VOLUME'] = int(volume)
                df['NET'] = cost
                df['FEE'] = fee
                
                df['TOTAL'] = cost + fee if kind == 'BUY' else cost + fee

        
                self.__init__(self.append(df))
                
                if sort:
                    return self.sorted()
                else:
                    return self     
            return wrapper
        return decorator
    #%%-----------------------------------------------------------------------#
    def load(self, name, **kwargs):
        
        default = dict(header=0, index_col=[0, 1], parse_dates=True)
        
        default.update(kwargs)
        self.__init__(pd.read_csv(name, **kwargs))
        
        return self
    #%%-----------------------------------------------------------------------#
    @trade('BUY')
    def buy(self, symbol, price, volume, date=None, broker=None): pass
    #%%-----------------------------------------------------------------------#
    @trade('SELL')
    def sell(self, symbol, price, volume, date=None, broker=None): pass     
    #%%-----------------------------------------------------------------------#
    def sorted(self, by='SYMBOL', **kwargs):
        
        self.__init__(self.sort_values(by=by, **kwargs))
        
        return self
    #%%-----------------------------------------------------------------------#
    def get_share(self, symbol, drop_level=False):
        
        return self.xs(symbol, drop_level=drop_level)
    #%%-----------------------------------------------------------------------#
    def summarize(self): 

        def fun(se):
            
            dates = se.index.get_level_values(1)
            days = (dates.max() - dates.min()).days
            
            
            df = pd.Series()
            df['PAY'] = se['NET'].where(se['VOLUME'] >0).sum() + se['FEE'].sum()
            df['RETURN'] = se['NET'].where(se['VOLUME'] <0).sum()
            for key in ['VOLUME', 'NET', 'FEE', 'TOTAL']:
                
                df[key] = se[key].sum()
            
            df['PROFIT'] = -1 * df['TOTAL'] / df['PAY'] * 100
            df['DAYS'] = days if days > 0 and df['VOLUME']==0  else np.nan
            df['PPD'] = -df['TOTAL'] / days / df['PAY'] if days > 0 and df['VOLUME']==0  else np.nan
            return df
        
        
        
        grp =  self.groupby(level=0).apply(fun)
        
        return grp
    #%%-----------------------------------------------------------------------#
    def copy(self):
        
        return Depot(self)
    
#%%---------------------------------------------------------------------------#
        
if __name__ == '__main__':
    
    pd.set_option('display.max_columns', 500)
    

    dp = Depot()
    
    dp.buy('DBK.DE', 6.32, 180, (2019, 10, 8)) 
    dp.buy('IFX.DE', 16.27, 70, (2019, 10, 8)) 
    dp.buy('BOSS.DE', 37.7, 31, (2019, 10, 15)) 
    
    dp.sell('DBK.DE', 7.1, 180, (2019, 10, 16))
    dp.buy('SDF.DE', 12.87, 90, (2019, 10, 16)) 
    dp.sell('IFX.DE', 16.65, 70, (2019, 10, 16))
    
    dp.buy('LHA.DE', 15.28, 75, (2019, 10, 17))
    dp.buy('CBK.DE', 5.35, 220, (2019, 10, 17))
    
    dp.buy('NDX1.DE', 11.84, 85, (2019, 10, 21))
    
    dp.buy('SGL.DE', 4.82, 210, (2019, 10, 21))
    dp.sell('LHA.DE', 15.82, 75, (2019, 10, 24))
    dp.sell('CBK.DE', 5.52, 220, (2019, 10, 24))
    
    dp.sell('SGL.DE', 4.35, 210, (2019, 10, 25))
    dp.buy('IFX.DE', 17.6, 60, (2019, 10, 25))
    dp.sell('SDF.DE', 12.74, 90, (2019, 10, 25))
    dp.sell('NDX1.DE', 11.58, 85, (2019, 10, 25))
    
    
    dp.buy('EVT.DE', 19.92, 50, (2019, 10, 25))
    dp.buy('HLE.DE', 45.54, 23, (2019, 10, 25))
    dp.buy('FRE.DE', 44.08, 23, (2019, 10, 25))
    
    print(dp.get_empty().summarize())
    
    dp.to_csv('DEPOT.DE')
    
