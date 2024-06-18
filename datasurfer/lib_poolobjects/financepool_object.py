import pandas as pd
import numpy as np
from datasurfer import DataPool
from datasurfer.lib_objects.pandas_object import FinanceObject
from datasurfer.lib_objects.parquet_object import ParquetObject
from functools import wraps
from datasurfer.datautils import show_pool_progress, bcolors

#%%
def strategy_wrapper(func):
    
    @wraps(func)
    def wrapper(obj, **kwargs):
        
        name = kwargs.pop('name', None) or func.__name__
        
        data = obj.df.copy()
        
        out = func(data, **kwargs)
        
        assert isinstance(out, pd.DataFrame), 'Expected a pandas DataFrame as output'
        
        return StockObject(out, name=obj.name, comment={**{'name': name}, **kwargs})
    
    return wrapper

#%%

class StockObject(FinanceObject):
        
    def backtesting(self, func, **kwargs):  
                   
        func_ = strategy_wrapper(func)
        
        return func_(self, **kwargs)
    
    def date2index(self, name):
        
        self.df.sort_values(by=name, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df.set_index(name, inplace=True, drop=False)
        self.df.index.name = None
        
        return self
    
    def portfolio(self, by):
        
        return self.df[by].dropna().iloc[-1]
    
    def plot_operation(self, date='trade_date', base='close', share='shares'):
        
        chg_shares = self[share].diff()
        buys = self[base].copy()
        buys[chg_shares <= 0] = np.nan
        
        sells = self[base].copy()
        sells[chg_shares >= 0] = np.nan       

        ax = self.plot.line(base, x=date, setax=True, labels=['Close'], color='grey', lw=1)
        self.plot.line(buys.values, x=date, ls='None', marker='^', markersize=10, color='g', ax=ax, labels=['Buy'])
        self.plot.line(sells.values, x=date, ls='None', marker='v', markersize=10, color='r', ax=ax, labels=['Sell'])
        ax.legend(loc='best', ncols=3)
        
        ax.set_xlabel(date)
        ax.set_ylabel(base)
        
        return ax

#%%

class StockPool(DataPool):
    
    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        
        super().__init__(path, interface=StockObject, config=config, name=name, comment=comment)
        
    def portfolio(self, by):
        
        return pd.Series(self.map(lambda x: x.portfolio(by)), name=self.name, index=self.names())
        
        
    def backtesting(self, func, pbar=True, **kwargs):
        
        name = kwargs.get('name', None) or func.__name__
        
        comment = kwargs.pop('comment', None) 
                              
        @show_pool_progress(f'Backtesting {bcolors.OKGREEN}{bcolors.BOLD}{name}{bcolors.ENDC}', show=pbar)
        def get(self):           
            for obj in self.objs:                                           
                yield obj.backtesting(func, **kwargs)
        
        return StockPool(list(get(self)), name=name, comment=comment)
    
    
    def mlp_backtesting(self, func, **kwargs):
        comment = kwargs.pop('comment', None) 
        name = kwargs.get('name', None) or func.__name__
        func_ = strategy_wrapper(func)
        fun = lambda obj: func_(obj, **kwargs)  
        
        return StockPool(self.mlp.map(fun), name=name, comment=comment)
    

    
    

