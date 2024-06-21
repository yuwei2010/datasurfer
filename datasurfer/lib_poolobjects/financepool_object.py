import pandas as pd
import numpy as np
from datasurfer import DataPool
from datasurfer.lib_objects.pandas_object import FinanceObject
from functools import wraps
from datasurfer.datautils import show_pool_progress, bcolors
from datasurfer.lib_web.yahoofinance_access import YahooFinanceAccess

WEB_ACCESS = {'yahoo': YahooFinanceAccess}
#%%
def strategy_wrapper(func):
    """
    A decorator function that wraps a strategy function and returns a StockObject.

    Parameters:
    - func: The strategy function to be wrapped.

    Returns:
    - wrapper: The wrapped function that returns a StockObject.

    """
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
    """
    Represents a stock object.

    Attributes:
        - symbol: The symbol of the stock.
        - freq: The frequency of the stock data.
        - days: The number of days of data to retrieve.
        - start: The start date of the data.
        - end: The end date of the data.
        - config: The configuration for retrieving the data.
        - access: The web access method to use for retrieving the data.

    Methods:
        - from_web: Retrieves stock data from the web.
        - backtesting: Performs backtesting on the stock data.
        - date2index: Sets the date column as the index of the stock data.
        - portfolio: Retrieves the latest portfolio value.
        - plot_operation: Plots the buy and sell operations on a graph.
    """

    @classmethod
    def from_web(cls, symbol, freq, days=None, start=None, 
                   end=None, config=None, access='yahoo'):       
        """
        Retrieves stock data from the web.

        Args:
            - symbol: The symbol of the stock.
            - freq: The frequency of the stock data.
            - days: The number of days of data to retrieve.
            - start: The start date of the data.
            - end: The end date of the data.
            - config: The configuration for retrieving the data.
            - access: The web access method to use for retrieving the data.

        Returns:
            A new StockObject instance with the retrieved stock data.
        """
        web_engine = WEB_ACCESS[access]       
        obj = web_engine(symbol, freq, days=days, start=start, end=end, config=config)        
        new_obj = cls.from_other(obj)        
        return new_obj
        
    def backtesting(self, func, **kwargs):  
        """
        Performs backtesting on the stock data.

        Args:
            - func: The backtesting strategy function to apply.
            - kwargs: Additional keyword arguments to pass to the strategy function.

        Returns:
            The result of the backtesting strategy function.
        """
        func_ = strategy_wrapper(func)
        
        return func_(self, **kwargs)
    
    def date2index(self, name):
        """
        Sets the date column as the index of the stock data.

        Args:
            - name: The name of the date column.

        Returns:
            The StockObject instance with the date column set as the index.
        """
        self.df.sort_values(by=name, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df.set_index(name, inplace=True, drop=False)
        self.df.index.name = None
        
        return self
    
    def portfolio(self, by):
        """
        Retrieves the latest portfolio value.

        Args:
            - by: The column to use for calculating the portfolio value.

        Returns:
            The latest portfolio value.
        """
        return self.df[by].dropna().iloc[-1]
    
    def plot_operation(self, date='trade_date', base='close', share='shares', **kwargs):
        """
        Plots the buy and sell operations on a graph.

        Args:
            - date: The column to use as the x-axis for the graph.
            - base: The column to use as the y-axis for the graph.
            - share: The column to use for calculating the change in shares.
            - kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            The matplotlib Axes object containing the plotted graph.
        """
        chg_shares = self[share].diff()
        buys = self[base].copy()
        buys[chg_shares <= 0] = np.nan
        
        sells = self[base].copy()
        sells[chg_shares >= 0] = np.nan       

        ax = self.plot(**kwargs).line(base, x=date, setax=True, labels=['Close'], color='grey', lw=1)
        self.plot.line(buys.values, x=date, ls='None', marker='^', markersize=10, color='g', ax=ax, labels=['Buy'])
        self.plot.line(sells.values, x=date, ls='None', marker='v', markersize=10, color='r', ax=ax, labels=['Sell'])
        ax.legend(loc='best', ncols=3, title=self.name)
        
        ax.set_xlabel(date)
        ax.set_ylabel(base)
        
        return ax

#%%

class StockPool(DataPool):
    """
    A class representing a pool of stock objects.

    Parameters:
    - path (str): The path to the data.
    - config (dict): Configuration options for the pool (optional).
    - name (str): The name of the pool (optional).
    - comment (str): Additional comments about the pool (optional).
    - **kwargs: Additional keyword arguments.

    Attributes:
    - path (str): The path to the data.
    - config (dict): Configuration options for the pool.
    - name (str): The name of the pool.
    - comment (str): Additional comments about the pool.
    - objs (list): List of stock objects in the pool.

    Methods:
    - from_web: Create a stock pool by downloading data from the web.
    - portfolio: Calculate the portfolio for the stock pool.
    - backtesting: Perform backtesting on the stock pool.
    - mlp_backtesting: Perform backtesting using a multi-layer perceptron (MLP) on the stock pool.
    """

    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        super().__init__(path, interface=StockObject, config=config, name=name, comment=comment, **kwargs)

    @classmethod
    def from_web(cls, symbols, pause=5, **kwargs):
        """
        Create a stock pool by downloading data from the web.

        Parameters:
        - symbols (list): List of stock symbols to download.
        - pause (int): Pause time between each download (default: 5 seconds).
        - **kwargs: Additional keyword arguments.

        Returns:
        - StockPool: A stock pool object containing the downloaded data.
        """
        import time
        from tqdm import tqdm

        def get():
            pbar = tqdm(symbols)
            for symbol in pbar:
                msg = f'Download "{bcolors.OKGREEN}{bcolors.BOLD}{symbol}{bcolors.ENDC}"'
                pbar.set_description(f'{msg}')
                yield StockObject.from_web(symbol, **kwargs)
                time.sleep(pause)

        return cls(list(get()))

    def portfolio(self, by):
        """
        Calculate the portfolio for the stock pool.

        Parameters:
        - by (str): The method to calculate the portfolio.

        Returns:
        - pd.Series: A pandas Series object representing the portfolio.
        """
        return pd.Series(self.map(lambda x: x.portfolio(by)), name=self.name, index=self.names())

    def backtesting(self, func, pbar=True, **kwargs):
        """
        Perform backtesting on the stock pool.

        Parameters:
        - func (function): The backtesting function to apply.
        - pbar (bool): Whether to show a progress bar (default: True).
        - **kwargs: Additional keyword arguments.

        Returns:
        - StockPool: A stock pool object containing the backtesting results.
        """
        name = kwargs.get('name', None) or func.__name__
        comment = kwargs.pop('comment', None)

        @show_pool_progress(f'Backtesting {bcolors.OKGREEN}{bcolors.BOLD}{name}{bcolors.ENDC}', show=pbar)
        def get(self):
            for obj in self.objs:
                yield obj.backtesting(func, **kwargs)

        return StockPool(list(get(self)), name=name, comment=comment)

    def mlp_backtesting(self, func, **kwargs):
        """
        Perform backtesting using a multi-layer perceptron (MLP) on the stock pool.

        Parameters:
        - func (function): The backtesting function to apply.
        - **kwargs: Additional keyword arguments.

        Returns:
        - StockPool: A stock pool object containing the backtesting results.
        """
        comment = kwargs.pop('comment', None)
        name = kwargs.get('name', None) or func.__name__
        func_ = strategy_wrapper(func)
        fun = lambda obj: func_(obj, **kwargs)

        return StockPool(self.mlp.map(fun), name=name, comment=comment)
    

    
    

