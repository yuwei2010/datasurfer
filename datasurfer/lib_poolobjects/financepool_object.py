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
        
        inplace = kwargs.pop('inplace', False)
        
        if inplace:
            data = obj.df
        else:        
            data = obj.df.copy()
        
        out = func(data, **kwargs)
        
        assert isinstance(out, pd.DataFrame), 'Expected a pandas DataFrame as output'
        if inplace:
            return obj
        else:
            return StockObject(out, name=obj.name, comment=obj.comment, config=obj.config)
    
    return wrapper




#%%
class Backbloker(object):
    
    """
    Backtester class for backtesting trading strategies.
    https://algotrading101.com/learn/build-my-own-custom-backtester-python/
    https://github.com/IgorWounds/Backtester101/tree/main/backtester
    """

    def __init__(self, initial_capital: float = 1_000_000.0, commission_pct: float = 0, commission_fixed = 0, col_price='close', col_signal='signal'):
        """Initialize the backtester with initial capital and commission fees."""
        
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed 
        
        self.col_price = col_price
        self.col_signal = col_signal
        
        
    def __call__(self, strategy):    
            """
            Call method for the FinancePoolObject class.

            Args:
                strategy (callable): The strategy function to be executed.

            Returns:
                callable: A trade function that applies the strategy to the given data.

            """
            def trade(data, **kwargs):
                """
                Trade function that applies the strategy to the given data.

                Args:
                    data (object): The data to be processed.
                    **kwargs: Additional keyword arguments to be passed to the strategy function.

                Returns:
                    object: The processed data.

                """
                data = strategy(data, **kwargs)
                return self.trade(data)
            
            return trade
    
    def daily_trade(self, signal, price, cash, positions):
        
        if signal > 0 and cash > 0:
            shares_to_buy = cash * signal // ((1+self.commission_pct)*price)  
            commission = max(self.commission_fixed, shares_to_buy*price*self.commission_pct)  
            trade_value = shares_to_buy * price + commission
            positions += shares_to_buy
            cash -= trade_value
            
        elif signal < 0 and positions > 0:
            shares_to_sell = round(-signal * positions)
            trade_value = shares_to_sell * price
            commission = max(self.commission_fixed, trade_value*self.commission_pct)  
            cash += trade_value - commission
            positions -= int(shares_to_sell)
        
        else:
            commission = 0
            
        position_value = positions * price
        portfolio_value = position_value + cash
        
        return cash, positions, commission, portfolio_value
    
    def trade(self, data):
        
        cash = self.initial_capital
        positions = 0
        
        cash_history = []
        position_history = []
        commission_history = []
        portfolio_history = []
        
        for _, row in data.iterrows():      
            signal = row[self.col_signal]
            price = row[self.col_price]                    
            cash, positions, commission, portfolio_value = self.daily_trade(signal, price, cash, positions)
            cash_history.append(cash)
            position_history.append(positions)
            commission_history.append(commission)
            portfolio_history.append(portfolio_value)  
              
        data['position'] = position_history
        data['cash'] = cash_history
        data['commission'] = commission_history
        data['portfolio'] = portfolio_history
        data['daily_return'] = data['portfolio'].pct_change()
        data['total_daily_return'] = (1 + data['daily_return']).cumprod() 

        return data            
                      
    
    def get_performance(self, data:pd.DataFrame, risk_free_rate:float=0)->pd.Series:
        """
        Calculate the performance metrics of a portfolio.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the portfolio data.
        - risk_free_rate (float): Risk-free rate of return (default: 0).

        Returns:
        - out (pd.Series): Series containing the calculated performance metrics.
            - final_portfolio_value: The final value of the portfolio.
            - total_return: The total return of the portfolio.
            - annualized_return: The annualized return of the portfolio.
            - annualized_volatility: The annualized volatility of the portfolio.
            - sharpe_ratio: The Sharpe ratio of the portfolio.
            - sortino_ratio: The Sortino ratio of the portfolio.
            - CALMAR_ratio: The Calmar ratio of the portfolio.
            - maximal_drawdown: The maximal drawdown of the portfolio.
        """
        out = pd.Series()
        out['final_portfolio_value'] = data['portfolio'].iloc[-1]
        out['total_return'] = (out['final_portfolio_value'] - self.initial_capital) / self.initial_capital
        out['annualized_return'] = (1 + out['total_return']) ** (252 / len(data)) - 1
        out['annualized_volatility'] = data['daily_return'].std() * np.sqrt(252)
        out['sharpe_ratio'] = (out['annualized_return'] - risk_free_rate) / out['annualized_volatility'] if out['annualized_volatility'] != 0 else np.nan

        downside_volatility = data['daily_return'][data['daily_return'] < 0].std() * np.sqrt(252)
        out['sortino_ratio'] = (out['annualized_return'] - risk_free_rate) / downside_volatility if downside_volatility > 0 else np.nan

        out['CALMAR_ratio'] = out['annualized_return'] / abs(data['daily_return'].min())
        out['maximal_drawdown'] = (data['total_daily_return'].cummax() - data['total_daily_return']).max()

        return out
    
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
    
    def date2index(self, col_date='date', drop=False, offset=0, unit='s'):
        
        self.col2index(col_date, drop=drop)
        self.df.set_index(self.df.index + pd.Timedelta(offset, unit='s'), inplace=True)
        
        return self
        
    def backtesting(self, func, inplace=False, **kwargs):  
        """
        Performs backtesting on the stock data.

        Args:
            - func: The backtesting strategy function to apply.
            - kwargs: Additional keyword arguments to pass to the strategy function.

        Returns:
            The result of the backtesting strategy function.
        """
        func_ = strategy_wrapper(func)
        
        return func_(self, inplace=inplace, **kwargs)
    
    def get_performance(self, trader, **kwargs):
        """
        Calculates the performance of the finance pool object based on the given trader.

        Parameters:
        - trader: The trader object used to calculate the performance.
        - **kwargs: Additional keyword arguments to be passed to the trader's get_performance method.

        Returns:
        - out: The performance of the finance pool object as calculated by the trader.
        """
        out = trader.get_performance(self.df, **kwargs)
        out.name = self.name
        
        return out
    
    def plot_operation(self, col_date=None, col_price='close', col_position='position', **kwargs):
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
        
        date = self.df.index.to_list() if col_date is None else col_date
        chg_position = self[col_position].diff()
        buys = self[col_price].copy()
        buys[chg_position <= 0] = np.nan
        
        sells = self[col_price].copy()
        sells[chg_position >= 0] = np.nan       

        ax = self.plot(**kwargs).line(col_price, x=date, setax=True, labels=['Close'], color='grey', lw=1)
        self.plot.line(buys.values, x=date, ls='None', marker='^', markersize=10, color='g', ax=ax, labels=['Buy'])
        self.plot.line(sells.values, x=date, ls='None', marker='v', markersize=10, color='r', ax=ax, labels=['Sell'])
        ax.legend(loc='best', ncols=3, title=self.name)
        
        ax.set_xlabel('date')
        ax.set_ylabel(col_price)
        
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
        """
        Initialize a FinancePoolObject.

        Args:
            path (str): The path to the object.
            config (dict, optional): Configuration options for the object. Defaults to None.
            name (str, optional): The name of the object. Defaults to None.
            comment (str, optional): A comment for the object. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(path, interface=StockObject, config=config, name=name, comment=comment, keep_df_index=True, **kwargs)

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

    def date2index(self, col_date='date', pbar=True, **kwargs):
        self.map(lambda x: x.date2index(col_date, **kwargs), pbar=pbar) 
        return self        
        
    

    def backtesting(self, func, pbar=True, inplace=False, **kwargs):
        """
        Perform backtesting on the stock pool.

        Parameters:
        - func (function): The backtesting function to apply.
        - pbar (bool): Whether to show a progress bar (default: True).
        - **kwargs: Additional keyword arguments.

        Returns:
        - StockPool: A stock pool object containing the backtesting results.
        """
        name = kwargs.pop('name', None) or func.__name__
        comment = kwargs.pop('comment', None)
        
        objs = self.map(lambda obj: obj.backtesting(func, inplace=inplace, **kwargs), pbar=pbar, 
                        description=f'Backtesting "{bcolors.OKGREEN}{bcolors.BOLD}{name}{bcolors.ENDC}"')

        if inplace:
            return self
        else:
            return StockPool(objs, name=name, comment=comment)

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
        
    def get_performance(self, trader, **kwargs):
        """
        Retrieves the performance data for each object in the finance pool.

        Parameters:
        - trader: The trader object used to calculate performance.
        - **kwargs: Additional keyword arguments to be passed to the `get_performance` method of each object.

        Returns:
        - performance: A pandas DataFrame containing the performance data for each object in the finance pool.
        """
        return pd.concat(self.map(lambda x: x.get_performance(trader, **kwargs)), axis=1)

    
    

