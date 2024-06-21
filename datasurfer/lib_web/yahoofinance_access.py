
import datetime
import requests
import pandas as pd
import numpy as np

from datasurfer.datainterface import DataInterface
from datasurfer.datautils import translate_config

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

#%%
class YahooFinanceAccess(DataInterface):
    """
    A class for accessing Yahoo Finance data.

    Parameters:
    - symbol (str): The symbol of the stock or asset.
    - freq (str): The frequency of the data (e.g., '1d' for daily, '1wk' for weekly).
    - days (int): The number of days of data to retrieve (optional).
    - start (datetime or tuple): The start date of the data (optional).
    - end (datetime or tuple): The end date of the data (optional).
    - config (dict): Configuration settings (optional).

    Attributes:
    - url (str): The URL for accessing the Yahoo Finance data.
    - name (str): The symbol of the stock or asset.
    - response (Response): The HTTP response object.
    - comment (dict): Additional information about the data.
    - data (dict): The raw data returned from the API.
    """

    def __init__(self, symbol, freq, days=None, start=None, end=None, config=None):
        """
        Initialize the YahooFinanceAccess object.

        Args:
        - symbol (str): The symbol of the stock or asset.
        - freq (str): The frequency of the data (e.g., '1d' for daily, '1wk' for weekly).
        - days (int): The number of days of data to retrieve (optional).
        - start (datetime or tuple): The start date of the data (optional).
        - end (datetime or tuple): The end date of the data (optional).
        - config (dict): Configuration settings (optional).
        """

        super().__init__(path=None, name=symbol, config=config)
                       
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
        try:
            freq = FREQ_STRS[np.abs(FREQ_ARR - freq).argmin()]
        except:
            if freq not in FREQ_STRS:
                raise ValueError(f'Frequency {freq} not supported, please use one of {FREQ_STRS}')
        
        self.url = URL_YAHOO.format(symbol=symbol, 
                                time_start=int(start.timestamp()),
                                time_end=int(end.timestamp()), 
                                frequency=freq)  

        
    @property
    def name(self):
        """
        Get the symbol of the stock or asset.

        Returns:
        - str: The symbol of the stock or asset.
        """
        return self.data['meta']['symbol']
        
    @property
    def response(self):
        """
        Get the HTTP response object.

        Returns:
        - Response: The HTTP response object.
        """
        if not hasattr(self, '_response'):
            self._response = requests.get(self.url, headers = {'User-agent': 'your bot 0.1'})
            assert self._response.status_code == 200, f'Request failed, status code: {self._response.status_code}'
            
        return self._response
    
    @property
    def comment(self):       
        """
        Get additional information about the data.

        Returns:
        - dict: Additional information about the data.
        """
        return self.response.json()['chart']['result'][0]['meta']
    

    @property
    def data(self):
        """
        Get the raw data returned from the API.

        Returns:
        - dict: The raw data returned from the API.
        """
        data = self.response.json()['chart']['result'][0]
                
        return data
    
    @translate_config()
    def get_df(self):
        """
        Get the data as a pandas DataFrame.

        Returns:
        - DataFrame: The data as a pandas DataFrame.
        """
        keys = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame({**self.data['indicators']['quote'][0], 
                           **dict(date=pd.to_datetime(self.data['timestamp'], unit='s'))}).dropna()[keys]
        df.index.name = self.data['meta']['symbol']
        
        return df
        
        
