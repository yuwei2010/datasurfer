#%% Import Libraries
import re
import pandas as pd
import numpy as np
import warnings
import asammdf
from asammdf import set_global_option    
set_global_option("raise_on_multiple_occurrences", False) 
from xml.etree.ElementTree import fromstring
from datasurfer.lib_objects import DataInterface
from datasurfer.datautils import translate_config, extract_channels

#%% ASAMmdfObject
class ASAMmdfObject(DataInterface):
    """
    Represents an ASAM MDF object.

    Args:
        path (str): The path to the MDF file.
        config (dict, optional): The configuration dictionary. Defaults to None.
        sampling (float, optional): The sampling rate. Defaults to 0.1.
        name (str, optional): The name of the object. Defaults to None.
        comment (str, optional): The comment for the object. Defaults to None.
        autoclose (bool, optional): Whether to automatically close the object. Defaults to False.
    """
  
    exts = ['.mf4']
    
    def __init__(self, path, config=None, sampling=0.1, name=None, 
                 comment=None, autoclose=False):
        """
        Initializes a new instance of the ASAMmdfObject class.

        Args:
            path (str): The path to the MDF file.
            config (dict, optional): The configuration dictionary. Defaults to None.
            sampling (float, optional): The sampling rate. Defaults to 0.1.
            name (str, optional): The name of the object. Defaults to None.
            comment (str, optional): The comment for the object. Defaults to None.
            autoclose (bool, optional): Whether to automatically close the object. Defaults to False.
        """
        super().__init__(path, config, comment=comment, name=name)
        
        self.sampling = sampling
        self.autoclose = autoclose

    def __enter__(self):
        """
        Enters the context manager.

        Returns:
            ASAMmdfObject: The current instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exits the context manager.

        Args:
            exc_type (type): The type of the exception.
            exc_value (Exception): The exception instance.
            exc_traceback (traceback): The traceback information.

        Returns:
            bool: True if the exception was handled, False otherwise.
        """
        self.close()
        
        if exc_type:
            return False
        
        return True   

    def __len__(self):
        """
        Returns the length of the object.

        Returns:
            int: The length of the object.
        """
        return self.t.size

    @property
    def info(self):
        """
        Gets the information dictionary of the object.

        Returns:
            dict: The information dictionary.
        """
        if not hasattr(self, '_info'):
            try:
                self._info = self.fhandler.info()
            except AttributeError:
                self._info = None
                
        return self._info
        
    @property
    def comment(self):
        """
        Gets the comment of the object.

        Returns:
            dict: The comment dictionary.
        """
        if self._comment is None :
            info = self.info
            if info and 'comment' in info:
                xmlobj= fromstring(info['comment'])             
                comment = dict()
                try:
                    tx, = xmlobj.findall('TX')
                    if tx.text:
                        comment = dict(re.findall(r'(.+):\s+(.+)', tx.text))
                    if not comment:
                        comment = {'Comment': tx.text}
                except ValueError:
                    comment = dict()
                try:
                    cp, = xmlobj.findall('common_properties')
                    for item in cp:
                        name = item.attrib['name']
                        value = item.text
                        if value is not None:
                            comment[name] = value
                except ValueError:
                    if not comment:
                        comment = None
                self._comment = comment
        return self._comment
    
    @property
    def t(self):
        """
        Gets the time axis of the object.

        Returns:
            numpy.ndarray: The time axis.
        """
        if not len(self.df) and self.config is None:
            warnings.warn('Time axis may have deviation due to missing configuration.')
            n = 0
            while 1:
                df = self.get_channels(self.channels[n])
                if len(df):
                    break
                n = n + 1
            t = df.index
        else:
            t = self.df.index
        return np.asarray(t)
    
    @property
    def df(self):
        """
        Gets the DataFrame of the object.

        Returns:
            pandas.DataFrame: The DataFrame.
        """
        if not hasattr(self, '_df'):
            if self.config is None:
                self._df = pd.DataFrame()
            else:
                self._df = self.get_df()
        return self._df
    
    @property
    def fhandler(self):
        """
        Gets the MDF file handler.

        Returns:
            asammdf.MDF: The MDF file handler.
        """
        if not hasattr(self, '_fhandler'):
            self._fhandler = asammdf.MDF(self.path)
        return self._fhandler
        
    @property
    def channels(self):
        """
        Gets the list of channels in the object.

        Returns:
            list: The list of channels.
        """
        out = [chn.name for group in self.fhandler.groups for chn in group['channels']]
        return sorted(set(out)) 

    @translate_config()
    @extract_channels()
    def get_channels(self, *channels):
        """
        Gets the channels from the object.

        Args:
            *channels (str): The names of the channels.

        Returns:
            pandas.DataFrame: The DataFrame containing the channels.
        """
        df = self.fhandler.to_dataframe(channels=channels, 
                                    raster=self.sampling,
                                    time_from_zero=True)
        df.index.name = 'time'
        return df 
    
    def keys(self):
        """
        Gets the keys of the object.

        Returns:
            list: The keys of the object.
        """
        if not len(self.df):
            res = self.channels
        else:
            res = list(self.df.keys())
        return res   
    
    def get_df(self, close=None):
        """
        Gets the DataFrame of the object.

        Args:
            close (bool, optional): Whether to close the object after getting the DataFrame. Defaults to None.

        Returns:
            pandas.DataFrame: The DataFrame.
        """
        close = self.autoclose if close is None else close
        if self.config is None:
            df = pd.DataFrame()
        else:
            df = self.get_channels(*self.config.keys())
        if close:   
            self.comment
            self.close()
        return df

    def get(self, *names):
        """
        Gets the data from the object.

        Args:
            *names (str): The names of the data.

        Returns:
            object: The data.
        """
        if all(na in self.df.keys() for na in names):
            res = super().get(*names)
        elif len(names) == 1 and (names[0].lower() == 't' 
                                  or names[0].lower() == 'time' 
                                  or names[0].lower() == 'index'):
            if names[0] in self.df.keys():
                res = self.df[names]
            else:
                res = pd.DataFrame(self.t, index=self.t, columns= ['time'])
        else:
            res = self.get_channels(*names)
        return res
    
    def search_channel(self, patt):
        """
        Searches for channels matching a pattern.

        Args:
            patt (str): The pattern to search for.

        Returns:
            list: The list of matching channels.
        """
        r = re.compile(patt)
        return list(filter(r.match, self.channels))