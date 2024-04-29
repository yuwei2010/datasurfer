import os
import re
import pandas as pd
import numpy as np
import warnings
import traceback

from abc import ABC, abstractmethod
from pathlib import Path
from difflib import SequenceMatcher
from itertools import chain, zip_longest
from datasurfer.datautils import parse_config, translate_config, extract_channels

#%% Data_Interface
def list_interfaces():
    """
    Returns a DataFrame containing information about the DataInterface classes in the datasurfer.lib_objects module.
    
    The DataFrame has the following columns:
    - name: The name of the DataInterface class.
    - class: The DataInterface class object.
    - doc: The docstring of the DataInterface class.
    
    Returns:
    DataFrame: A DataFrame containing information about the DataInterface classes.
    """
    import importlib
    import inspect
    
    out = dict()
    
    for pth in (Path(__file__).parent/'lib_objects').glob('*.py'):
        
        mdlname = pth.stem        
        try:
            mdl = importlib.import_module(f'datasurfer.lib_objects.{mdlname}')
        except ImportError:
            warnings.warn(f'Cannot import module "{mdlname}".')
            continue
        
        for item in dir(mdl):
            cls = getattr(mdl, item)

            if isinstance(cls, type):               
                names = [c.__name__ for c in inspect.getmro(cls)]
                if 'DataInterface' in names and cls.__name__ != 'DataInterface':
                    if hasattr(cls, 'exts'):
                        exts = sorted(cls.exts)
                    else:
                        exts = []
                    out[cls.__name__] = (cls, cls.__doc__, exts)
                    
    df = pd.DataFrame(out, index=['class', 'doc', 'file extension']).T
    df.index.name = 'name'
    
    return df


#%%
class DataInterface(ABC):
    """
    A class representing parent class of all data interfaces.

    Parameters:
    - path (str or Path): The path to the data file.
    - config (str, Path, list, tuple, set, or dict): The configuration for the data object.
    - name (str): The name of the data object.
    - comment (str): Additional comment or description for the data object.

    Attributes:
    - path (Path): The absolute path to the data file.
    - config (dict): The configuration for the data object.
    - name (str): The name of the data object.
    - comment (str): Additional comment or description for the data object.
    - df (DataFrame): The data stored in a pandas DataFrame.

    Methods:
    - __enter__(): Enter method for context management.
    - __exit__(exc_type, exc_value, exc_traceback): Exit method for context management.
    - __repr__(): Returns a string representation of the data object.
    - __str__(): Returns a string representation of the data object.
    - __len__(): Returns the number of rows in the data object.
    - __getitem__(name): Returns a column or subset of columns from the data object.
    - __setitem__(name, value): Sets the value of a column in the data object.
    - index(): Returns the index of the data object.
    - meta_info(): Returns metadata information about the data object.
    - size(): Returns the size of the data file in bytes.
    - comment(): Returns the comment or description of the data object.
    - comment(value): Sets the comment or description of the data object.
    - df(): Returns the data stored in a pandas DataFrame.
    - name(): Returns the name of the data object.
    - name(value): Sets the name of the data object.
    - initialize(): Initializes the data object by loading the data into a DataFrame.
    - keys(): Returns a list of column names in the data object.
    - describe(): Returns descriptive statistics of the data object.
    - count(): Returns the number of non-null values in each column of the data object.
    - get(*names): Returns a subset of columns from the data object.
    - search(patt, ignore_case=True, raise_error=False): Searches for columns that match a pattern.
    - memory_usage(): Returns the memory usage of the data object.
    - clean_config(): Cleans the configuration by removing keys that do not exist in the data object.
    - search_similar(name): Searches for column names that are similar to the given name.
    - drop(*names, nonexist_ok=True): Drops columns from the data object.
    - search_get(patt, ignore_case=False, raise_error=False): Returns a subset of columns that match a pattern.
    - load(*keys, mapping=None): Loads additional columns into the data object.
    - reload(): Reloads the data object by clearing the DataFrame cache.
    - merge(obj0): Merges another data object into the current data object.
    - squeeze(*keys): Removes columns from the data object, keeping only the specified columns.
    - pipe(*funs): Applies a series of functions to the data object.
    - rename(**kwargs): Renames columns in the data object.
    - resample(new_index=None): Resamples the data object to a new index.
    - to_numpy(): Returns the data as a numpy array.
    - to_dict(): Returns the data as a dictionary.
    - to_csv(name=None, overwrite=True): Saves the data as a CSV file.
    - to_excel(name=None, overwrite=True): Saves the data as an Excel file.
    - save(name, overwrite=True): Saves the data object to a file.
    - close(clean=True): Closes the data object and cleans up resources.
    """
   
   
    def __init__(self, path, config=None, name=None, comment=None):
        """
        Initialize a DataObject instance.

        Args:
            path (str or Path): The path to the data object.
            config (str, Path, list, tuple, set, dict, optional): The configuration for the data object.
                If a string or Path is provided, it is assumed to be a path to a JSON file and will be loaded as a dictionary.
                If a list, tuple, or set of strings is provided, it will be converted into a dictionary with each string as both the key and value.
                If a list of dictionaries is provided, the dictionaries will be combined into a single dictionary.
                If a dictionary is provided, it will be used as is.
                Defaults to None.
            name (str, optional): The name of the data object. Defaults to None.
            comment (str, optional): A comment or description for the data object. Defaults to None.
        """
          
        if path is not None:
            path = Path(path).absolute() 
            
        self._name = name
        self.path = path
        self._config = parse_config(config)
        self._comment = comment
        
    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        
        self.close()
        
        if exc_type:
            
            return False
        
        return True
    
    def __eq__(self, other):
        
        return self.__hash__() == other.__hash__()
        
    def __hash__(self):
        
        if hasattr(self, '_df'):
            h = pd.util.hash_pandas_object(self.df, index=True).sum()
            return hash((self.path, self.name, h, self.__class__))
        else:
            return hash((self.path, self.name, self.__class__))
    
    def __repr__(self):
        
        return f'<{self.__class__.__name__}@{self.name}>'   

    def __str__(self):
        
        return self.__repr__()
    
    def __len__(self):
        
        return len(self.df)
        
    def __getitem__(self, name):
        
        if isinstance(name, str) and '*' in name:

            res = self.search(name)
        
        else:  
            
            if isinstance(name, str):
                res = self.df[name]
            else:
                res = self.get(*name)
       
        return res
    
    def __setitem__(self, name, value):
        
        self.df[name] = value
        
    def __contains__(self, name):
        
        return name in self.df.columns  
    
    def __iter__(self):
        
        return self.df.items()
    
    def __rshift__(self, cls):
                
        return self.to_object(cls)
    
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, val):                
        self._config = parse_config(val)  
    
    def apply(self, name, value):
        """
        Applies the given value to the specified column name in the DataFrame.

        Args:
            name (str): The name of the column to apply the value to.
            value: The value to apply to the column.

        Returns:
            self: The updated DataInterface object.

        """
        self.df[name] = value
        return self
    def set_index(self, colname, name=None, drop=False):
        """
        Apply the values of a specified column as the new index of the DataFrame.

        Parameters:
        colname (str): The name of the column to be used as the new index.
        name (str, optional): The name to be assigned to the new index.

        Returns:
        self: The modified DataInterface object with the new index applied.
        """
        if drop: 
            self.df.set_index(colname, inplace=True)
        else:     
            self.df.index = self.df[colname]
        if name:
            self.df.name = name

        return self
    
    def reset_index(self, drop=True):
        """
        Resets the index of the DataFrame to the default integer index.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            self: The modified DataInterface object with the index reset.
        """
        self.df.reset_index(inplace=True, drop=drop)
        return self

    @property
    def index(self):
        """
        Returns the index of the DataFrame as a NumPy array.

        Returns:
            numpy.ndarray: The index of the DataFrame.
        """
        return self.df.index
    
    @property
    def meta_info(self):
        """
        Returns a dictionary containing metadata information about the data interface object.

        Returns:
            dict: A dictionary containing the following metadata information:
                - 'path': The path of the data interface object.
                - 'name': The name of the data interface object.
                - 'size': The size of the data interface object.
                - 'shape' (optional): The shape of the data interface object if it has a DataFrame associated with it.
                - Additional metadata information from the 'comment' attribute, if available.
        """
        out = dict()
        out['path'] = str(self.path)
        out['name'] = self.name
        out['size'] = self.size

        try:
            out.update(self.comment)
        except:
            pass

        if hasattr(self, '_df'):
            out['shape'] = self.df.shape

        return out
    
    @property
    def size(self):
        
        return os.path.getsize(self.path)
    
    @property
    def comment(self):
        
        return self._comment
    
    @comment.setter
    def comment(self, value):
        
        self._comment = value
    
    @property
    def df(self):
        
        if not hasattr(self, '_df'):
            
            self._df = self.get_df()
            
        return self._df 
    
    @property
    def dataframe(self):
        return self.df
    
    @property
    def name(self):
        
        if self._name is None:
            
            assert self.path is not None, 'Expect a name for data object.'
            return self.path.stem
        else:
            return self._name
        
    @name.setter
    def name(self, val):
        
        self._name = val
        
    @abstractmethod
    def get_df(self):
        pass
    
    def to_object(self, cls, name=None):
        
        if hasattr(cls, 'from_other'):
            obj = cls.from_other(self)
            obj.name = name or self.name
            return obj
        else:
            raise NotImplementedError('Convert method is not implemented.')
        
    def initialize(self, buffer=None):
        """
        Initializes the data interface by retrieving the data frame.
        
        Returns:
            self: The initialized data interface object.
        """
        if buffer is not None:
            self._df = buffer
        elif not hasattr(self, '_df'):
            self._df = self.get_df()
        
        return self
    
    def keys(self):
        """
        Returns a list of column names in the DataFrame.

        Returns:
            list: A list of column names.
        """
        return list(self.df.columns)
    
    def describe(self):
        """
        Returns a statistical summary of the DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing various statistical measures such as count, mean, standard deviation, minimum, maximum, and quartiles.
        """
        return self.df.describe()
    
    def count(self):
        """
        Returns the number of rows in the DataFrame.

        Returns:
            int: The number of rows in the DataFrame.
        """
        return self.df.count()
        
    def get(self, *names):
        """
        Retrieve data from the DataFrame based on the given column names.

        Parameters:
            *names: str
                The names of the columns to retrieve from the DataFrame.

        Returns:
            pandas.DataFrame or pandas.Series
                The requested data from the DataFrame.

        """
        if len(names) == 1:
            signame, = names

            if signame not in self.df.columns and (signame.lower() == 't' or signame.lower() == 'index'):
                return pd.DataFrame(np.asarray(self.df.index), index=self.df.index)

        return self.df[list(names)]
    
    
    def search(self, patt, ignore_case=True, raise_error=False):
        """
        Search for keys in the data structure that match a given pattern.

        Parameters:
        patt (str): The pattern to search for.
        ignore_case (bool, optional): Whether to ignore case when matching the pattern. Defaults to True.
        raise_error (bool, optional): Whether to raise a KeyError if no matching keys are found. Defaults to False.

        Returns:
        list: A list of keys that match the pattern.
        """
        
        found = []
        
        if ignore_case: patt = patt.lower()
        
        for key in self.keys():
            
            if ignore_case:
                
                key0 = key.lower()
            else:
                key0 = key
            
            if re.match(patt, key0):
                
                found.append(key)
                
                
        if not found and raise_error:
            
            raise KeyError(f'Cannot find any signal with pattern "{patt}".')
            
        try:
            
            ratios = [SequenceMatcher(a=patt, b=f).ratio() for f in found]
            
            _, found = zip(*sorted(zip(ratios, found))[::-1])
            
            
        except ValueError:
            pass
                
        return list(found)


    
    def memory_usage(self):
        """
        Returns the memory usage of the DataFrame.

        Returns:
            pandas.Series: A Series containing the memory usage of each column in the DataFrame.
        """
        return self.df.memory_usage(deep=True)
    
    def clean_config(self, replace=False):
        """
        Clean the configuration dictionary by removing any keys that are not present in the DataFrame.

        Returns:
            self: The current instance of the object.
        """
        if self.config is not None:
            new_cfg = dict()
            keys = self.df.keys()
            for k, v in self.config.items():
                if k in keys:
                    new_cfg[k] = v
                    
            retval = set(self.config.keys()) - set(new_cfg.keys())
            if replace:
                self.config = new_cfg
                  
        return sorted(retval)
    
    def search_similar(self, name, channel=True):
        """
        Searches for keys in the data structure that are similar to the given name.
        
        Args:
            name (str): The name to search for similarities.
        
        Returns:
            list: A list of keys in the data structure that are similar to the given name, sorted in descending order of similarity.
        """
        keys = self.keys() if not (hasattr(self, 'channels') and channel) else self.channels
        
        ratios = [SequenceMatcher(a=name, b=k).ratio() for k in keys]
        
        _, sorted_keys = zip(*sorted(zip(ratios, keys))[::-1])
        
        return sorted_keys
    
    def search_missing_config_items(self, num=5, channel=True, by_key=False):
        """
        Searches for missing configuration items in the data interface.

        Args:
            num (int): The maximum number of similar items to display for selection. Default is 5.
            channel (bool): Flag indicating whether to search for missing items in channels. Default is True.
            by_key (bool): Flag indicating whether to search for similar items by key. Default is False.

        Returns:
            dict: The updated configuration dictionary.
        """

        config = self.config or dict()

        chns = self.keys() if not (hasattr(self, 'channels') and channel) else self.channels

        for key, val in config.items():

            if not isinstance(val, (tuple, list, set)):

                val = [val]
                config[key] = val

            if any(c in chns for c in val):

                continue

            else:
                if by_key:
                    found = [self.search_similar(key, channel=channel)]
                else:
                    found = [self.search_similar(v, channel=channel) for v in val]
                topfound = list(chain(*zip_longest(*found)))[:num]

                while 1:

                    select = input(f'Select one of following key for "{key}" or type "n" for None: \n\t'
                                   + '\n\t'.join(f'{idx}: {s}' for idx, s in enumerate(topfound, start=1))
                                   + '\n')

                    if select == 'n':
                        break
                    if select == 'x':
                        raise KeyboardInterrupt
                    else:
                        try:
                            idx = int(select) - 1

                            k = topfound[idx]
                            config[key].append(k)
                            break
                        except (IndexError, ValueError):
                            print('Invalid input, try again')
                            continue

        return self.config
           
    def drop(self, *names, nonexist_ok=True):
        """
        Drop columns from the DataFrame.

        Args:
            *names: Variable length argument list of column names to be dropped.
            nonexist_ok (bool, optional): If True, ignore columns that do not exist. If False, raise KeyError for non-existent columns. Defaults to True.

        Returns:
            self: Returns the modified DataInterface object.

        Raises:
            KeyError: If a non-existent column is specified and nonexist_ok is False.
        """
        keys = []

        for name in names:
            if name in self.df:
                keys.append(name)
            elif not nonexist_ok:
                raise KeyError(f'"{name}" does not exist in "{self.name}".')

        self.df.drop(columns=keys, inplace=True)

        return self
    
    def select_rows(self, conds):
        """
        Selects rows from the DataFrame based on the given conditions.

        Parameters:
        - conds: A callable or a sequence of boolean values.
                 If a callable is provided, it should take the DataFrame as input and return a boolean Series or DataFrame.
                 If a sequence is provided, it should be a boolean sequence with the same length as the DataFrame.

        Raises:
        - ValueError: If the provided condition is invalid.

        Returns:
        - None

        """
        
        from collections import abc
        if hasattr(conds, '__call__'):
            self._df = self.df[conds(self.df)]
        elif isinstance(conds, abc.Sequence) or isinstance(conds, pd.Series) or isinstance(conds, np.ndarray):
            self._df = self.df.loc[conds]

        else:
            raise ValueError('Invalid condition.')
        
        return self
    
    def search_get(self, patt, ignore_case=False, raise_error=False):
        """
        Returns the subset of the DataFrame that matches the given pattern.

        Args:
            patt (str): The pattern to search for.
            ignore_case (bool, optional): Whether to ignore case when searching. Defaults to False.
            raise_error (bool, optional): Whether to raise an error if no matches are found. Defaults to False.

        Returns:
            pandas.DataFrame: The subset of the DataFrame that matches the pattern.
        """
        return self.df[self.search(patt, ignore_case, raise_error)]

    
    def load(self, *keys, mapping=None):
        """
        Load data from the specified keys into the data object.
        
        Args:
            keys: Variable number of keys to load data from.
            mapping: Optional mapping configuration for data extraction.
            
        Returns:
            The loaded data as a dictionary.
        """
        if not keys:       
            keys = mapping.values()
            
        @translate_config(mapping)
        @extract_channels(mapping)
        def get(self, *keys):
            return self.get(*keys)

        df = get(self, *keys)
        
        for k, v in df.items():
            self.df[k] = v
            
        return df
    

    def reload(self):
        """
        Reloads the data by deleting the existing dataframe and returning the instance.

        Returns:
            self: The instance of the DataInterface class.
        """
        if hasattr(self, '_df'):
            del self._df
        return self
   
    def merge(self, obj0, reset_index=False):
        """
        Merges the columns of another object into the current object.

        Parameters:
        - obj0: Another object to merge with.

        Returns:
        - self: The merged object.
        """
        keys = obj0.df.columns.difference(self.df.columns)
        
        if len(keys):
            if reset_index:
                self.reset_index()
                obj0.reset_index()
                
            self._df[keys] = obj0[keys]
                    
        return self
    
    def squeeze(self, *keys):
        """
        Selects and returns a new DataFrame containing only the specified columns.

        Parameters:
        *keys: Variable length argument representing the column names to be selected.

        Returns:
        self: Returns the modified DataInterface object with the new DataFrame.

        Raises:
        KeyError: If any of the specified column names are not found in the DataFrame, it raises a KeyError.
        """
        try:
            self._df = self.df[list(keys)]
            
        except KeyError:
            new_keys = list(set(self._df.columns) & set(keys))
            self._df = self.df[new_keys]
            
        return self
    
    def select_cols(self, keys):
        return self.squeeze(*keys)

    @staticmethod
    def pipeline(*funs, hook=None, ignore_error=True, asiterator=True):
        """
        Executes a series of functions on an object in a pipeline fashion.

        Args:
            *funs: Variable number of functions to be executed on the object.
            hook (str, optional): Path to a hook file containing functions to be executed. If provided, the functions 
                                  passed as arguments will be ignored. Defaults to None.
            ignore_error (bool, optional): Flag to indicate whether to ignore errors raised during execution. 
                                            If set to True, any exceptions raised during execution will be caught and 
                                            a warning will be issued. If set to False, the exception will be raised 
                                            and the execution will stop. Defaults to True.

        Raises:
            ValueError: If the input values are not callable functions.
            ValueError: If no callable functions are provided.

        Returns:
            A generator that yields the results of executing the functions on the object.
        """
        if not hook and len(funs)==1 and (isinstance(funs[0], (str, Path)) and Path(funs[0]).is_file()):
            hook = funs[0]
            funs = []   

        if funs and not all(hasattr(fun, '__call__') for fun in funs):
            raise ValueError('Input values must be callable.')

        elif not funs and hook and Path(hook).is_file():

            from datasurfer.datautils import parse_hook_file           
            funs = parse_hook_file(hook)

        if not funs:
            raise ValueError('No callable functions provided.')

        def wrapper(obj):
            def run():
                for fun in funs:
                    try:
                        yield fun(obj)
                    except Exception as err:
                        if ignore_error:
                            errname = err.__class__.__name__
                            tb = traceback.format_exc(limit=0, chain=False)
                            warnings.warn(f'Exception "{errname}" is raised while processing "{obj.name}": "{tb}"')
                            yield None
                        else:
                            raise
            if asiterator:
                return run()
            else:
                return list(run())            
            
        return wrapper

    
    def rename(self, **kwargs):
        """
        Renames the columns of the DataFrame using the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs where the key represents the current column name and the value represents the new column name.

        Returns:
            The modified DataFrame object.

        Example:
            df.rename(column1='new_column1', column2='new_column2')

        """
        keymap = kwargs

        @translate_config(keymap)
        def get(self):
            return self._df

        get(self)

        return self
    
    def resample(self, new_index=None):
        """
        Return a new DataFrame with all columns values interpolated
        to the new_index values.

        Parameters:
        - new_index (array-like, optional): The new index values to interpolate the columns to.
                                            If not provided, the original DataFrame is returned.

        Returns:
        - new_obj (NumpyObject): A new instance of the NumpyObject class with the resampled DataFrame.
        """
        from datasurfer.lib_objects.numpy_object import NumpyObject
        
        if new_index is not None:
            new_index = np.asarray(new_index)

            if new_index.size == 1:
                idx = self.df.index
                new_index = np.arange(min(idx), max(idx)+new_index, new_index)

            df_out = pd.DataFrame(index=new_index)
            df_out.index.name = self.df.index.name

            for colname, col in self.df.items():
                try:
                    col = col.astype(float)
                    df_out[colname] = np.interp(new_index, self.df.index, col)
                except (TypeError, ValueError):
                    warnings.warn(f'"{colname}" can not be resampled.')
        else:
            df_out = self.df

        new_obj = NumpyObject(path=str(self.path), config=self.config,
                              name=self.name,
                              comment=self.comment, df=df_out)

        return new_obj
    
    def fill_missing_keys(self, config=None):
        """
        Fills in missing keys in the configuration dictionary.

        Args:
            config (dict, optional): The configuration dictionary to fill in missing keys for.
                If not provided, the method will use the default configuration.

        Returns:
            dict: The updated configuration dictionary with missing keys filled in.
        """
        from datasurfer.datapool import combine_configs
        config = config or self.config
                
        config = combine_configs(parse_config(config))

        missing_keys = [k for k in config if k not in self.df.columns]

        for mk in missing_keys:        
            sigs = config[mk]
            for sig in sigs:
                for k, vs in config.items():
                    
                    if sig in vs and k in self.df.columns:
                        self.df[mk] = self.df[k]
            
        return self
    
    def to_clipboard(self, decimal='.'):
        from datasurfer.lib_objects.string_object import StringObject
        
        self.to_object(StringObject).to_clipboard(decimal=decimal)
        return self
    
    def to_numpy(self):
        """
        Converts the data to a NumPy array.

        Returns:
            numpy.ndarray: The data as a NumPy array.
        """
        return self.resample()
        
    def to_dict(self):
        """
        Converts the data interface object to a dictionary.

        Returns:
            dict: A dictionary representation of the data interface object.
        """
        out = dict()
        
        out['path'] = str(self.path)
        out['config'] = self.config
        out['comment'] = self.comment  
        out['name'] = self.name
        out['df'] = self.df.to_numpy()
        out['index'] = self.df.index
        out['columns'] = self.df.columns
        
        return out
    
    def to_csv(self, name=None, overwrite=True):
        """
        Export the DataFrame to a CSV file.

        Args:
            name (str, optional): The name of the CSV file. If not provided, the name will be generated based on the object's name. Defaults to None.
            overwrite (bool, optional): Whether to overwrite an existing file with the same name. Defaults to True.

        Returns:
            self: The current instance of the DataInterface object.
        """
        if name is None:
            name = self.name + '.csv'
            
        if overwrite or not Path(name).is_file():
            self.df.to_csv(name)
        
        return self

    def to_excel(self, name=None, overwrite=True):
        """
        Export the DataFrame to an Excel file.

        Args:
            name (str, optional): The name of the Excel file. If not provided, the name will be generated based on the object's name. Defaults to None.
            overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.

        Returns:
            self: The current instance of the DataInterface object.
        """
        if name is None:
            name = self.name + '.xlsx'
        
        if overwrite or not Path(name).is_file():
            self.df.to_excel(name)
        
        return self
    
    def save(self, name, overwrite=True):
        
        if overwrite or not Path(name).is_file():
            
            from datasurfer.lib_objects.numpy_object import NumpyObject
            
            dobj = NumpyObject(path=self.path, config=self.config, name=self.name, 
                           comment=self.comment, df=self.df)
            dobj.save(name)
        
        return self
    
    def clean(self):
        """
        Cleans the data by deleting the internal DataFrame object.
        
        Returns:
            self: The DataInterface object after cleaning.
        """
        if hasattr(self, '_df'):
            del self._df
        return self
                
    def close(self, clean=True):
        """
        Closes the file handler and performs optional cleanup.

        Args:
            clean (bool, optional): Flag indicating whether to perform cleanup. Defaults to True.
        """
        if hasattr(self, '_fhandler') and hasattr(self._fhandler, 'close'):
            self._fhandler.close()
            del self._fhandler

        if clean:
            self.clean()
            
        return self
    
    def link_library(self, lib, link_name=None):
        """
        Links a library to the data interface.

        Args:
            lib: The library to be linked.
            link_name (optional): The name to be used for the linked library. If not provided, the library's __name__ attribute will be used.

        Returns:
            The updated data interface object.

        """
        link_name = link_name or lib.__name__
        setattr(self, link_name, lib(self))
        return self
   
    @property
    def plot(self):
        """
        Generate a plot instance using the Plots class.

        Returns:
            Stat_Plots: An instance of the Plots class.
        """
        from datasurfer.lib_plots import Plots
        
        return Plots(self)
    
    @property
    def signal(self):
        """
        Generate statistic instance for the datapool objects.

        Returns:
            Stats: An instance of the Stats class.
        """
        from datasurfer.lib_signals import Signal
        
        return Signal(self)

    
    @property
    def mlearn(self):
        """
        This method initializes and returns an instance of the MLearn(machine learning) class.

        Returns:
            MLearn: An instance of the MLearn class.
        """
        from datasurfer.lib_mlearn import MLearn
        
        return MLearn(self)
    
    @property
    def multiprocessor(self):
        
        if not hasattr(self, '_multiproc'):   
                
            from datasurfer.lib_multiproc import MultiProc       
            self._multiproc = MultiProc(self)
            
        return self._multiproc
 
    mlp = multiprocessor
# %%
