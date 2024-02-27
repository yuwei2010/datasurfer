# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:45:05 2023

@author: YUW1SI
"""


#%% Import Libraries
import sys
import os
import re
import datetime
import json
import importlib

import pandas as pd
import numpy as np
import warnings
import random

import gc
import traceback

from pathlib import Path
from tqdm import tqdm
   
from itertools import chain
from functools import reduce, wraps

from .lib_objects import Data_Interface

random.seed()
#%% Collect files

def collect_files(root, *patts, warn_if_double=True, ignore_double=False):
    '''
    Gather files from the data directory that match patterns
    
    Parameters:
        root (str): The root directory to start searching for files.
        *patts (str): Variable number of patterns in the format of regular expressions.
        warn_if_double (bool, optional): If True, a warning will be raised if files with the same name exist. Defaults to True.
        ignore_double (bool, optional): If True, files with the same name will be ignored and not returned. Defaults to False.
        
    Yields:
        Path: An iterator of found file paths.
        
    Usage:
        # Collect csv files in the current folder, return a list
        collection = list(collect_files('.', r'.*\.csv', warn_if_double=True))
    '''
    if isinstance(root, (list, tuple, set)):
        root = chain(*root)
    
    found = {}
       
    for r, ds, fs in os.walk(root):
        for f in fs:
            ismatch = any(re.match(patt, f) for patt in patts)
            
            if ismatch: 
                path = (Path(r) / f)
                
                if warn_if_double and f in found:
                    dirs = '\n\t'.join(found[f])
                    warnings.warn(f'"{path}" exists already in:\n{dirs}')
                    
                if (f not in found) or (not ignore_double) :
                    yield path
                    
                found.setdefault(f, []).append(r)
                
                
                
#%% Show_Pool_Progress_bar
def show_pool_progress(msg, show=False, set_init=True, count=None):
    """
    Decorator function that adds progress bar functionality to a method in a data pool object.

    Parameters:
    - msg (str): The message to display in the progress bar.
    - show (bool): Whether to show the progress bar or not. Default is False.
    - set_init (bool): Whether to set the 'initialized' attribute of the data pool object to True after the method is executed. Default is True.
    - count (int): The total count of objects to iterate over. If None, the count is determined by the length of the data pool object. Default is None.

    Returns:
    - The decorated method.
    """
    
    def decorator(func):
    
        @wraps(func)
        def wrapper(self, *args, **kwargs):
                    
            res = func(self, *args, **kwargs)

            flag_pbar = (not self.silent) and (not self.initialized or show)
            
            if count is None:                
                num = len(self.objs)
            else:
                num = count
                
            rng = range(num)
            
            if flag_pbar:
                pbar = tqdm(rng)
                iterator = pbar
            else:
                iterator = rng
             
            for idx in iterator:
                
                if flag_pbar: 
                
                    if count is None:

                        pbar.set_description(f'{msg} "{self.objs[idx].name}"')
                        
                    else:
                        
                        pbar.set_description(f'{msg}')
            
                yield next(res)
                
            if set_init:
                
                self.initialized = True
                        
        return wrapper
    
    return decorator


#%% Combine configs

def combine_configs(*cfgs):
    """
    Combines multiple configuration dictionaries into a single dictionary.

    Args:
        *cfgs: Variable number of configuration dictionaries.

    Returns:
        dict: A dictionary containing the combined configurations.

    Example:
        >>> cfg1 = {'a': 'apple', 'b': 'banana'}
        >>> cfg2 = {'a': 'avocado', 'c': 'cherry'}
        >>> combine_configs(cfg1, cfg2)
        {'a': ['apple', 'avocado'], 'b': ['banana'], 'c': ['cherry']}
    """
    out = dict()
    
    for k, v in chain(*[cfg.items() for cfg in cfgs]):
        
        if isinstance(v, str):   
             
            out.setdefault(k, set()).add(v)
            
        else:
            if k in out:
                out[k] = out[k].union(set(v))
            else:
                out[k] = set(v)
                
            
    if not out:
        out = None
    else:
        for k, v in out.items():
            out[k] = sorted(v)
            
    return out
  
#%% Data Pool

class DataPool(object):
    """
    A class representing a data pool to process datasets.
    
    Attributes:
        Mapping_Interfaces (dict): A dictionary mapping file extensions to corresponding data interface objects.
        
    Methods:
        __init__: Initializes the DataPool object.
        __enter__: Enters the context manager.
        __exit__: Exits the context manager.
        __repr__: Returns a string representation of the DataPool object.
        __iter__: Returns an iterator over the DataPool object.
        __len__: Returns the number of objects in the DataPool.
        __add__: Adds two DataPool objects.
        __or__: Performs a union operation on two DataPool objects.
        __and__: Performs an intersection operation on two DataPool objects.
        __sub__: Performs a difference operation on two DataPool objects.
        __getitem__: Retrieves data from the DataPool object.
        __setitem__: Sets a callable object for a given name in the DataPool object.
        config: Gets or sets the configuration of the DataPool object.
        describe: Returns a DataFrame with descriptive information about the DataPool object.
        memory_usage: Returns the memory usage of each object in the DataPool.
        keys: Returns the names of all objects in the DataPool.
        names: Returns the names of all objects in the DataPool.
        sorted: Sorts the objects in the DataPool by name.
        items: Returns a list of all objects in the DataPool.
        types: Returns a dictionary mapping object types to objects in the DataPool.
        length: Returns the number of objects in the DataPool.
        paths: Returns the file paths of all objects in the DataPool.
        file_size: Returns the file size of all objects in the DataPool.
        file_date: Returns the file date of all objects in the DataPool.
        comments: Returns a DataFrame with comments for each object in the DataPool.
        configs: Returns a DataFrame with configurations for each object in the DataPool.
        signals: Returns a list of all unique signal names in the DataPool.
        signal_count: Returns the number of signals in each object of the DataPool.
    """
    
    # Mapping_Interfaces = {
    #     '.csv': 'PANDAS_OBJECT',
    #     '.xlsx': 'PANDAS_OBJECT',
    #     '.mf4': 'ASAMMDF_OBJECT',
    #     '.mat': 'MATLAB_OBJECT',
    #     '.results': 'AMERES_OBJECT',
    #     '.amegp': 'AMEGP_OBJECT',
    # }
    
    def __init__(self, datobjects=None, config=None, interface=None, **kwargs):
        """
        Initializes the DataPool object.
        
        Args:
            datobjects (str, Path, list, set, tuple): Path of data directory or list of data file paths.
            config (dict, str): Configuration in dictionary format or path to a config file.
            interface (type): Data interface class.
            **kwargs: Additional keyword arguments.
        
        Usage:
            # Create a data pool
            pool = DataPool(r'/data', config='config.json')
            
            # Get data from pool, return a pandas dataframe
            TRotor = pool['tRotor']
        """

        if Path(__file__).parent.joinpath('map_interface.json').exists():
            sys.path.insert(0, Path(__file__).parent / '..')
            map_interface = json.load(open(Path(__file__).parent.joinpath('map_interface.json'), 'r', encoding='utf8')) 
        else:
            map_interface = dict()

        
        self.silent = kwargs.pop('silent', False)
        self.name = kwargs.pop('name', None)
                    

        if isinstance(datobjects, (str, Path)): 
            
            dpath = Path(datobjects)
            
            if dpath.is_dir():
                
                self.name = dpath.name if self.name is None else self.name
                self.path =  Path(dpath).absolute()
            
                patts = kwargs.pop('pattern', None)
                
                patts = patts if patts else ['.+\\'+k for k  in map_interface.keys()]
                
                if not isinstance(patts, (list, tuple, set)):
                    patts = [patts]
                
                datobjects = sorted(collect_files(dpath, *patts))
                
            elif dpath.is_file():
                
                datobjects = self.__class__([]).load(datobjects).objs
                
            else:
                
                raise IOError(f'Can not find the dir or file "{dpath}".')    
                
        elif isinstance(datobjects, (list, set, tuple)) and len(datobjects):
            
            if all(isinstance(s, (str, Path)) for s in datobjects) and all(Path(s).is_dir() for s in datobjects):
                
                datobjects = reduce(lambda x, y: 
                                    DataPool(x, config=config, interface=interface, **kwargs)
                                    +DataPool(y, config=config, interface=interface,  **kwargs), datobjects)    


        if datobjects is None:
            datobjects = []
            
        objs = []
        
        for obj in datobjects:
            
            if isinstance(obj, Data_Interface):
                
                if config:
                    
                    obj.config = config
                
                objs.append(obj)
                
            elif isinstance(interface, type) and issubclass(interface, Data_Interface):
                objs.append(interface(obj, config=config, **kwargs))
                
            else:
                key = Path(obj).suffix.lower()
                
                
                if key in map_interface:
                    cls = getattr(importlib.import_module('datastructure'), map_interface[key] )               
                    objs.append(cls(obj, config=config, **kwargs))
       
                
        self.objs = sorted(set(objs), key=lambda x:x.name)
        
        self.initialized = False
        

    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        
        
        self.close()
        
        if exc_type:
            
            return False
        
        return True        

    def __repr__(self):
        """
        Returns a string representation of the object.

        The string representation includes the class name, object name (if available),
        and the count of each object type in the `objs` list.

        Returns:
            str: A string representation of the object.
        """
        obj_names = [obj.__class__.__name__ for obj in self.objs]
        obj_types = set(obj_names)
        s = ';'.join(f'{typ}({obj_names.count(typ)})' for typ in obj_types)
        if self.name:
            return f'<{self.__class__.__name__}"{self.name}"@{s}>'
        else:
            return f'<{self.__class__.__name__}@{s}>'

    def __iter__(self):
        
        return iter(self.objs)
    
    def __contains__(self, name):
        
        return name in self.names()
    
    def __len__(self):
        
        return len(self.objs)    

    def __add__(self, pool0):
        
        objs = list(set(self.objs + pool0.objs))
        
        return self.__class__(objs)
    
    def __or__(self, pool0):
        
        return self.__class__(set(self.objs) | set(pool0.objs))

    def __and__(self, pool0):
        
        return self.__class__(set(self.objs) & set(pool0.objs))
    
    def __sub__(self, pool0):
        
        objs = list(set(self.objs).difference(pool0.objs))
        
        return self.__class__(objs)
    
    
    def __getitem__(self, inval):
        """
        Retrieve an item from the datapool.

        Parameters:
            inval (str, list, tuple, set, function, np.ndarray, pd.Series, pd.DataFrame): The key or keys to retrieve from the datapool.

        Returns:
            object: The retrieved item(s) from the datapool.
        """
        if isinstance(inval, str):
            
            if '*' in inval:
                
                if inval.strip()[0] in '#@%':
                    
                    patt = inval.strip()[1:]
                    
                    out = self.search_signal(patt)
                else:
                
                    out = [self.get_object(name) for name in self.search_object(inval)]
            else:
                if inval in self.keys():
                    out = self.get_object(inval)
                else:
                    out = self.get_signal(inval)
            
        elif isinstance(inval, (list, tuple, set)):
            
            out = []
            
            if all(na in self.keys() for na in inval):
                
                out = [self.get_object(n) for n in inval]
                
            else:
                
                for n in inval:
                    
                    out.append(self.get_signal1D(n))
                    
                out = pd.concat(out, axis=1)
            
            
        elif hasattr(inval, '__call__'):
            if isinstance(inval, type) and issubclass(inval, Data_Interface):
                
                out = [obj for obj in self.objs if isinstance(obj, inval)]
            else:
                out = self.__class__(self.select(self.which(inval)))
        
        elif isinstance(inval, np.ndarray) and inval.dtype=='bool':
            
            out = self.__class__(self.select(inval))
            
        elif isinstance(inval, (pd.Series, pd.DataFrame)):
            
            out = self.__class__(self.select(inval.to_numpy()))
            
        else:
            out = self.objs[inval]
            
        return out
    
    def __setitem__(self, name, fun):
        """
        Set the value of an item in the datapool.

        Args:
            name (str): The name of the item.
            fun (callable): The function to be associated with the item.

        Raises:
            AssertionError: If the input value is not callable.
        """
        assert hasattr(fun, '__call__'), 'Input value must be callable.'

        self.apply(name, fun)

    @property
    def config(self):
        """
        Combines the configurations of all objects in the datapool.

        Returns:
            dict: The combined configuration.
        """
        out = combine_configs(*[obj.config for obj in self.objs if obj.config])
        return out
    
    @config.setter    
    def config(self, value):
        """
        Configure the objects in the datapool with the given value.
        
        Args:
            value: The value to be set as the configuration for the objects.
        """
        for obj in self.objs:
            if hasattr(obj, '_df'):
                del obj._df
                gc.collect()
            
            obj.config = value
            
    def describe(self, verbose=False, pbar=False):
        """
        Generates a summary DataFrame containing information about the data pool.

        Parameters:
            pbar (bool): Whether to display a progress bar during memory usage calculation. Default is False.

        Returns:
            pandas.DataFrame: Summary DataFrame with the following columns:
                - Signal: Number of signals in the data pool.
                - Signal Length: Length of each signal.
                - Count: Number of occurrences of each signal.
                - Memory: Memory usage of the data pool in megabytes.
                - Interface: Class name of each signal object.
                - File Type: File extension of each signal file.
                - Size: Size of the data pool file in megabytes.
                - Date: Date of the data pool file.
                - Path: File path of the data pool file.
        """
        path = self.paths()
        itype = pd.Series([obj.__class__.__name__ for obj in self.objs], 
                          index=self.names(), name='Interface')
        ftype = pd.Series([Path(p).suffix for p in self.paths()], index=self.names(), name='File Type')
        date = self.file_date() 
        size = (self.file_size() / 1e6).round(4)     
           
        if verbose:
            signal = self.signal_count(pbar=False)
            length = pd.Series([obj.__len__() for obj in self.objs], 
                            index=self.names(), name='Signal Length')
            count = self.count(pbar=False)
            
            memory = (self.memory_usage(pbar=pbar) / 1e6).round(4)        
        
            df = pd.concat([signal, length, count, memory, itype, ftype, size, date, path], axis=1)
        else:
            df = pd.concat([itype, ftype, size, date, path], axis=1)
                    
        return df
                
    def memory_usage(self, pbar=True):
            """
            Calculate the memory usage of each object in the datapool.

            Parameters:
            - pbar (bool): Whether to show a progress bar during calculation. Default is True.

            Returns:
            - s (pd.Series): Series containing the memory usage of each object.
            """
            
            @show_pool_progress('Calculating', show=pbar)
            def fun(self):
                
                for obj in self.objs:
                    
                    yield obj.name, obj.memory_usage().sum()
            
            s = pd.Series(dict(fun(self)), name='Memory Usage')
            return s

    def keys(self):
        """
        Returns a list of keys in the datapool.
        """
        return [obj.name for obj in self.objs]
    
    def names(self):
        
        return self.keys()
    
    def sorted(self):
        
        self.objs.sort(key=lambda x: x.name)
        return self
    
    def items(self):
        
        return self.objs
    
    def types(self):
        
        out = dict()
        
        for obj in self.objs:
            
            out.setdefault(obj.__class__.__name__, []).append(obj)
            
        return out
    
    def length(self):
        
        return self.__len__()
    
    
    def paths(self):
        
        return pd.Series([str(obj.path) for obj in self.objs], index=self.names(), name='File Path')

    def file_size(self):
        
        return pd.Series(dict(zip(self.names(), map(os.path.getsize, self.paths().values))), name='File Size')
   
    def file_date(self):
        """
        Returns a pandas Series containing the file modification dates of the objects in the datapool.
        
        Returns:
            pd.Series: A pandas Series with the file modification dates.
        """
        ctimes = [datetime.datetime.fromtimestamp(os.path.getmtime(obj.path)) 
                  for obj in self.objs]
        return pd.Series(ctimes, index=self.names(), name='File Date')
    
    def comments(self):
        """
        Returns a DataFrame containing the names and comments of the objects in the datapool.
        """
        return pd.DataFrame(dict((obj.name, obj.comment) for obj in self.objs))
    
    def configs(self):
        """
        Returns a DataFrame containing the configurations of all objects in the datapool.
        
        Returns:
            pandas.DataFrame: A DataFrame with object names as columns and their corresponding configurations as values.
        """
        return pd.DataFrame(dict((obj.name, obj.config) for obj in self.objs))
        
    def signals(self, count=None, pbar=False):
        """
        Returns a list of unique keys from the objects in the datapool.

        Args:
            count (int, optional): The maximum number of objects to consider. If not specified, all objects are considered.
            pbar (bool, optional): Whether to show a progress bar during processing. Default is False.

        Returns:
            list: A sorted list of unique keys from the objects in the datapool.
        """
        if count is not None: 
            count = max([1, count])
        
        @show_pool_progress('Processing', show=pbar, count=count)
        def get(self):
            
            if count is not None:
                objs = self.objs[:count]
                
            else:
                objs = self.objs
            
            for obj in objs:
                
                yield obj.keys()
                
        keys = sorted(set(chain(*get(self))))    
        
        return keys
    
    def signal_count(self, pbar=True):
        """
        Returns a pandas Series containing the count of signals for each object in the datapool.

        Parameters:
            pbar (bool): Whether to show a progress bar during processing. Default is True.

        Returns:
            pandas.Series: A Series object with the signal count for each object.
        """
        @show_pool_progress('Processing', show=pbar)
        def get(self):
            for obj in self.objs:
                yield obj.name, len(obj.keys())
                
        return pd.Series(dict(get(self)), name='Signal Count')
    
    def count(self, pbar=True):
        """
        Count the signal size for each object in the datapool.

        Args:
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            pd.Series: A pandas Series containing the signal size for each object.
        """
        @show_pool_progress('Processing', show=pbar)
        def get(self):
            for obj in self.objs:
                yield obj.name, obj.count().sum()
                
        return pd.Series(dict(get(self)), name='Signal Size')
    
    
    def initialize(self, pbar=True):
        """
        Initializes the datapool by calling the `initialize` method on each object in the datapool.

        Args:
            pbar (bool, optional): Whether to show a progress bar during initialization. Defaults to True.
        """
        @show_pool_progress('Initializing', show=pbar, set_init=True)
        def get(self):
            for obj in self.objs:
                obj.initialize()
                yield

        list(get(self))

        return self
    
    def load_signals(self, *keys, mapping=None, pbar=True):
        """
        Load signals from the data pool.

        Args:
            *keys: Variable length argument list of keys to load.
            mapping: Optional mapping object to apply to the loaded signals.
            pbar: Boolean flag indicating whether to show a progress bar.

        Returns:
            The updated data pool object.

        Examples:
            # Load signals with keys 'signal1' and 'signal2'
            dp.load_signals('signal1', 'signal2')

            # Load signals with key 'signal3' and apply a mapping object
            dp.load_signals('signal3', mapping=my_mapping_object)

        """
        @show_pool_progress('Loading', show=pbar)
        def get(self):
            for obj in self.objs:
                obj.load(*keys, mapping=mapping)
                yield

        list(get(self))
        return self

    @show_pool_progress('Processing', show=False, set_init=True)
    def iter_signal(self, signame, ignore_error=True, mask=None):
        '''
        Iterates over the data objects in the datapool and yields data frames for a given signal name.

        Parameters
        ----------
        signame : str
            The name of the signal to iterate over.

        ignore_error : bool, optional
            If True, any exceptions raised during processing will be ignored and a warning will be issued.
            If False, exceptions will be raised. Default is True.

        mask : numpy array, optional
            A selection mask to filter the data objects. Only objects corresponding to True values in the mask will be processed.
            Default is None, which means all objects will be processed.

        Yields
        ------
        pandas.DataFrame
            A data frame containing the data for the specified signal name.

        Examples
        --------
        >>> dp = DataPool()
        >>> dp.add_data_object(obj1)
        >>> dp.add_data_object(obj2)
        >>> dp.add_data_object(obj3)

        >>> for df in dp.iter_signal('temperature'):
        ...     print(df)
        ...
        # Output:
        #    obj1
        # 0  25.0
        # 1  26.0
        # 2  27.0
        #
        #    obj2
        # 0  30.0
        # 1  31.0
        # 2  32.0
        #
        #    obj3
        # 0  35.0
        # 1  36.0
        # 2  37.0
        '''

        for idx, obj in enumerate(self.objs):
            
            try:
                
                if (mask is None) or (mask is not None and mask[idx]):
                    
                        
                    df = obj.get(signame)
                        
                    df.columns = [obj.name]                      
                    df.index = np.arange(0, len(df))

                    yield df
                
            except Exception as err:
                
                if ignore_error:     
                    
                    errname = err.__class__.__name__
                    tb = traceback.format_exc(limit=0, chain=False)
                    warnings.warn(f'Exception "{errname}" is raised while processing "{obj.name}": "{tb}"')

                    df = pd.DataFrame(np.nan * np.ones(obj.__len__()), columns=[obj.name])                        
                    df.index = np.arange(0, obj.__len__())                    
                    yield df
                    
                else:
                    raise


    
    def get_signal(self, signame, ignore_error=True, mask=None):
        """
        Retrieves the data for a given signal name.

        Args:
            signame (str): The name of the signal to retrieve.
            ignore_error (bool, optional): Whether to ignore errors if the signal is not found. Defaults to True.
            mask (str, optional): A mask to filter the data. Defaults to None.

        Returns:
            pandas.DataFrame: The concatenated data for the given signal.

        Examples:
            >>> dp = DataPool()
            >>> dp.get_signal('signal1')
            # Returns the concatenated data for 'signal1'

            >>> dp.get_signal('signal2', ignore_error=False)
            # Raises an error if 'signal2' is not found

            >>> dp.get_signal('signal3', mask='2022-01-01')
            # Returns the concatenated data for 'signal3' filtered by the mask '2022-01-01'
        """
        dats = list(self.iter_signal(
            signame, ignore_error=ignore_error, mask=mask))

        return pd.concat(dats, axis=1)
    
    def get_signal1D(self, signame, ignore_error=True, mask=None, reindex=True):
        """
        Retrieve a 1-dimensional signal from the datapool.

        Parameters:
        - signame (str): The name of the signal to retrieve.
        - ignore_error (bool): Whether to ignore errors if the signal is not found. Default is True.
        - mask (ndarray): An optional mask to apply to the retrieved data. Default is None.
        - reindex (bool): Whether to reindex the output DataFrame. Default is True.

        Returns:
        - DataFrame: A pandas DataFrame containing the retrieved signal.
        """
        dats = list(self.iter_signal(
            signame, ignore_error=ignore_error, mask=mask))

        out = pd.DataFrame(np.concatenate(dats, axis=0), columns=[signame])

        if reindex:
            out.index = np.arange(len(out))

        return out
    
    def get_signals(self, *signames, ignore_error=True, mask=None):
        """
        Retrieves the values of multiple signals from the datapool.

        Args:
            *signames: Variable-length argument list of signal names.
            ignore_error (bool): Flag indicating whether to ignore errors when retrieving signals.
            mask: Optional mask to apply to the retrieved signals.

        Returns:
            dict: A dictionary mapping signal names to their corresponding values.
        """
        return dict((signame, self.get_signal(signame, ignore_error=ignore_error, mask=mask))
                    for signame in signames)
    
    def get_object(self, name):
        """
        Retrieve an object from the datapool by its name.

        Args:
            name (str): The name of the object to retrieve.

        Returns:
            object: The object with the specified name.

        Raises:
            NameError: If no object with the specified name is found.
        """
        for obj in self.objs:
            if obj.name == name:
                return obj
        else:
            raise NameError(f'Can not find any "{name}"')
            
    
    def get_testobj(self):
        """
        Returns the test object with the smallest file size for test purpos.
        
        Returns:
            The test object with the smallest file size.
        """
        idx = self.file_size().values.argsort()[0]
        return self.objs[idx]
    
            
    def search_object(self, patt, ignore_case=True, raise_error=False):
        """
        Search for objects in the datapool that match a given pattern.

        Parameters:
        - patt (str): The pattern to search for.
        - ignore_case (bool): Whether to ignore the case when matching the pattern. Default is True.
        - raise_error (bool): Whether to raise a KeyError if no objects are found. Default is False.

        Returns:
        - found (list): A list of keys of the objects that match the pattern.
        """
        found = []

        if ignore_case:
            patt = patt.lower()

        for key in self.keys():

            if ignore_case:
                key0 = key.lower()
            else:
                key0 = key

            if re.match(patt, key0):
                found.append(key)

        if not found and raise_error:
            raise KeyError(f'Cannot find any object with pattern "{patt}".')

        return found
    
    def search_signal(self, patt, ignore_case=False, raise_error=False, pbar=True):
        """
        Search for a signal in the datapool.

        Args:
            patt (str): The pattern to search for.
            ignore_case (bool, optional): Whether to ignore case when searching. Defaults to False.
            raise_error (bool, optional): Whether to raise an error if the signal is not found. Defaults to False.
            pbar (bool, optional): Whether to show a progress bar during the search. Defaults to True.

        Returns:
            list: A sorted list of unique signals matching the pattern.
        """

        @show_pool_progress('Searching', pbar)
        def get(self):        
            for obj in self.objs:                
                yield obj.search(patt, ignore_case=ignore_case, raise_error=raise_error)
            
        return sorted(set(chain(*get(self))))
    
    def sort_objects(self, key):
        """
        Sorts the objects in the datapool based on the specified key.

        Args:
            key: The key to sort the objects by.

        Returns:
            None. The objects in the datapool are sorted in-place.
        """
        return list(self.objs[:]).sort(key=lambda obj: obj.key)
    def sort_objects(self, key):
        
        return list(self.objs[:]).sort(key=lambda obj:obj.key)
    

    def apply(self, signame, methode, ignore_error=True, pbar=True):
        """
        Apply a given method to each object in the datapool.

        Args:
            signame (str): The name of the signal to be modified.
            methode (callable): The method to be applied to each object.
            ignore_error (bool, optional): Whether to ignore errors raised during processing. Defaults to True.
            pbar (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            self: The modified datapool object.
        """

        @show_pool_progress('Processing', pbar)
        def get(self):
            for obj in self.objs:

                try:
                    obj.df[signame] = methode(obj)
                except Exception as err:
                    if ignore_error:
                        errname = err.__class__.__name__
                        tb = traceback.format_exc(limit=0, chain=False)
                        warnings.warn(f'Exception "{errname}" is raised while processing "{obj.name}": "{tb}"')
                    else:
                        raise

                yield True

        list(get(self))
        return self

    def merge(self, pool0, raise_error=False):  
        """
        Merges the objects from another datapool into the current datapool.

        Args:
            pool0 (DataPool): The datapool to merge.

        Returns:
            DataPool: The merged datapool.
        """
        names = [obj.name for obj in self.objs]

        for obj in pool0.objs:
            if obj.name not in names:
                self.objs.append(obj)
            elif raise_error:
                raise ValueError(f'Object "{obj.name}" already exists in the datapool.')    

        return self
    
    def merge_data(self, pool0):
        """
        Merges the data from another datapool into the current datapool.

        Args:
            pool0 (DataPool): The datapool containing the data to be merged.

        Returns:
            DataPool: The current datapool with the merged data.
        """

        names0 = [obj.name for obj in pool0.objs]

        for obj in self.objs:

            if obj.name in names0:

                obj0 = pool0.objs[names0.index(obj.name)]

                obj.merge(obj0)

        return self
         
    def pop(self, name):
        """
        Removes and returns the object with the given name from the datapool.

        Args:
            name (str): The name of the object to be removed.

        Returns:
            object: The removed object.

        Raises:
            ValueError: If the object with the given name does not exist in the datapool.
        """
        return self.objs.pop(self.objs.index(self.get_object(name)))
        
        
    
    def squeeze(self, *keys, pbar=True):
        """
        Squeezes the objects in the datapool by applying the squeeze method to each object.

        Args:
            *keys: Variable length argument representing the keys to be passed to the squeeze method of each object.
            pbar (bool): Flag indicating whether to display a progress bar during the squeezing process. Default is True.

        Returns:
            self: The modified datapool object after squeezing.

        """
        @show_pool_progress('Processing', pbar)
        def get(self):
            for obj in self.objs:
                obj.squeeze(*keys)
                yield
        list(get(self))
        return self
        
    def select(self, mask_array):
        
        '''
            return data objects from pool according to mask.
        '''
        assert len(mask_array) == len(self.objs), "The length of the mask array does not match the number of data objects."
        
        return [obj for obj, msk in zip(self.objs, mask_array) if msk]
    
    def pipe(self, *funs):
        """
        Applies a series of functions to each object in the data pool and yields the result.

        Args:
            *funs: Variable number of functions to be applied to each object.

        Yields:
            The modified object after applying all the functions.
        """
        for obj in self.objs:
            for fun in funs:
                fun(obj)
            yield obj
                
                
    def deepcopy(self, pbar=True):
            """
            Create a deep copy of the DataPool object.

            Parameters:
            - pbar (bool): Whether to show a progress bar during the copy process. Default is True.

            Returns:
            - DataPool: A new DataPool object that is a deep copy of the original object.
            """
            @show_pool_progress('Copying', show=pbar)
            def fun(self):
                
                for name, dat in self.iter_dict():
                    df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
                    obj = DATA_OBJECT(path=dat['path'],
                                      config=dat['config'],
                                      comment=dat['comment'],
                                      name=dat['name'],
                                      df=df)                
                    yield obj
                    
                    
            objs = list(fun(self))
                    
            return self.__class__(objs)
    
    def iter_dict(self):
        """
        Iterate over the objects in the datapool and yield a dictionary for each object.
        
        Returns:
            A generator that yields a tuple containing the object name and a dictionary
            containing information about the object.
        """
        for obj in self.objs:
            out = dict()
            out['path'] = str(obj.path)
            out['config'] = obj.config
            out['comment'] = obj.comment  
            out['name'] = obj.name
            out['df'] = obj.df.to_numpy()
            out['index'] = obj.df.index
            out['columns'] = obj.df.columns
            
            yield obj.name, out
            
    def rename_signals(self, **kwargs):
        """
        Renames signals in the datapool.

        Args:
            **kwargs: Additional keyword arguments to be passed to the `rename` method of each signal object.

        Returns:
            self: The updated datapool object.
        """
        pbar = kwargs.pop('pbar', True)

        @show_pool_progress('Renaming', show=pbar)
        def get(self):
            for obj in self.objs:
                obj.rename(**kwargs)
                yield

        list(get(self))

        return self
    
    def to_dataframe(self, columns=None, pbar=True):
            """
            Convert the objects in the datapool to a pandas DataFrame.

            Args:
                columns (list, optional): List of column names to include in the DataFrame. Defaults to None.
                pbar (bool, optional): Whether to show a progress bar. Defaults to True.

            Returns:
                pandas.DataFrame: The combined DataFrame of all objects in the datapool.
            """
            
            @show_pool_progress('Processing', show=pbar)
            def fun(self):
                
                for obj in self.objs:
                    
                    df = obj.df.reset_index()
                    
                    if columns is not None:
                        
                        df = df[columns]
                    
                    index =  pd.MultiIndex.from_product([[obj.name], df.columns])
                    
                    df.columns = index
                                    
                    yield df

            dfs = list(fun(self))
            return pd.concat(dfs, axis=1)
                
    def to_csvs(self, wdir, pbar=True):
        """
        Save the data from each object in the datapool to separate CSV files.

        Args:
            wdir (str): The directory path where the CSV files will be saved.
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            self: The current instance of the datapool.
        """
        wdir = Path(wdir)
        wdir.mkdir(exist_ok=True)

        @show_pool_progress('Saving', show=pbar)
        def fun(self):
            for obj in self.objs:
                obj.df.to_csv(wdir / (obj.name+'.csv'))
                yield True

        list(fun(self))

        return self

    def to_excels(self, wdir, pbar=True):
        """
        Save each object's DataFrame to an Excel file in the specified directory.

        Args:
            wdir (str): The directory path where the Excel files will be saved.
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            self: The current instance of the DataPool class.
        """
        wdir = Path(wdir)
        wdir.mkdir(exist_ok=True)
        
        @show_pool_progress('Saving', show=pbar)
        def fun(self):       
            for obj in self.objs:
                obj.df.to_excel(wdir / (obj.name+'.xlsx'))
                yield True
            
        list(fun(self))        
        return self

    def to_excel(self, name, pbar=True):
        """
        Save the data pool to an Excel file.

        Parameters:
        - name (str): The name of the Excel file to save.
        - pbar (bool): Whether to show a progress bar while saving. Default is True.

        Returns:
        - self: The data pool object.

        """
        with pd.ExcelWriter(name) as writer:
            @show_pool_progress('Saving', show=pbar)
            def fun(self):       
                for obj in self.objs:        
                    obj.df.to_excel(writer, sheet_name=obj.name)
                    yield True            
            list(fun(self))        
            return self
            
    def save(self, name, pbar=True):
        """
        Save the data pool to a file.

        Parameters:
        - name (str): The name of the file to save the data pool to.
        - pbar (bool): Whether to show a progress bar during the saving process. Default is True.

        Returns:
        - self: The updated instance of the DataPool class.

        """
        @show_pool_progress('Saving', show=pbar, set_init=True)
        def fun(self):
            for res in self.iter_dict():
                yield res
               
        out = dict(list(fun(self)))                
        np.savez(name, **out)
        self.initialized = True
        return self
    
    def load(self, name, keys=None, pbar=True, count=None):
        """
        Load data from a numpy .npz file and create DATA_OBJECT instances.

        Parameters:
            name (str): The name of the .npz file to load.
            keys (list, optional): A list of keys to load from the .npz file. If None, all keys will be loaded. Default is None.
            pbar (bool, optional): Whether to show a progress bar during loading. Default is True.
            count (int, optional): The maximum number of objects to load. If None, all objects will be loaded. Default is None.

        Returns:
            self: The updated instance of the class.

        """
        with np.load(name, allow_pickle=True) as npz:
            npzkeys = npz.keys()
            
            if count is None:
                num = len(npzkeys)
            else:
                num = count
            
            @show_pool_progress('Loading', show=pbar, count=num)
            def fun(self):
                ncount = 0
                for k in npzkeys:
                    if (keys is None) or (k in keys):
                        dat = npz[k].item()            
                        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
                        obj = DATA_OBJECT(path=dat['path'],
                                          config=dat['config'],
                                          comment=dat['comment'],
                                          name=dat['name'],
                                          df=df)
                        yield obj
                        ncount = ncount + 1
                        if ncount >= num:
                            break
        
            self.objs = list(fun(self))                
            self.initialized = True
            
            del npz.f

        return self

    def split_pool(self, chunk=2, shuffle=True):
        """
        Splits the pool of objects into multiple chunks.

        Args:
            chunk (int): The number of chunks to split the pool into. Default is 2.
            shuffle (bool): Whether to shuffle the objects before splitting. Default is True.

        Returns:
            list: A list of DataPool instances, each containing a chunk of objects from the original pool.
        """
        out = dict()
        objs = self.objs[:]

        if shuffle:
            random.shuffle(objs)

        while objs:
            for k in range(chunk):
                if not objs:
                    break
                out.setdefault(k, []).append(objs.pop())

        return [self.__class__(v) for v in out.values()]
            

    
    def close(self, clean=True, pbar=True):
        """
        Closes the datapool.

        Args:
            clean (bool, optional): Indicates whether to clean up the datapool. Defaults to True.
            pbar (bool, optional): Indicates whether to show a progress bar. Defaults to True.
        """
        @show_pool_progress('Closing', show=pbar)
        def get(self):
            for obj in self.objs:
                if hasattr(obj, 'close'):
                    obj.close()
                if clean and hasattr(obj, '_df'):
                    del obj._df
                yield

        list(get(self))
        if clean:
            self.initialized = False
        gc.collect()

        return None





#%% Main Loop

if __name__ == '__main__':
    
    pass
    

