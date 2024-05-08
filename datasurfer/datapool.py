# -*- coding: utf-8 -*-

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
from itertools import chain
from functools import reduce

from datasurfer.datautils import collect_files, combine_configs, show_pool_progress
    
random.seed()

  
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
        comments = kwargs.pop('comments', {})
        if isinstance(comments, (pd.DataFrame, pd.Series)):
            comments = comments.to_dict()
        
        
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
                
        elif isinstance(datobjects, (list, set, tuple)):
            
            if all(isinstance(s, (str, Path)) for s in datobjects) and all(Path(s).is_dir() for s in datobjects):
                
                datobjects = reduce(lambda x, y: 
                                    DataPool(x, config=config, interface=interface, **kwargs)
                                    + DataPool(y, config=config, interface=interface,  **kwargs), datobjects)    
        elif datobjects is not None:
                datobjects = [datobjects]
        
        if datobjects is None or len(datobjects)==0:
            datobjects = []
            

            
        objs = []
        from datasurfer import DataInterface
        for obj in datobjects:
            
            if isinstance(obj, DataInterface):
                
                if config:               
                    obj.config = config
                
                objs.append(obj)
                
            elif isinstance(obj, self.__class__):
                if config:
                
                    obj.config = config     
                objs.extend(obj.objs)        
                
                               
            elif isinstance(interface, type) and issubclass(interface, DataInterface):
                objs.append(interface(obj, config=config, **kwargs))
                
            elif isinstance(interface, str):
                from datasurfer import list_interfaces
                itfs = list_interfaces()
                interface_ = itfs['class'][interface]
                objs.append(interface_(obj, config=config, **kwargs))
                
            elif isinstance(obj, (str, Path)):
                key = Path(obj).suffix.lower()               
                if key in map_interface:
                    module, cls = map_interface[key]
                    cls = getattr(importlib.import_module(f'datasurfer.{module}'), cls)               
                    objs.append(cls(obj, config=config, **kwargs))
                else:
                    warnings.warn(f'Can not find any interface for "{obj}"')
      
        self.objs = sorted(set(objs), key=lambda x:x.name)
        self.apply_comments(**comments)
        
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
    
    def __contains__(self, obj):
        
        return obj in self.objs
    
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
    
    def __hash__(self):
        

        return hash(tuple([obj.__hash__() for obj in self.objs ]))
    
    def __eq__(self, pool0):
        
        return all(obj in pool0.objs for obj in self.objs) and all(obj in self.objs for obj in pool0.objs)
    
    def __rshift__(self, cls):
        
        return self.convert(cls)   
    
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
                    out = self.search_signal(patt, ignore_case=True)
                else:
                
                    out = self.__class__([self.get_object(name) for name in self.search_object(inval)])
            else:
                if inval in self.keys():
                    out = self.get_object(inval)                    
                else:
                    out = self.get_signal(inval.strip())
                    
            
        elif isinstance(inval, (list, tuple, set)) and len(inval):
            
            
            if all(na in self.keys() for na in inval):
                
                out = self.__class__([self.get_object(n) for n in inval])
                
            else:
                out = self.get_signal1Ds(*inval)
            
            
        elif hasattr(inval, '__call__'):
            from datasurfer import DataInterface
            if isinstance(inval, type) and issubclass(inval, DataInterface):
                
                out = self.__class__([obj for obj in self.objs if isinstance(obj, inval)])
            else:
                out = self.__class__(self.select(inval))

        
        elif isinstance(inval, np.ndarray) and inval.dtype=='bool':
            
            out = self.__class__(self.select(inval))
            
        elif isinstance(inval, pd.Series):
            
            out = self.__class__(self.select(inval))
        
        elif isinstance(inval, slice):
            
            out = self.__class__(self.objs[inval])
            
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
    
    @property
    def size(self):
        return self.__len__()
    
    
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
        try:
            ftype = pd.Series([Path(p).suffix for p in self.paths()], index=self.names(), name='File Type')
            date  = self.file_date() 
            size  = (self.file_size() / 1e6).round(4)     
        except FileNotFoundError:
            ftype = pd.Series(np.nan*np.ones(len(self.objs)), name='File Type')
            date = pd.Series(np.nan*np.ones(len(self.objs)), name='File Date')
            size = pd.Series(np.nan*np.ones(len(self.objs)), name='File Size')
           
        if verbose:
            signal = self.signal_count(pbar=False)
            length = pd.Series([obj.__len__() for obj in self.objs], 
                            index=self.names(), name='Signal Length')
            count = self.count_signal_size(pbar=False)
            
            memory = (self.memory_usage(pbar=pbar) / 1e6).round(4) 
            comments = pd.Series(self.comments(), name='Comment')      
        
            df = pd.concat([comments, signal, length, count, memory, itype, ftype, size, date, path], axis=1)
        else:
            df = pd.concat([itype, ftype, size, date, path], axis=1)
                    
        return df.dropna(how='all')
                
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
        """
        Sorts the objects in the datapool based on their names.

        Returns:
            self: The sorted datapool object.
        """
        self.objs.sort(key=lambda x: x.name)
        return self
    
    def items(self):
        
        return self.objs
    
    def types(self):
        """
        Returns a dictionary containing the objects in the datapool grouped by their class types.
        
        Returns:
            dict: A dictionary where the keys are the class names and the values are lists of objects
                    belonging to that class.
        """
        out = dict()
        
        for obj in self.objs:
            out.setdefault(obj.__class__.__name__, []).append(obj)
            
        return out
    
    def length(self):
        """
        Returns the length of the data pool.
        
        Returns:
            int: The length of the data pool.
        """
        return self.__len__()
    
    
    def paths(self):
        """
        Returns a pandas Series containing the file paths of the objects in the datapool.

        Returns:
            pd.Series: A pandas Series object with the file paths as values and object names as index.
        """
        return pd.Series([obj.path for obj in self.objs], index=self.names(), name='File Path')

    def file_size(self):
        """
        Calculate the file size for each file in the datapool.

        Returns:
            pd.Series: A pandas Series object containing the file sizes, with the file names as the index.
        """
        
        return pd.Series(dict((name, path.stat().st_size) for name, path in self.paths().items()), name='File Size')
   
    def file_date(self):
        """
        Returns a pandas Series containing the file modification dates of the objects in the datapool.
        
        Returns:
            pd.Series: A pandas Series with the file modification dates.
        """

        ctimes = [datetime.datetime.fromtimestamp(os.path.getmtime(obj.path)) 
                  for obj in self.objs]
        return pd.Series(ctimes, index=self.names(), name='File Date')
    
    def comments(self, pbar=True):
        """
        Returns a DataFrame containing the names and comments of the objects in the datapool.
        """
        @show_pool_progress('Processing', show=pbar)
        def get(self):
            for obj in self.objs:
                yield obj.name, obj.comment
                
        return pd.Series(dict((name, comment) for name, comment in get(self)), name='Comment')
    
    def apply_comments(self, **comments):
        """
        Apply comments to the objects in the datapool.

        Args:
            **comments: Keyword arguments mapping object names to comments.

        Returns:
            self: The modified datapool object.
        """
        for name, comment in comments.items():
            try:
                if comment:
                    self.get_object(name)._comment = comment
            except (NameError, AttributeError):
                pass
        return self
    
    def configs(self):
        """
        Returns a DataFrame containing the configurations of all objects in the datapool.
        
        Returns:
            pandas.DataFrame: A DataFrame with object names as columns and their corresponding configurations as values.
        """
        return pd.DataFrame(dict((obj.name, obj.config) for obj in self.objs))
        
    def list_signals(self, count=None, pbar=True):
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
    
    def count_signal_size(self, pbar=True):
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
    
    
    def initialize(self, buffer=None, pbar=True):
        """
        Initializes the datapool by calling the `initialize` method on each object in the datapool.

        Args:
            pbar (bool, optional): Whether to show a progress bar during initialization. Defaults to True.
        """
        @show_pool_progress('Initializing', show=pbar, set_init=True)
        def get(self):
            for obj in self.objs:
                bf = None if obj not in buffer else buffer[buffer.index(obj)].df
                obj.initialize(buffer=bf)               
                yield
                
        buffer = [] if buffer is None else buffer
        list(get(self))

        return self
    
    def append(self, obj):
        """
        Appends an object to the datapool.

        Args:
            obj: The object to append.

        Returns:
            self: The updated datapool object.
        """
        from datasurfer import DataInterface
        assert isinstance(obj, DataInterface), 'Input object must be a DataInterface object.'

        if obj in self.objs:
            self.objs.pop(self.objs.index(obj))
            warnings.warn(f'Object "{obj.name}" is already in the datapool.')
            
        self.objs.append(obj)
        return self
    
    def extend(self, dp):
        """
        Extends the datapool with a list of DataInterface objects.

        Args:
            objs (list): A list of DataInterface objects to be added to the datapool.

        Returns:
            self: The updated datapool object.

        Raises:
            AssertionError: If any of the input objects is not an instance of DataInterface.
        """
        assert isinstance(dp, DataPool), 'Input objects must be Data Pool object.'
        
        for obj in dp.objs:
            
            self.append(obj)

        return self
    
    def extend(self, objs):
        from datasurfer import DataInterface
        assert all(isinstance(obj, DataInterface) for obj in objs), 'Input objecs must be DataInterface object.'
        self.objs.extend(objs)
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
        >>> dp = DataPool([obj1, obj2, obj3])
    
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
        if mask is not None:
            assert len(mask) == len(self.objs), 'Mask length must match the number of objects.'
            objs = [obj for obj, msk in zip(self.objs, mask) if msk]
        else:
            objs = self.objs
            
        for obj in objs:
            
            try:                

                                            
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
    
    def iter_objsignals(self, *signames, ignore_error=True):
        """
        Iterates over the objects in the datapool and returns a DataFrame for each specified signal name.

        Args:
            *signames: Variable length argument list of signal names.
            ignore_error (bool, optional): If True, ignores any errors encountered during iteration. Defaults to True.

        Yields:
            tuple: A tuple containing the object name and the corresponding DataFrame for each specified signal name.

        """
        for idx, obj in enumerate(self.objs):
            msk = ~np.ones(len(self.objs), dtype=bool)
            msk[idx] = True
            vals = [next(self.iter_signal(signame, ignore_error=ignore_error, mask=msk)) for signame in signames]
            df = pd.concat(vals, axis=1)
            df.columns = signames
            yield obj.name, df
        
    def get_row(self, row, columns=None, ignore_error=True, pbar=True):    
        
        @show_pool_progress('Processing', pbar)
        def get(self):

            for name, df in self.iter_objsignals(*columns, ignore_error=ignore_error):
                try: 
                    res = df.loc[row]
                    
                    ds = pd.Series(res, index=columns)
                    ds.name = name
                        
                except KeyError as err:
                    if ignore_error:     
                        
                        errname = err.__class__.__name__
                        tb = traceback.format_exc(limit=0, chain=False)
                        warnings.warn(f'Exception "{errname}" is raised while processing "{name}": "{tb}"')

                        ds = pd.Series(np.nan * np.ones(columns.__len__()), index=columns)                        
                        ds.name = name       
                        
                    else:
                        raise
                finally:
                    yield ds  
        columns = columns or self.list_signals()            
        rows = list(get(self))
        df = pd.concat(rows, axis=1).T
        return df                    
                
            
   
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

            
        """
        dats = list(self.iter_signal(
            signame, ignore_error=ignore_error, mask=mask))

        return pd.concat(dats, axis=1)
    
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

        cols = [signame]   
        out = pd.DataFrame(np.concatenate(dats, axis=0), columns=cols)

        if reindex:
            out.index = np.arange(len(out))

        return out
    
    def get_signal1Ds(self, *signals, ignore_error=True, mask=None):
        """
        Retrieves multiple 1D signals from the datapool.

        Args:
            *signals: Variable length argument list of signal names to retrieve.
            ignore_error (bool, optional): Whether to ignore errors when retrieving signals. Defaults to True.
            mask (array-like, optional): Mask to apply to the retrieved signals. Defaults to None.

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved signals concatenated along the columns.
        """
        out = []
        for sig in signals:          
            out.append(self.get_signal1D(sig, ignore_error=ignore_error, mask=mask))
            
        out = pd.concat(out, axis=1)
        return out
    
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
            
    
    def get_testobj(self, idx=None):
        """
        Returns the test object with the smallest file size for test purposes.

        Args:
            idx (int, optional): The index of the test object to retrieve. If not provided, the test object with the smallest file size will be returned.

        Returns:
            The test object with the smallest file size.
        """
        if not idx:
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
    
    def map(self, func, ignore_error=True, pbar=True):
               
        @show_pool_progress('Processing', pbar)
        def get(self):
            
            for obj in self.objs:

                try:
                    yield func(obj)
                except Exception as err:
                    if ignore_error:
                        errname = err.__class__.__name__
                        tb = traceback.format_exc(limit=0, chain=False)
                        warnings.warn(f'Exception "{errname}" is raised while processing "{obj.name}": "{tb}"')
                        yield None
                    else:
                        raise

        return list(get(self))
        

        
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

    def merge(self, pool0, only_data=True, raise_error=False):  
        """
        Merges the objects from another datapool into the current datapool.

        Args:
            pool0 (DataPool): The datapool to merge.

        Returns:
            DataPool: The merged datapool.
        """
        names = self.names()

        for obj in pool0.objs:
            if not only_data and obj.name not in names:
                self.objs.append(obj)
            elif obj.name in names:
                self.get_object(obj.name).merge(obj)

        return self
    
    def merge_data(self, pool0):
        """
        Merges the data from another datapool into the current datapool.

        Args:
            pool0 (DataPool): The datapool containing the data to be merged.

        Returns:
            DataPool: The current datapool with the merged data.
        """

        names0 = pool0.names()

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
    
    def convert(self, cls, pbar=True):
        """
        Converts all the objects in the DataPool to a different class.

        Args:
            cls (class): The class to convert the objects to.
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            DataPool: A new DataPool object with the converted objects.
        """
        
        @show_pool_progress('Converting', show=pbar)
        def get(self):
            for obj in self.objs:
                new_obj = obj.to_object(cls)
                yield new_obj
        
        return DataPool(list(get(self)), config=self.config, name=self.name)
        
    def select(self, mask_array):
        
        '''
            return data objects from pool according to mask.
        '''
        if isinstance(mask_array, (dict, pd.Series)):
            
            out = [self.get_object(name) for name, bool in mask_array.items() if (name in self.names() and bool)]
        elif hasattr(mask_array, '__call__'):

            out = [obj for obj in self if mask_array(obj)]
        else:
            assert len(mask_array) == len(self.objs), "The length of the mask array does not match the number of data objects."
            
            out = [obj for obj, msk in zip(self.objs, mask_array) if msk]
        return self.__class__(out)
    
    @staticmethod
    def pipeline(*funs, hook=None, pbar=True, ignore_error=True, asiterator=False):
        """
        Applies a series of functions to each object in the data pool and yields the result.

        Args:
            *funs: Variable number of functions to be applied to each object.

        Yields:
            The modified object after applying all the functions.
        """
        from datasurfer import DataInterface
        if asiterator: pbar=False
        
        def wrapper(dp):
            @show_pool_progress('Processing', show=pbar)
            def fun(dp):
                for obj in dp.objs:                            
                    list(DataInterface.pipeline(*funs, hook=hook, ignore_error=ignore_error)(obj))
                    yield obj
                    
            if asiterator:
                return fun(dp)
            else:
                return list(fun(dp))
        return wrapper      
                
    def deepcopy(self, *pipeline, pbar=True):
        """
        Create a deep copy of the DataPool object.

        Parameters:
        - pbar (bool): Whether to show a progress bar during the copy process. Default is True.

        Returns:
        - DataPool: A new DataPool object that is a deep copy of the original object.
        """
        from datasurfer import DataInterface
        from datasurfer.lib_objects.numpy_object import NumpyObject
        
        @show_pool_progress('Copying', show=pbar)
        def fun(self):           
            for obj in self.objs:
                obj = obj.to_object(NumpyObject)
                if pipeline:
                    list(DataInterface.pipeline(*pipeline)(obj))
                              
                yield obj
                                
        objs = list(fun(self))
        
                        
        return self.__class__(datobjects=objs)
    
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
    
    def rename(self, **kwargs):
        """
        Renames objects in the datapool.

        Args:
            **kwargs: Keyword arguments where the key is the new name and the value is the current name of the object.

        Returns:
            self: The updated datapool object.

        """
        for key, val in kwargs.items():
            try:
                self.get_object(val).name = key
            except NameError:
                pass
                
        return self
               
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
    
    def def2json(self):
        """
        Convert the data in the datapool to a JSON string.

        Returns:
            str: A JSON string representation of the datapool data.
        """
        out = dict()

        for name, feature in (('path', self.paths()),
                              ('comment', self.comments())):

            for key, value in feature.items():
                out.setdefault(key, dict())[name] = value

        out = json.dumps(out, indent=4)
        return out
    
    def to_dataframe(self, *names, pbar=True):
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
            
            for objname, df in self.iter_objsignals(*names):
                                
                index = pd.MultiIndex.from_product([[objname], df.columns])
                
                df.columns = index
                                
                yield df

        if not names:
            
            names = self.list_signals()
            
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

    def to_summary_excel(self, name, keys=None, pbar=True):
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
                for name, df in self.iter_objsignals(*keys):        
                    df.to_excel(writer, sheet_name=name)
                    yield True  
                    
            keys = keys or self.list_signals()              
            list(fun(self))        
            return self
    
    def to_datalake(self, *hooks, **condis):
        """
        Converts the data in the DataPool to a DataLake based on the given conditions.

        Parameters:
        - condis (keyword arguments): Conditions to filter the data. Each condition should be a callable function.

        Returns:
        - dlk (DataLake): The resulting DataLake object.

        Raises:
        - AssertionError: If the value of any condition is not a callable function.
        """

        from datasurfer import DataLake
        from datasurfer.datautils import parse_hook_file

        out = dict()
        
        for hook in hooks:
            for func in parse_hook_file(hook):
                assert hasattr(func, '__call__'), 'The value of the condition must be a callable function.'
                for obj in self.objs:
                    if func(obj):
                        out.setdefault(func.__name__, []).append(obj) 
        

        for name, func in condis.items():
            assert hasattr(func, '__call__'), 'The value of the condition must be a callable function.'

            for obj in self.objs:
                if func(obj):
                    out.setdefault(name, []).append(obj)

        pools = [DataPool(objs, name=name) for name, objs in out.items() if len(objs)]

        if not len(condis):
            pools = [self]
                   
        dlk = DataLake(pools)

        return dlk
               
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
    
    def save_def(self, name=None, pbar=True, save_comment=True):
        """
        Save the definitions of objects in the datapool to a file.

        Args:
            name (str): The name of the file to save the definitions to. If not provided, the definitions will not be saved.

        Returns:
            dict: A dictionary containing the saved definitions.

        Raises:
            None

        """
        out = dict()
        @show_pool_progress('Exporting', show=pbar)
        def get(self):
            for obj in self.objs:
                out[obj.name] = dict()
                out[obj.name]['path'] = str(obj.path)
                if save_comment and obj.comment:
                    out[obj.name]['comment'] = obj.comment
                yield
        list(get(self))
        if name:   
            if name.lower().endswith('.json'):
                with open(name, 'w') as file:
                    json.dump(out, file, indent=4)
            elif name.lower().endswith('.yaml') or name.lower().endswith('.yml'):
                import yaml
                with open(name, 'w') as file:
                    yaml.safe_dump(out, file)

        return out
    
    @staticmethod
    def from_def(path, **kwargs):
        """
        Load data from a JSON file and create a DataPool object.

        Parameters:
        - name (str): The name of the JSON file to load.
        - **kwargs: Additional keyword arguments to pass to the DataPool constructor.

        Returns:
        - dp (DataPool): The DataPool object created from the loaded data.
        """
        if isinstance(path, str):
            if path.lower().endswith('.json'):
                with open(path, 'r') as file:
                    data = json.load(file)
            elif path.lower().endswith('.yaml') or path.lower().endswith('.yml'):
                import yaml
                with open(path, 'r') as file:
                    data = yaml.safe_load(file)
        elif isinstance(path, dict):
            data = path
        elif isinstance(path, pd.DataFrame):
            data = path.to_dict()
        else:
            raise ValueError('Input value must be a string or a dictionary.')
            
        _, values = zip(*data.items())
        
        paths = [val['path'] for val in values]
        comments = (dict((key, val.get('comment', None)) for key, val in data.items()) 
                    if kwargs.pop('apply_comments', True) else dict())

        dp = DataPool(paths, comments=comments, **kwargs)    

        return dp
    

    @staticmethod    
    def load_numpy(name, keys=None, pbar=True, count=None):
        """
        Load data from a numpy .npz file and create NumpyObject instances.

        Parameters:
            name (str): The name of the .npz file to load.
            keys (list, optional): A list of keys to load from the .npz file. If None, all keys will be loaded. Default is None.
            pbar (bool, optional): Whether to show a progress bar during loading. Default is True.
            count (int, optional): The maximum number of objects to load. If None, all objects will be loaded. Default is None.

        Returns:
            self: The updated instance of the class.

        """
        from datasurfer.lib_objects.numpy_object import NumpyObject
        
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
                        obj = NumpyObject(path=dat['path'],
                                          config=dat['config'],
                                          comment=dat['comment'],
                                          name=dat['name'],
                                          df=df)
                        yield obj
                        ncount = ncount + 1
                        if ncount >= num:
                            break
                        
            self = DataPool(list(fun(None)))              
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
                key = f'Pool_{k:02d}'
                out.setdefault(key, []).append(objs.pop())

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
    
    def fill_missing_keys(self, config=None, pbar=True):
        """
        Fills missing keys in the datapool objects using the provided config.

        Args:
            config (dict, optional): The configuration dictionary containing the missing keys and their values. Defaults to None.
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            self: The updated datapool object.
        """
        @show_pool_progress('Processing', show=pbar)
        def get(self):
            for obj in self.objs:               
                obj.fill_missing_keys(config=config)
                yield
        
        list(get(self))
        return self
    
    def link_library(self, lib, link_name=None):
        """
        Links the datapool to a library.

        Args:
            lib (Library): The library to link the datapool to.

        Returns:
            self: The updated datapool object.
            
        """
        link_name = link_name or lib.__name__
        setattr(self, link_name, lib(self))
        return self
    
    @property
    def plot(self):
        """
        Generate a statistical plot using the Stat_Plots class.

        Returns:
            Stat_Plots: An instance of the Stat_Plots class.
        """
        from datasurfer.lib_plots import Plots
        
        return Plots(self)
    
    @property
    def signals(self):
        """
        Generate statistical summaries for the datapool objects.

        Returns:
            Stats: An instance of the Stats class.
        """
        from datasurfer.lib_signals import Signal
        
        return Signal(self)
    
    @property
    def mlearn(self):
        
        from datasurfer.lib_mlearn import MLearn
        
        return MLearn(self)
    
    @property
    def multiprocessor(self):
        
        if not hasattr(self, '_multiproc'):   
                
            from datasurfer.util_multiproc import MultiProc       
            self._multiproc = MultiProc(self)
            
        return self._multiproc
 
    mlp = multiprocessor
    
    @property
    def configurator(self):
        
        if not hasattr(self, '_configurator'):
            from datasurfer.util_config import Configurator        
            self._configurator = Configurator(self)
            
        return self._configurator

    cfg = configurator


#%% Main Loop

if __name__ == '__main__':
    
    pass
    

