# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:45:05 2023

@author: YUW1SI
"""


#%% Import Libraries

import os
#import h5py
import re
import datetime

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

from dataobjects import Data_Interface, DATA_OBJECT, AMERES_OBJECT,\
                         ASAMMDF_OBJECT, PANDAS_OBJECT, MATLAB_OBJECT,\
                         AMEGP_OBJECT

random.seed()
#%% Collect files

def collect_files(root, *patts, warn_if_double=True, ignore_double=False):
    
    '''
    
        Gather files from the data directory that match patterns
        
        Paramters:
            root: directory
            patts: patterns in format of regular expression
            warn_if_double: warning if files with same name exist.
            
        Return:
            iterator of found files
            
        Usage:
            
            # collecte csv-file in current folder, return a list
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
    
    Mapping_Interfaces = {
                         '.csv': PANDAS_OBJECT,
                         '.xlsx': PANDAS_OBJECT,
                         '.mf4': ASAMMDF_OBJECT,
                         '.mat': MATLAB_OBJECT,
                         '.results': AMERES_OBJECT,
                         '.amegp': AMEGP_OBJECT,
                     }
    
    def __init__(self, datobjects=None, config=None, interface=None, **kwargs):
        
        '''
            Creating a data pool to process dataset.
            
            Inputs:
                
                datobjects: path of data directory or list of data file paths 
                config:     configuration in dictionary format or path to a config file, in format of excel or json 

                
            Output:
                
                A Pool-Object to access the data
                
            usage:
                
                # Create a data pool
                
                pool = Data_Pool(r'/data', config='config.json')
                
                # Get data from pool, return a pandas dataframe
                
                TRotor = pool['tRotor']
        
        
        '''
        
        self.silent = kwargs.pop('silent', False)
        self.name = kwargs.pop('name', None)
                    

        if isinstance(datobjects, (str, Path)): 
            
            dpath = Path(datobjects)
            
            if dpath.is_dir():
                
                self.name = dpath.name if self.name is None else self.name
                self.path =  Path(dpath).absolute()
            
                patts = kwargs.pop('pattern', None)
                
                patts = patts if patts else ['.+\\'+k for k  in self.__class__.Mapping_Interfaces.keys()]
                
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
                try:                   
                    objs.append(self.__class__.Mapping_Interfaces[key](obj, config=config, **kwargs))
                except KeyError:
                    pass        
                
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
        
        obj_names = [obj.__class__.__name__ for obj in self.objs]
        
        obj_types = set(obj_names)
        
        s = ';'.join(f'{typ}({obj_names.count(typ)})' for typ in obj_types)
        
        if self.name:
            return  f'<{self.__class__.__name__}"{self.name}"@{s}>'   
        else: 
            return f'<{self.__class__.__name__}@{s}>' 

    def __iter__(self):
        
        return iter(self.objs)
    
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
        
        assert hasattr(fun, '__call__'), 'Input value must be callable.'
        
        self.apply(name, fun)

    @property
    def config(self):
        
        out = combine_configs(*[obj.config for obj in self.objs if obj.config])
                        
        return out
    
    @config.setter    
    def config(self, value):
        
        for obj in self.objs:
            if hasattr(obj, '_df'):
                del obj._df
                gc.collect()
            
            obj.config = value
            
    def describe(self, pbar=False):
        
        path = self.paths()
        signal = self.signal_count(pbar=False)
        length = pd.Series([obj.__len__() for obj in self.objs], 
                          index=self.names(), name='Signal Length')
        count = self.count(pbar=False)
        size = (self.file_size() / 1e6).round(4)
        memory = (self.memory_usage(pbar=pbar) / 1e6).round(4)        
        itype = pd.Series([obj.__class__.__name__ for obj in self.objs], 
                          index=self.names(), name='Interface')
        ftype = pd.Series([Path(p).suffix for p in self.paths()], index=self.names(), name='File Type')
        date = self.file_date()        
        df = pd.concat([signal, length, count, memory, itype, ftype, size, date, path], axis=1)
                    
        return df
                
    def memory_usage(self, pbar=True):
        
        @show_pool_progress('Calculating', show=pbar)
        def fun(self):
            
            for obj in self.objs:
                
                yield obj.name, obj.memory_usage().sum()
        
        s = pd.Series(dict(fun(self)), name='Memory Usage')
        return s

    def keys(self):
        
        '''
            return all file names from pool
        
            usage:
                
                keys = obj.keys()
        '''
        
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
        
        
        ctimes = [datetime.datetime.fromtimestamp(os.path.getmtime(obj.path)) 
                  for obj in self.objs]
        return pd.Series(ctimes, index=self.names(), name='File Date')
    
    def comments(self):
        
        return pd.DataFrame(dict((obj.name, obj.comment) for obj in self.objs))
    
    def configs(self):
        
        return pd.DataFrame(dict((obj.name, obj.config) for obj in self.objs))
        
    def signals(self, count=None, pbar=False):
        
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
        
        @show_pool_progress('Processing', show=pbar)
        def get(self):
            
            for obj in self.objs:
                
                yield obj.name, len(obj.keys())
                
        return pd.Series(dict(get(self)), name='Signal Count')
    
    def count(self, pbar=True):
        
        @show_pool_progress('Processing', show=pbar)
        def get(self):
            
            for obj in self.objs:
                
                yield obj.name, obj.count().sum()
                
        return pd.Series(dict(get(self)), name='Signal Size')
    
    
    def initialize(self, pbar=True):
        
        @show_pool_progress('Initializing', show=pbar, set_init=True)
        def get(self):
            
            for obj in self.objs:
                
                obj.initialize()
                
                yield
        
        list(get(self))
        
        return self
    
    def load_signals(self, *keys, mapping=None, pbar=True):
        
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
        
            Parameters
            ----------
            signame : string, signal name
                
            ignore_error : type:bool
                The default is True.
                
            mask : numpy array, optional
                selection mask. The default is None.
    
            Yields
            ------
            object of data interface

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
        
        dats = list(self.iter_signal(
                signame, ignore_error=ignore_error, mask=mask))
        
        return pd.concat(dats, axis=1)
    

    def get_signal1D(self, signame, ignore_error=True, mask=None, reindex=True):
        
        dats = list(self.iter_signal(
                signame, ignore_error=ignore_error, mask=mask))
                        
        out = pd.DataFrame(np.concatenate(dats, axis=0), columns=[signame])
        
        if reindex:
            
            out.index = np.arange(len(out))
        
        return out
    
    def get_signals(self, *signames, ignore_error=True, mask=None):
        
        return dict((signame, self.get_signal(signame, ignore_error=ignore_error, mask=mask))
                for signame in signames)
    
    def get_object(self, name):
        
        for obj in self.objs:
            
            if obj.path.stem == name:
                return obj
        else:
            raise NameError(f'Can not find any "{name}"')
            
    
    def get_testobj(self):
        
        idx = self.file_size().values.argsort()[0]
        
        return self.objs[idx]
    
            
    def search_object(self, patt, ignore_case=True, raise_error=False):
        
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
            
            raise KeyError(f'Cannot find any object with pattern "{patt}".')
                
        return found
    
    def search_signal(self, patt, ignore_case=False, raise_error=False, pbar=True):
    
        @show_pool_progress('Searching', pbar)
        def get(self):        
            for obj in self.objs:                
                yield obj.search(patt, ignore_case=ignore_case, 
                                      raise_error=raise_error)
                
        
        return sorted(set(chain(*get(self)))) 
    
    def sort_objects(self, key):
        
        return list(self.objs[:]).sort(key=lambda obj:obj.key)
    

    def apply(self, signame, methode, ignore_error=True, pbar=True):
        

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

    def merge(self, pool0):
        
        names = [obj.name for obj in self.objs]
        
        for obj in pool0.objs:
            
            if obj.name not in names:
                
                self.objs.append(obj)
                
        return self
    
    def merge_data(self, pool0):
        

        names0 = [obj.name for obj in pool0.objs]
        
        for obj in self.objs:
            
            if obj.name in names0:
                
                obj0 = pool0.objs[names0.index(obj.name)]
                
                obj.merge(obj0)
                
                
        return self
         
    def pop(self, name):
        
        return self.objs.pop(self.objs.index(self.get_object(name)))
        
        
    
    def squeeze(self, *keys, pbar=True):
        
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
        
        for obj in self.objs:
            
            for fun in funs:
                
                fun(obj)
                
            yield obj
                
                
    def deepcopy(self, pbar=True):

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
        
        pbar = kwargs.pop('pbar', True)
        
        @show_pool_progress('Renaming', show=pbar)
        def get(self):
            
            for obj in self.objs: 
                
                obj.rename(**kwargs)
                
                yield
                
        list(get(self))
        
        return self
    
    def to_dataframe(self, columns=None, pbar=True):
        
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
        
        with pd.ExcelWriter(name) as writer:
            @show_pool_progress('Saving', show=pbar)
            def fun(self):       
                for obj in self.objs:        
                    obj.df.to_excel(writer, sheet_name=obj.name)
                    yield True            
            list(fun(self))        
            return self             
            
    def save(self, name, pbar=True):
        
        @show_pool_progress('Saving', show=pbar, set_init=True)
        def fun(self):
            
            for res in self.iter_dict():
                
                yield res
               
        out = dict(list(fun(self)))                
        np.savez(name, **out)
        self.initialized = True
        return self
    
    def load(self, name, keys=None, pbar=True, count=None):
                
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
        
        out = dict()
        objs = self.objs[:]
        
        if shuffle:
            random.shuffle(objs)

        while objs:
            
            for k in range(chunk):

                if not objs: break
                
                out.setdefault(k, []).append(objs.pop())
                

        return [self.__class__(v) for v in out.values()]        
            

    
    def close(self, clean=True, pbar=True):
        
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
    
    dp = DataPool(r'D:\01_Python\stock\Analyse_Data\20191128')
    
    ps = dp.split_pool(2)
    
    from multiprocessing.dummy import Pool
    
    def foo(p):
        
        return p.keys()
    
    p = Pool(2)
    
    print(p.map(foo, ps))
    

