# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:19:25 2023

@author: YUW1SI
"""

#%% Import Libraries

import os
#import h5py
import re


import pandas as pd
import numpy as np
import json
import warnings
import scipy.io


from xml.etree.ElementTree import fromstring

from pathlib import Path


try:
    import asammdf    
    from asammdf import set_global_option
    
    set_global_option("raise_on_multiple_occurrences", False)   
except:    
    warnings.warn('Can not import "asammdf"')

try:
    import mdfreader
except:    
    warnings.warn('Can not import "mdfreader"')
    

from difflib import SequenceMatcher
from itertools import chain
from functools import wraps


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
#%% Extract channels

def extract_channels(newconfig=None):
    
    def decorator(func):
    
        @wraps(func)
        def wrapper(self, *keys, **kwargs):
            
            mapping = dict() if newconfig is None else newconfig
            
            config = mapping if self.config is None else self.config

            foo = lambda k: config[k] if k in config else k
            newkeys = [foo(k) for k in keys]
            
            channels = self.channels if hasattr(self, 'channels') else self.df.columns
            
            outkeys = []
            
            for k in newkeys:

                if isinstance(k, str): 
                    
                    if k in channels:
                        
                        outkeys.append(k)
                    else:
                        
                        warnings.warn(f'"{k}" not found.')
                else:
                    
                    for kk in k:
                        
                        if kk in channels:
                            
                            outkeys.append(kk)
                            break
                    else:
                        warnings.warn(f'"{k}" not found.')
                        
            return func(self, *outkeys, **kwargs)             
            
        return wrapper
    
    return decorator

#%% Translate Config

def translate_config(newconfig=None):
    
    def decorator(func):
    
        @wraps(func)
        def wrapper(self, *keys, **kwargs):
                        
            if (hasattr(self, 'config') and self.config is not None) or newconfig is not None:
                
                config = newconfig if newconfig is not None else self.config
                                             
                res = func(self, *keys, **kwargs)
                
                if isinstance(res, pd.DataFrame):
                    
                    for k, v in config.items():
                        
                        if isinstance(v, str):
                        
                            res.columns = res.columns.str.replace(v, k, regex=False)
                            
                        elif isinstance(v, (list, tuple, set)):
                            
                            for vv in v:
                                
                                res.columns = res.columns.str.replace(vv, k, regex=False)

                return res
            
            else: 
                
                return func(self, *keys, **kwargs)

        return wrapper
    
    return decorator

#%% Data_Interface

class Data_Interface(object):
   
    def __init__(self, path, config=None, name=None, comment=None):

        
        if config is not None: 
            
            if isinstance(config, (str, Path)):
                if str(config).endswith('.json'):
            
                    config = json.load(open(config))
                    
                else:
                    raise IOError('Unknown config format, expect json.')
                
            elif isinstance(config, (list, tuple, set)):
                
                if all(isinstance(s, str) for s in config):
                    config = dict((v, v) for v in config)
                    
                elif all(isinstance(s, dict) for s in config):
                    config = combine_configs(*list(config))
                    
                else:
                    raise TypeError('Can not handle config type.')
                    
            elif not isinstance(config, dict):
                raise TypeError('Unknown config format, expect dict')
                
        if path is not None:
            
            path = Path(path).absolute() 
        
        self._name = name
        self.path = path
        self.config = config
        self._comment = comment
        
    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        
        self.close()
        
        if exc_type:
            
            return False
        
        return True

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

    @property
    def index(self):
        return np.asarray(self.df.index)
    
    @property
    def meta_info(self):
        
        out = dict()
        
        out['path'] = str(self.path)
        out['comment'] = self.comment  
        out['name'] = self.name
        
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
    def name(self):
        
        if self._name is None:
            
            assert self.path is not None, 'Expect a name for data object.'
            return self.path.stem
        else:
            return self._name
        
    @name.setter
    def name(self, val):
        
        self._name = val
        
    def initialize(self):
        
        self._df = self.get_df()
        
        return self
    
    def keys(self):
        
        return list(self.df.columns)   
    
    def describe(self):
        
        return self.df.describe()
    
    def count(self):
        
        return self.df.count()
        
    def get(self, *names):
        
        
        if len(names) == 1:
            
            signame, = names
            
            if signame.lower() == 't' or signame.lower() == 'time' or signame.lower() == 'index':
                
                return pd.DataFrame(np.asarray(self.df.index), index=self.df.index)
        
        return self.df[list(names)]
    
    
    def search(self, patt, ignore_case=True, raise_error=False):
        
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
        
        return self.df.memory_usage(deep=True)
    
    def clean_config(self):
        
        if not self.config is None:
        
            new_cfg = dict()
            
            keys = self.df.keys()
            
            for k, v in self.config.items():
                   
                if k in keys:
                    
                    new_cfg[k] = v
                                            
            self.config = new_cfg
        
        return self 
    
    def search_similar(self, name):
        
        # keys = set(list(chain(self.keys(), self.channels)))
        
        keys = self.keys()
        
        ratios = [SequenceMatcher(a=name, b=k).ratio() for k in keys]
        
        _, sorted_keys = zip(*sorted(zip(ratios, keys))[::-1])
        
        return sorted_keys
    
    
    def drop(self, *names, nonexist_ok=True):
        
        keys = []
        
        for name in names:
            
            if name in self.df:
               
               keys.append(name)
               
            elif not nonexist_ok:
                
                raise KeyError(f'"{name}" does not exist in "{self.name}".')
                
        self.df.drop(columns=keys, inplace=True)
        
        return self
    
    def search_get(self, patt, ignore_case=False, raise_error=False):
               
       return self.df[self.search(patt, ignore_case, raise_error)]

    
    def load(self, *keys, mapping=None):
        
        @translate_config(mapping)
        @extract_channels(mapping)
        def get(self, *keys):
                        
            return self.get(*keys)

               
        df = get(self, *keys)
        
        for k, v in df.items():
            
            self.df[k] = v
            
        return df
    

    def reload(self):
                   
       if hasattr(self, '_df'):
           
           del self._df
          
       return self 
   
    def merge(self, obj0):
        
        keys = obj0.df.columns.difference(self.df.columns)
        
        if len(keys):
            
            self._df[keys] = obj0[keys]
            
            
        return self
    
    def squeeze(self, *keys):
        try:
            self._df = self.df[list(keys)]
            
        except KeyError:
            new_keys = list(set(self._df.columns) & set(keys))
            self._df = self.df[new_keys]
            
        return self
    
    @classmethod
    def pipe(self, *funs):
        
        def pipfunc(*inval):
            
            outval = None
            
            for fun in funs:
                
                if outval is None:
                    outval = fun(*inval)
                else:
                    outval = fun(outval)
                
            return outval
    
        return pipfunc
    
    def rename(self, **kwargs):
        
        keymap = kwargs#dict((v, k) for k, v in kwargs.items())
        
        @translate_config(keymap)
        def get(self):
            
            return self._df
        
        get(self)
        
        return self
    
    def resample(self, new_index=None):
        
        """Return a new DataFrame with all columns values interpolated
        to the new_index values."""
        
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

            
            
        new_obj = DATA_OBJECT(path=str(self.path), config=self.config,
                              name=self.name,
                              comment=self.comment, df=df_out)

    
        return new_obj 
    
    def to_numpy(self):
        
        return self.resample()
        
    def to_dict(self):
                
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
        
        if name is None:
            
            name = self.name + '.csv'
            
        if overwrite or not Path(name).is_file():
            
            self.df.to_csv(name)
        
        return self

    def to_excel(self, name=None, overwrite=True):
        
        if name is None:
            
            name = self.name + '.xlsx'
        
        if overwrite or not Path(name).is_file():
            self.df.to_excel(name)
        
        return self
    
    def save(self, name, overwrite=True):
        
        if overwrite or not Path(name).is_file():
        
            dobj = DATA_OBJECT(path=self.path, config=self.config, name=self.name, 
                           comment=self.comment, df=self.df)
            dobj.save(name)
        
        return self
    

        
        
        
    def close(self, clean=True):

        if hasattr(self, '_fhandler') and hasattr(self._fhandler, 'close'):
            
            self._fhandler.close()  
            
            del self._fhandler
            
        if clean and hasattr(self, '_df'):
            
            del self._df

#%% DATA_OBJECT
class DATA_OBJECT(Data_Interface):
    '''
    
    '''
    

    def __init__(self, path=None, config=None, name=None, comment=None, df=None):
        
        if df is None:
            
            self.load(path)
            
        else:
            super().__init__(path, config=config, name=name, comment=comment)
            self._df = df
        
    
    def save(self, name=None):
                
        if name is None:
            
            name = self.name
                    
        np.savez(name, **self.to_dict())
        
        return self
    
    def load(self, path):
        
        with np.load(path, allow_pickle=True) as dat:
            dat = np.load(path, allow_pickle=True)
            df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
            
            
            self.__init__(path=str(dat['path']),
                              config=dat['config'].item(),
                              comment=dat['comment'].item(),
                              name=dat['name'].item(),
                              df=df)
    
            return self
        
#%% PANDAS_OBJECT, FINANCE_OBJECT

class PANDAS_OBJECT(Data_Interface):
    
    dict_fun = {
            '.csv':  pd.read_csv,
            '.xlsx': pd.read_excel,
            }
    
    def __init__(self, path=None, config=None, name=None, comment=None):
        
        super().__init__(path, config=config, name=name, comment=comment)
        
    @property   
    def t(self):
        
        return np.asarray(self.df.index)
    
    @translate_config()
    def get_df(self):
        
        fun = self.__class__.dict_fun[self.path.suffix.lower()]
                
        df = fun(self.path, index_col=0)
        
        return df
    

class FINANCE_OBJECT(PANDAS_OBJECT):
    
    def __init__(self, path=None, config=None, name=None, comment=None, 
                 df=None, time_format='%Y%m%d'):
        
        super().__init__(path, config=config, name=name, comment=comment)
        self.time_format = time_format
    
    @translate_config()
    def get_df(self):
        
        fun = self.__class__.dict_fun[self.path.suffix.lower()]
                
        df = fun(self.path)
                
        cols_dat = df.columns[df.columns.str.contains('date')]
               
        for c in cols_dat:
            
            df[c] = pd.to_datetime(df[c], format=self.time_format)
        
        return df    

    

#%% ASAMMDF_OBJECT
class ASAMMDF_OBJECT(Data_Interface):
       
    def __init__(self, path, config=None, sampling=0.1, name=None, 
                 comment=None, autoclose=False):

        super().__init__(path, config, comment=comment, name=name)
        
        self.sampling = sampling
        self.autoclose = autoclose

    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        
        self.close()
        
        if exc_type:
            
            return False
        
        return True   

    def __len__(self):
        
        return self.t.size

    @property
    def info(self):
        
        if not hasattr(self, '_info'):
            try:
                self._info = self.fhandler.info()
            except AttributeError:
                self._info = None
                
        return self._info
        
    @property
    def comment(self):
        
        if self._comment is None :
            
            info = self.info
            
            if info and 'comment' in info:
                
                xmlobj= fromstring(info['comment'])             
                comment = dict()
                try:
                    tx, = xmlobj.findall('TX')

                    if tx.text:
                        comment = dict(re.findall(r'(.+):\s+(.+)', tx.text))
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
        
        if not hasattr(self, '_df'):
            
            if self.config is None:
                
                self._df = pd.DataFrame()
                
            else:
                
                self._df = self.get_df()
            
        return self._df
    
    @property
    def fhandler(self):
                
        if not hasattr(self, '_fhandler'):
            
            self._fhandler = asammdf.MDF(self.path)
            
        return self._fhandler
        
    @property
    def channels(self):
        
        out = [chn.name for group in self.fhandler.groups for chn in group['channels']]
                        
        return sorted(set(out)) 
    

    @translate_config()
    @extract_channels()
    def get_channels(self, *channels):
        
        df = self.fhandler.to_dataframe(channels=channels, 
                                    raster=self.sampling,
                                    time_from_zero=True)
        
        df.index.name = 'time'
        
        return df 
    

        
    def keys(self):
        
        if not len(self.df):
            
            res = self.channels
            
        else:
            
            res = list(self.df.keys())
        
        return res   
    

    
    def get_df(self, close=None):
        
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
        
        r = re.compile(patt)
        
        return list(filter(r.match, self.channels))
    
#%% MDF_OBJECT 

class MDF_OBJECT(Data_Interface):
       
    def __init__(self, path, config=None, sampling=0.1, name=None, comment=None):

        super().__init__(path, config, comment=comment, name=name)
        
        self.sampling = sampling
        
    @property
    def fhandler(self):
        
        if not hasattr(self, '_fhandler'):
            
            self._fhandler = mdfreader.Mdf(self.path, no_data_loading=True) 
                                   
        return self._fhandler
    
    @property
    def info(self):
        
        return self.fhandler.info
    
    @property
    def comment(self):
        
        if self._comment is None :
            
            cmmt = self.info['HD']['Comment']
                
            if cmmt.get('TX') is not None:
            
                txt = dict(re.findall(r'(.+):\s+(.+)', cmmt.get('TX')))
                
                cmmt.update(txt)
            
            self._comment = cmmt
            
        return self._comment  
    
    @property
    def channels(self):
        
        return sorted(set(self.fhandler.keys()))   

    @property
    def t(self):
        
        if self.config is None:
            
            warnings.warn('Time axis may have deviation due to missing configuration.')
        
        if not len(self.df):
            
            df = self.get_channels(self.channels[0])
            
            t = df.index
            
        else:
            
            t = self.df.index
            
        return np.asarray(t)
    
    def keys(self):
        
        if not len(self.df):
            
            res = self.channels
            
        else:
            
            res = list(self.df.keys())
        
        return res   
    
    def search_channel(self, patt):
        
        r = re.compile(patt)
        
        return list(filter(r.match, self.channels))
    
    @translate_config()
    @extract_channels()
    def get_channels(self, *channels):
               
        def get(chn):
            
            mname = mdfobj.get_channel_master(chn)
            
            if mname is None:
                
                raise ValueError
                                
            dat = mdfobj.get_channel(chn)['data']
            t = mdfobj.get_channel(mname)['data']

            
            return pd.Series(dat, index=t-t.min(), name=chn)        

        mdfobj = mdfreader.Mdf(self.path, channel_list=channels)
        
        
        if self.sampling is not None:            
            mdfobj.resample(self.sampling)    
        
        outs = []
        
        for chn in channels: 
            try:
                res = get(chn)
                outs.append(res)
                
            except ValueError:
                raise
                warnings.warn(f'Channel "{chn}" not found.')  
                
        return pd.concat(outs, axis=1)
    
    def get_df(self):
        

        
        if self.config is None:
            
            df = pd.DataFrame()
        
        else:
            
            df = self.get_channels(*self.config.keys())
                        
        return df   

        
    def get(self, *names):
        
        if all(na in self.df.keys() for na in names):
            
            res = self.df[list(names)]
            
        elif len(names) == 1 and (names[0].lower() == 't' 
                                  or names[0].lower() == 'time' 
                                  or names[0].lower() == 'index'):
            
            if names[0] in self.df.keys():
                
                res = self.df[names]
                
            else:
                
                res = pd.DataFrame(self.t, index=self.t, columns= ['time'])
        
                
        else:
            
            res = self.get_channels(*names)
            
            #self._df = pd.concat([self.df, res], axis=0)
        
                        
        return res      
    
#%% MATLAB_OJBECT

class MATLAB_OBJECT(Data_Interface):
    
    
    def __init__(self, path, config=None, key=None, name=None, comment=None, **kwargs):

        super().__init__(path, config, comment=comment, name=name)
        self.matkey = key
        
    @property
    def fhandler(self):
        
        if not hasattr(self, '_fhandler'):
            
            mat = scipy.io.loadmat(self.path)
            if self.matkey:
                
                self._fhandler = mat[self.matkey]
                
            else:
                for v in mat.values():
                    
                    if isinstance(v, np.ndarray) and v.size > 0:
                        self._fhandler = v
                        break
                else:
                    
                    raise ValueError('Can not find data area')
                    
        return self._fhandler   
        

    
    @property
    def t(self):
        key = None
        if 't' in self.df.columns :
            
            key = 't'
            
        elif 'time' in self.df.columns:
            
            key = 'time'
        
        if key: 
            return self.df[key].to_numpy()
        else:
            return self.df.index.to_numpy()
    
    
    def get_df(self):
        
        dat = dict((k, self.fhandler[k].ravel()[0].ravel()) for 
                   k in self.fhandler.dtype.fields.keys())
        

        df = pd.DataFrame(dat)    
        
            
        return df

#%% AMERES_OJBECT
class AMERES_OBJECT(Data_Interface):

    def __init__(self, path=None, config=None, name=None, comment=None):
        
        super().__init__(path, config=config, name=name, comment=comment)  
        
    @property
    def params(self):
        
        fparam = self.path.parent / (self.path.stem+'.ssf')
        out = dict()
        
        with open(fparam, 'r') as fobj:
            
            lines = fobj.readlines()
        
        for idx, l in enumerate(lines, start=1):
            
            item = dict()
            l = l.strip()
            
            try:
                
                raw, = re.findall(r'Data_Path=\S+', l)
                l = l.replace(raw, '').strip()
                s, = re.findall(r'Data_Path=(.+)', raw)    
                item['Data_Path'] = s
                
                raw, = re.findall(r'Param_Id=\S+', l)
                l = l.replace(raw, '').strip()
                s, = re.findall(r'Param_Id=(.+)', raw)                    
                item['Param_Id'] = s

            except ValueError:
                continue
            
            try:
                
                raw, = re.findall(r'\[\S+\]', l)
                l = l.replace(raw, '').strip()
                s, = re.findall(r'\[(\S+)\]', raw)                    
                item['Unit'] = s   
            except ValueError:
                pass
            

            raw, = re.findall(r'^[01]+\s+\S+\s+\S+\s+\S+', l)
            l = l.replace(raw, '').strip()
            s, = re.findall(r'^[01]+\s+(\S+\s+\S+\s+\S+)', raw)    
            item['Label'] = s
            item['Description'] = l           
            
            out[idx] = item                
            

        return out

    @property
    def channels(self):
        
        return sorted(v['Data_Path'] for v in self.params.values())


    # @translate_config()
    # @extract_channels()
    # def get_channels(self, *channels):
        
    #     df = self.fhandler.to_dataframe(channels=channels, 
    #                                 raster=self.sampling,
    #                                 time_from_zero=True)
        
    #     df.index.name = 'time'
        
    #     return df         
        
#%% Main Loop

if __name__ == '__main__':
    
    obj = AMERES_OBJECT(r'C:\90_Software\57_AMESim'
                        r'\40_Workspace\10_eATS_1p6_v2'
                        r'\eATS_1p6_v2_comod_.results')
    print(obj.channels)
