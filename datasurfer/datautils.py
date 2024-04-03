import os
import re
import warnings
import json
import pandas as pd
import numpy as np

from functools import wraps
from pathlib import Path
from itertools import chain
from collections import abc
from tqdm import tqdm


#%%
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

            flag_pbar = (hasattr(self, 'silent') and not self.silent) and (not self.initialized or show)
            
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
                        if self.name is None:
                            pbar.set_description(f'{msg} "{bcolors.OKBLUE}{self.objs[idx].name}{bcolors.ENDC}"')
                        else:
                            pbar.set_description(f'{msg} "{bcolors.OKGREEN}{self.name}{bcolors.ENDC}/{bcolors.OKBLUE}{self.objs[idx].name}{bcolors.ENDC}"')
                        
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

#%%
def check_config_duplication(cfg):
    """
    Check for duplication in the configuration dictionary.

    Args:
        cfg (dict): The configuration dictionary to check.

    Returns:
        dict: A dictionary containing the count of each duplicated value and the set of values that are duplicated.
    """
    vals = list()
    for val in cfg.values():
        if isinstance(val, str):
            vals.append(val)
        if isinstance(val, abc.Sequence) and not isinstance(val, str):
            vals.extend(list(val))

    stat = dict()
    [stat.setdefault(vals.count(s), set([])).add(s) for s in vals]

    return stat

#%% Extract channels

def extract_channels(newconfig=None):
    """
    A decorator that extracts channels from the given configuration and passes them as arguments to the decorated function.
    
    Args:
        newconfig (dict, optional): A dictionary containing the mapping of channel names. Defaults to None.
    
    Returns:
        function: The decorated function.
    """
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
    """
    A decorator function that translates column names in the output of a decorated function based on a configuration dictionary.
    
    Args:
        newconfig (dict, optional): A dictionary that maps the original column names to the desired translated column names. 
            If not provided, the decorator will use the `config` attribute of the decorated object.
    
    Returns:
        function: The decorated function.
    """
    
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

#%%
def parse_config(config):
    """
    Parses the given configuration and returns a processed config object.

    Args:
        config: The configuration to be parsed. It can be a path to a JSON or YAML file,
                a list/tuple/set of strings, a list/tuple/set of dictionaries, or a dictionary.

    Returns:
        The processed config object.

    Raises:
        IOError: If the config format is unknown and not in JSON or YAML format.
        TypeError: If the config type is not supported.

    """
    if isinstance(config, (str, Path)):
        if str(config).lower().endswith('.json'):
            config = json.load(open(config))
        elif str(config).lower().endswith('.yaml') or str(config).lower().endswith('.yml'):
            import yaml
            config = yaml.safe_load(open(config))
        else:
            raise IOError('Unknown config format, expect json or yaml.')
    elif isinstance(config, (list, tuple, set)):
        if all(isinstance(s, str) for s in config):
            config = dict((v, v) for v in config)
        elif all(isinstance(s, dict) for s in config):
            from datasurfer.datapool import combine_configs
            config = combine_configs(*list(config))
        else:
            raise TypeError('Can not handle config type.')
    elif (not isinstance(config, dict)) and (config is not None):

        raise TypeError('Unknown config format, expect dict')
    
    return config

#%%
def collect_dirs(root, *patts, patt_filter=None):
    """
    Collects directories and their corresponding files under the given root directory.

    Args:
        root (str or list or tuple or set): The root directory or a collection of root directories.
        *patts (str): Patterns to match against directory names.
        patt_filter (list or None): Patterns to filter out directory names. Defaults to None.

    Yields:
        tuple: A tuple containing the path of the directory and a list of files in that directory.

    Examples:
        >>> for path, files in collect_dirs('/path/to/root', 'dir*', patt_filter=['dir2']):
        ...     print(f"Directory: {path}")
        ...     print(f"Files: {files}")
        ...
        Directory: /path/to/root/dir1
        Files: ['file1.txt', 'file2.txt']
        Directory: /path/to/root/dir3
        Files: ['file3.txt', 'file4.txt']
    """

    patt_filter = patt_filter or [r'^\..*']

    if isinstance(root, (list, tuple, set)):
        root = chain(*root)

    for r, _, fs in os.walk(root):
        d = Path(r).stem
        ismatch = (any(re.match(patt, d) for patt in patts) 
                    and (not any(re.match(patt, d) for patt in patt_filter)))

        if ismatch:
            path = Path(r)
            if fs:
                yield path, fs
                
#%%
def arghisto(data, bins):
    """
    Compute the histogram of the input data based on the given bins.

    Parameters:
    data (ndarray): Input data array.
    bins (ndarray): Bins for computing the histogram.

    Returns:
    list: List of arrays containing the indices of data points falling into each bin.
    """
    out = []
    dat = data.ravel()
       
    for idx in range(0, len(bins)-1):
        if idx == 0:
            out.append(np.where((bins[idx]<=dat) & (bins[idx+1]>=dat))[0])
        else:
            out.append(np.where((bins[idx]<dat) & (bins[idx+1]>=dat))[0])
        
    return out

#%%
def parse_data(func):   
    """
    A decorator function that parses the input data before passing it to the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    Raises:
        ValueError: If the keys are not strings or numpy arrays.

    """
    @wraps(func)
    def wrapper(self, *keys, **kwargs):
        
        def get(keys):
            out = []
            lbls = []
            for key in keys:
                if isinstance(key, str):
                    out.append(self.dp[[key]].dropna().to_numpy().ravel())
                    lbls.append(key)
                elif isinstance(key, pd.Series):
                    out.append(key.dropna().to_numpy())
                    lbls.append(key.name)
                elif isinstance(key, pd.DataFrame):
                    out.extend(key.dropna().to_numpy().T)
                    lbls.extend(key.columns)
                elif isinstance(key, np.ndarray):
                    out.append(key)
                    lbls.append(None)
                elif isinstance(key, abc.Sequence):
                    o, ls = get(key)
                    out.append(o)
                    lbls.extend(ls)
                else:
                    raise ValueError('keys must be strings or numpy arrays')
                
            return out, lbls
        
        if all(isinstance(key, str) for key in keys):
            out = self.dp[keys].dropna().to_numpy().T    
            lbls = keys
        else:        
            out, lbls = get(keys)
        
        if ('labels' not in kwargs) and all(lbl is not None for lbl in lbls) :
            kwargs['labels'] = lbls     

        return func(self, *out, **kwargs)
    
    return wrapper

#%%

class bcolors:
    """
    A class that defines ANSI escape codes for text colors and styles.
    
    Attributes:
        HEADER (str): ANSI escape code for header color.
        OKBLUE (str): ANSI escape code for blue color.
        OKCYAN (str): ANSI escape code for cyan color.
        OKGREEN (str): ANSI escape code for green color.
        WARNING (str): ANSI escape code for warning color.
        FAIL (str): ANSI escape code for fail color.
        ENDC (str): ANSI escape code to reset text color and style.
        BOLD (str): ANSI escape code for bold text.
        UNDERLINE (str): ANSI escape code for underlined text.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
