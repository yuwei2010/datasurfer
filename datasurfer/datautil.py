
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

#%%
def check_config_duplication(cfg):
    
    vals = list()
    for val in cfg.values():
        
        if isinstance(val, str):
            vals.append(val)
        if isinstance(val, abc.Sequence) and not isinstance(val, str):
            vals.extend(list(val))
    
    stat = dict()
    [stat.setdefault(vals.count(s), set([])).add(s) for s in vals]
    
    return stat