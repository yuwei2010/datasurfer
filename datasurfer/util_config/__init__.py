import re
import warnings
import pandas as pd

        
from datasurfer import DataInterface
from datasurfer import DataPool
from datasurfer.datautils import show_pool_progress

#%%

class Configurator(object):
    
    def __init__(self, db=None, config=None):
        
        self._cfg = config or dict()
        self.db = db
        
        if (self.db is not None) and (config is None):
            self._cfg = self.db.config or self._cfg
            
        self.init_cfg()
        
    def __repr__(self):
        
        return self.dataframe.__repr__()
    
    def __str__(self):
        
        return self.dataframe.__str__()
    
    def __len__(self):
        
        return self._cfg.__len__()
    
    def __getitem__(self, key):
        
        if key.strip()[0] in '#%$&§':
            
            return self.search_signal(key.strip()[1:])
        else:
            return self.search(key.strip())
    
    def __setitem__(self, key, value):
        if key is Ellipsis:
            
            if isinstance(value, str):
                self.add_keys(value)
                
            elif isinstance(value, (list, tuple, set)):
                self.add_keys(*value)
                
            elif isinstance(value, dict):
                self.add_keys(**value)
                
        elif isinstance(key, str):
            if any(s in key for s in ['*', '?']) or isinstance(key, re.Pattern) or key in self._cfg:
                self.add_keys(key, rename=value)
            else:
                self.add_keys(**{key:value})
    
    def __delitem__(self, key):
        
        self.pop(key)
        
    def __contains__(self, key):
        
        return key in self._cfg
    
    def __hash__(self) -> int:
        
        h = pd.util.hash_pandas_object(self.dataframe, index=True).sum()
        return hash((h, self.__class__))
    
    def __call__(self, config=None, **kwargs):
        if isinstance(config, str):
            
            self.load(config, **kwargs)
        else:
            self._cfg = config or dict()
            self.init_cfg()
        
        return self
    
    @property
    def dataframe(self):
        
        df = pd.DataFrame.from_dict(self._cfg, orient='index')
        
        df.index.name = 'Key'
        df.columns = [f'Signal_{i}' for i in range(len(df.columns))]
        
        df.sort_index(inplace=True)
        
        return df
               
    df = dataframe
    
    @property
    def values(self):
        return self._cfg
    
    @property
    def size(self):
        return self.dataframe.size

    def init_cfg(self):
        """
        Initializes the configuration by converting string values to sets.

        Returns:
            self: The instance of the class.
        """
        for key, value in self._cfg.items():
            if isinstance(value, str):
                self._cfg[key] = set([value])
            elif not isinstance(value, set):
                self._cfg[key] = set(value)
        return self
    
    def clear(self):
        
        self._cfg = dict()
        return self
    
    def search(self, pattern):
        """
        Searches for keys in the configuration dictionary that match the given pattern.

        Parameters:
        - pattern (str): The pattern to search for.

        Returns:
        - DataFrame: A DataFrame containing the keys that match the pattern.
        """
        r = re.compile(pattern)
        return self.dataframe.loc[list(filter(r.match, list(self._cfg.keys())))]
        
    def search_value(self, pattern, return_key=False):
        
        from itertools import chain  
        
        out = dict()

        r = re.compile(pattern)
        
        for key, values in self._cfg.items():
            found = list(filter(r.match, values))
            if found:
                out[key] = found
                
        if return_key:
            return out  
        else:
            return sorted(chain(*out.values()))
    
    
    def search_signal(self, pattern, ignore_case=False, raise_error=False, pbar=True):
        """
        Search for a signal in the database.

        Args:
            pattern (str): The pattern to search for.
            ignore_case (bool, optional): Whether to ignore case when searching. Defaults to False.
            raise_error (bool, optional): Whether to raise an error if the signal is not found. Defaults to False.
            pbar (bool, optional): Whether to display a progress bar during the search. Defaults to True.

        Returns:
            object: The result of the search.

        """
        return self.db.search_signal(pattern, ignore_case=ignore_case, raise_error=raise_error, pbar=pbar)
    
    def add_key(self, key, signal, newkey=None):
        """
        Adds a signal to the specified key in the configuration.

        Parameters:
        - key (str): The key to add the signal to.
        - signal: The signal to add.
        - newkey (str, optional): If provided, renames the key to the newkey.

        Returns:
        - self: The updated configuration object.

        """
        self._cfg.setdefault(key, set()).add(signal)

        if newkey is not None:
            self.rename(key, newkey)

        return self
        
    def add_keys(self, *names, **kwargs):
        """
        Add keys to the configuration.

        Args:
            *names: Variable length argument list of names to be added as keys.
            **kwargs: Variable length keyword argument list of key-value pairs, where the key is the name of the key to be added and the value is the corresponding signal.

        Returns:
            self: The instance of the configuration object.

        """
        rename = kwargs.pop('rename', None)
        
        for name in names:
            
            if any(s in name for s in ['*', '?']) or isinstance(name, re.Pattern):
                self.add_keys(*self.search_signal(name))
                if rename is not None:
                    self.rename(name, rename)
            else:
                self.add_key(name, name, newkey=rename)
        
        for key, signal in kwargs.items():
            
            if isinstance(signal, (list, tuple, set)):
                for s in signal:
                    self.add_key(key, s, newkey=rename)
            else:
                self.add_key(key, signal, newkey=rename) 
                
        return self
                
    def rename(self, pattern, repl):
        """
        Renames keys in the configuration dictionary based on a pattern.

        Args:
            pattern (str or re.Pattern): The pattern to match against the keys.
            repl (str): The replacement string for the matched keys.

        Raises:
            KeyError: If the specified key pattern is not found in the configuration.

        Returns:
            self: The instance of the class with the renamed keys.
        """
        if any(s in pattern for s in ['*', '?']) or isinstance(pattern, re.Pattern):
            r = re.compile(pattern)
            for key in filter(r.match, list(self._cfg.keys())):
                new_key = re.sub(pattern, repl, key)
                for name in self._cfg.pop(key):
                    self.add_key(new_key, name)
        elif pattern in self._cfg:
            for name in self._cfg.pop(pattern):
                self.add_key(repl, name)
        else:
            raise KeyError(f'Key "{pattern}" not found in the configuration.')

        return self
        
    
    def pop(self, key):
        """
        Remove and return the value associated with the given key.
        
        Args:
            key: The key to be removed.
        
        Returns:
            The value associated with the given key.
        
        """
        return self._cfg.pop(key)
    
    def clean(self):
        """
        Clean the configuration by removing keys that do not have any matching signals in the database.

        Returns:
            self: The updated instance of the configuration object.
        """
        assert self.db is not None, 'Database is not set.'

        signals = set(self.db.list_signals())

        for key, values in list(self._cfg.items()):
            found = values.intersection(signals)
            if found:
                self._cfg[key] = found
            else:
                self._cfg.pop(key)
        return self
 
    
    def describe(self, pbar=True):
        """
        Returns a DataFrame that describes the data.

        Args:
            pbar (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            pandas.DataFrame: A DataFrame containing the description of the data.
        """
        if self.dataframe.size == 0:
            return self.dataframe

        if isinstance(self.db, DataInterface):
            df = self.dataframe
            assert len(df.columns) == 1, f'Found {len(df.columns)} signals mapped to same output name, expect only one.'

            df.columns = [self.db.name]

            
        elif isinstance(self.db, DataPool):
            @show_pool_progress('Processing', show=pbar)
            def fun(self):
                for obj in self.objs:
                    yield obj.cfg(config=config).clean().describe()
            config = self._cfg
            dfs = list(fun(self.db))
            df = pd.concat(dfs, axis=1)
        else:
            raise ValueError('Database is not set.')

        return df
    
    def apply2obj(self, pbar=True):
        """
        Applies the configuration settings to the objects in the database.

        Args:
            pbar (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            str: A description of the applied configuration settings.
        """
        assert self.db is not None, 'Database is not set.'

        if isinstance(self.db, DataInterface):

            df = self.clean().describe(pbar=pbar)
            obj = self.db

            obj.config = dict((key, val) for key, val in df[obj.name].items() if isinstance(val, str))

        elif isinstance(self.db, DataPool):

            df = self.clean().describe(pbar=pbar)

            for obj in self.db.objs:

                if obj.name not in df.columns:
                    warnings.warn(f'Object "{obj.name}" not found in the configuration.')
                    continue
                obj.config = dict((key, val) for key, val in df[obj.name].items() if isinstance(val, str))

        return self.describe(pbar=pbar)
            
    def save(self, name):
        """
        Save the configuration to a file.

        Args:
            name (str): The name of the file to save the configuration to.

        Returns:
            self: The instance of the class.

        Raises:
            None

        """
        out = dict((k, sorted(v)) for k, v in self.init_cfg()._cfg.items())

        if name.lower().endswith('.json'):
            import json
            with open(name, 'w') as file:
                json.dump(out, file, indent=4)
        elif name.lower().endswith('.yaml') or name.lower().endswith('.yml'):
            import yaml
            with open(name, 'w') as file:
                yaml.safe_dump(out, file)            

        return self
    
    def append(self, cfg):
        """
        Appends the given configuration dictionary to the existing configuration.

        Args:
            cfg (dict): The configuration dictionary to append.

        Returns:
            self: Returns the updated instance of the configuration object.
        """
        assert isinstance(cfg, dict), 'Value must be a dictionary.'
        
        self.add_keys(**cfg)
        self.init_cfg() 
        return self
        
    
    def load(self, name, append=True):
        """
        Load configuration from a file.

        Args:
            name (str): The name of the file to load the configuration from.
            append (bool, optional): Whether to append the loaded configuration to the existing configuration. 
                                     Defaults to True.

        Returns:
            self: The instance of the Config class.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file extension is not supported.

        """
        if name.lower().endswith('.json'):
            import json
            with open(name, 'r') as file:
                cfg = json.load(file)
                
        elif name.lower().endswith('.yaml') or name.lower().endswith('.yml'):
            import yaml
            with open(name, 'r') as file:
                cfg = yaml.safe_load(file)
                
        if append:
            self.add_keys(**cfg)
        else:
            self._cfg = cfg
                
        self.init_cfg()        
        
        return self
    
    def clear_all(self):
        self._cfg = dict()
        return self
        
        
if __name__ == '__main__':
    
    pass
        
        
# %%
