import re
import warnings
import pandas as pd

        
from datasurfer import DataInterface
from datasurfer import DataPool

#%%

class Config(object):
    
    def __init__(self, db=None, config=None):
        
        self._cfg = config or dict()
        self.db = db
        
        if self.db is not None and config is None:
            self._cfg = self.db.config or self._cfg
        
        for key, value in self._cfg.items():
            if isinstance(value, str):
                self._cfg[key] = set([value])
            elif not isinstance(value, set):
                self._cfg[key] = set(value)
        
    def __repr__(self):
        
        return self.dataframe.__repr__()
    
    def __str__(self):
        
        return self.dataframe.__str__()
    
    def __getitem__(self, key):
        
        return self.search(key)
    
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
    
    def __call__(self, config=None):
        
        self._cfg = config or dict()
        
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
    def dict(self):
        return self._cfg
    
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
                      
        self._cfg.setdefault(key, set()).add(signal)
        
        if newkey is not None:
            
            self.rename(key, newkey)
            
        return self
        
    def add_keys(self, *names, **kwargs):
        
        rename = kwargs.pop('rename', None)
        
        for name in names:
            
            if any(s in name for s in ['*', '?']) or isinstance(name, re.Pattern):
                self.add_keys(*self.search_signal(name))
                if rename is not None:
                    self.rename(name, rename)
            else:
                self.add_key(name, name, newkey=rename)
        
        for key, signal in kwargs.items():
            
            self.add_key(key, signal, newkey=rename) 
                
        return self
                
    def rename(self, pattern, repl):
            
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
        
        return self._cfg.pop(key)
    
    def clean(self):
        
        assert self.db is not None, 'Database is not set.'
        signals = set(self.db.list_signals())
        
        for key, value in self._cfg.items():
            self._cfg[key] = value.intersection(signals)
              
        return self
 
    
    def describe(self):

        if isinstance(self.db, DataInterface):
            
            df = self.dataframe
            df.columns = [self.db.name]
            
        
        elif isinstance(self.db, DataPool):
            
            dfs = [obj.cfg(config=self._cfg).clean().describe() for obj in self.db.objs]
            
            df = pd.concat(dfs, axis=1)
            
        else:
            raise ValueError('Database is not set.')
            
        return df
    
    def apply2obj(self):
        
        assert self.db is not None, 'Database is not set.'
        
        if isinstance(self.db, DataInterface):
            
            df = self.clean().describe()
            obj = self.db
            
            obj.config = dict((key, val) for key, val in df[obj.name].items() if isinstance(val, str))
            
        elif isinstance(self.db, DataPool):
            
            df = self.clean().describe()
            
            for obj in self.db.objs:
                
                obj.config = dict((key, val) for key, val in df[obj.name].items() if isinstance(val, str))
            
            
        return self
            
        
        
        
        
        
        
    
        
        
        
if __name__ == '__main__':
    
    pass
        
        
# %%
