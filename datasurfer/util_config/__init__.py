import warnings

#%%

class Config(object):
    
    def __init__(self, db=None, config=None):
        
        self._cfg = config or dict()
        self.db = db
        
    def search_signal(self, pattern, ignore_case=False, raise_error=False, pbar=True):
        
        return self.db.search_signal(pattern, ignore_case=ignore_case, raise_error=raise_error, pbar=pbar)
    
    def add_key(self, name, key):
        
        if name in self._cfg:
            warnings.warn(f"Key {name} already exists and will be replaced.")     
              
        self._cfg[name] = key 
        
    def add_keys(self, *keys, **kwargs):
        
        for key in keys:
            
            if any(s in key for s in ['*', '?']):
                self.add_keys(*self.search_signal(key))
            else:
                self.add_key(key, key)
        
        
        
        
        
    
        
        
        
if __name__ == '__main__':
    
    pass
        
        