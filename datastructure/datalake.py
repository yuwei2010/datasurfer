import os
import warnings

from itertools import chain
from pathlib import Path

#%%

def collect_files(root, *patts, warn_if_double=True, ignore_double=False):
    

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


class DataLake(object):
    
    
    def __init__(self, patt):
        

        pass
        
        
    def keys(self):
        
        zcount = len(str(len(self.objs))) + 1
        
        strfmt = f':0{zcount}'        
        
        fmt = 'DataPool_' + '{' + strfmt + '}'
                
        def get():
            
            for idx, dp in enumerate(self.objs):
                
                if dp.name is None:
                    
                    yield fmt.format(idx)
                else:
                    
                    yield dp.name
                
        return list(get())

    def search(self, patt, ignore_case=True, raise_error=False):
                
        pass
    
    #%%
    if __name__ == '__main__':
        
        import sys
        
        sys.path.insert(0, '.')
        
        from datapool import DataPool
        
        
        