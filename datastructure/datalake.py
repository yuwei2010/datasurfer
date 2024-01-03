import os
import re
import warnings

from itertools import chain
from pathlib import Path


import sys

sys.path.insert(0, '.')
from datapool import DataPool

#%%

def collect_dirs(root, *patts, warn_if_double=True, ignore_double=False):
    

    if isinstance(root, (list, tuple, set)):
        root = chain(*root)
    
    found = {}
       
    for r, ds, _ in os.walk(root):
        
        for d in ds:
        
            ismatch = any(re.match(patt, d) for patt in patts)
            
            if ismatch: 
                
                path = (Path(r) / d)
                
                if warn_if_double and d in found:
                    
                    dirs = '\n\t'.join(found[d])
                    
                    warnings.warn(f'"{path}" exists already in:\n{dirs}')
                    
                if (d not in found) or (not ignore_double) :
                    
                    yield path
                    
                found.setdefault(d, []).append(r)


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
        
        pass
        
        
        
        
        