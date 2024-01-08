import os
import re
import warnings

from itertools import chain
from pathlib import Path


import sys

sys.path.insert(0, '.')
from datapool import DataPool

#%%

def collect_dirs(root, *patts,  patt_filter=None):
    
    patt_filter = patt_filter or [r'^\..*']

    if isinstance(root, (list, tuple, set)):
        root = chain(*root)
    
    found = {}
       
    for r, _, fs in os.walk(root):
        

        d = Path(r).stem
        ismatch = (any(re.match(patt, d) for patt in patts) 
                    and (not any(re.match(patt, d) for patt in patt_filter)))
        
        if ismatch: 
            
            path = Path(r)                
            if fs:
                yield path, fs
                


#%%
class DataLake(object):
    
    
    def __init__(self, root, **kwargs):
        
        patts = kwargs.pop('pattern', r'.*')
        
        if not isinstance(patts, (list, tuple, set)):
            
            patts = [patts]
            
            
        objs = [DataPool([d/f for f in fs], name=d.stem) for d, fs in 
                     sorted(collect_dirs(root, *patts))]
        
        self.objs = [obj for obj in objs if len(obj)]

        
    def keys(self):
        
        return [obj.name for obj in self.objs]

    def search(self, patt, ignore_case=True, raise_error=False):
                
        pass
    
    #%%
if __name__ == '__main__':
        
    dlk = DataLake(r'C:\30_eATS1p6\33_Measurement_Evaluation\70_Alpen_Fahrt')
    
    print(dlk.keys())
    
    pass
        
        
        
        
        
# %%
