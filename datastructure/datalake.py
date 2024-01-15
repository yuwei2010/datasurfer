import os
import re
import warnings

from itertools import chain
from pathlib import Path


import sys

sys.path.insert(0, '.')
from datapool import DataPool
from difflib import SequenceMatcher


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
            
        founds = sorted(collect_dirs(root, *patts))
        objs = [DataPool([d/f for f in fs], name=d.stem) for d, fs in founds]
        
        for obj, (d, _) in zip(objs, founds):
            
            obj.path = d
        
        self.objs = [obj for obj in objs if len(obj)]

        
    def keys(self):
        
        return [obj.name for obj in self.objs]
    
    def paths(self):
        
        return [obj.path for obj in self.objs]

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
    
    def get_pool(self, name):
        
        for obj in self.objs:
            
            if obj.name == name:
                return obj
        else:
            raise NameError(f'Can not find any "{name}"')
    
    #%%
if __name__ == '__main__':
        
    dk = DataLake(r'C:\30_eATS1p6\33_Measurement_Evaluation\70_Alpen_Fahrt')
    
    dp = dk.get_pool('2023 09 26 - 174-67 - Alpen-Fahrt Vormittag')
    
    print(dk.paths())
    
    pass
        
        
        
        
        
# %%
