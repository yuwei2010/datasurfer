"""Top-level package for Data Structure."""

__author__ = """Wei Yu"""
__email__ = 'yuwei2005@gmail.com'

from datasurfer.datapool import Data_Pool
from datasurfer.datalake import Data_Lake
from datasurfer.lib_objects import DataInterface
from datasurfer.lib_objects import list_interfaces


__all__ = ['Data_Pool', 'Data_Lake', 'DataInterface', 'list_interfaces']

#%%

def interface_pool(*keys):
    """
    Returns a list of interfaces based on the provided keys.
    
    Parameters:
        *keys: Variable number of keys used to filter the interfaces.
        
    Returns:
        A list of interfaces that match the provided keys.
    """
    return list_interfaces()['class'][list(keys)]

#%%
def __getattr__(name):
    
    interfaces = list_interfaces()
    if name in interfaces.index:
        return interfaces['class'][name]
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

#%%
def read_string(s, name, **kwargs):
    
    from datasurfer.lib_objects.string_object import STRING_OBJECT
    
    return STRING_OBJECT(s, name, **kwargs)

#%%

def to_object(df, name, **kwargs):
    
    import pandas as pd
    
    from datasurfer.lib_objects.data_object import DATA_OBJECT
    
    assert isinstance(df, pd.DataFrame), "The input data must be a pandas DataFrame."
    
    return DATA_OBJECT(df, name=name, **kwargs)

#%%

def set_default_interface(ext, path, cls):
    
    import importlib
    import json
    from pathlib import Path 
          
    json_file = Path(__file__).parent / 'map_interface.json'
    
    if json_file.is_file():       
        dict_map = json.load(open(json_file, 'r'))
    else:
        dict_map = {}   
    
    c = getattr(importlib.import_module(f'datasurfer.{path}'), cls)
            
    dict_map[ext] = [path, cls]
    
    json.dump(dict_map, open(json_file, 'w'), indent=4)
    
    return dict_map
    

