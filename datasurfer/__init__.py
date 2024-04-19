"""Top-level package for Data Structure."""

__author__ = """Wei Yu"""
__email__ = 'yuwei2005@gmail.com'

from datasurfer.datapool import DataPool
from datasurfer.datalake import DataLake
from datasurfer.lib_objects import DataInterface
from datasurfer.lib_objects import list_interfaces


__all__ = ['DataPool', 'DataLake', 'DataInterface', 'list_interfaces']

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
    """
    Reads a string and returns a StringObject.

    Args:
        s (str): The input string.
        name (str): The name of the string object.
        **kwargs: Additional keyword arguments.

    Returns:
        StringObject: The created StringObject.

    """
    from datasurfer.lib_objects.StringObject import StringObject
    
    return StringObject(s, name, **kwargs)

#%%

def df2object(df, name, **kwargs):
    """
    Convert a pandas DataFrame to a NumpyObject.

    Parameters:
        df (pd.DataFrame): The input data as a pandas DataFrame.
        name (str): The name of the data object.
        **kwargs: Additional keyword arguments to be passed to the NumpyObject constructor.

    Returns:
        NumpyObject: The converted data object.

    Raises:
        AssertionError: If the input data is not a pandas DataFrame.
    """
    import pandas as pd
    from datasurfer.lib_objects.NumpyObject import NumpyObject

    assert isinstance(df, pd.DataFrame), "The input data must be a pandas DataFrame."

    return NumpyObject(df, name=name, **kwargs)

#%%

def set_default_interface(ext, path, cls):
    """
    Sets the default interface for a given file extension.

    Parameters:
    - ext (str): The file extension.
    - path (str): The path to the module containing the interface class.
    - cls (str): The name of the interface class.

    Returns:
    - dict: The updated dictionary mapping file extensions to interface information.
    """
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
    

