"""Top-level package for Data Structure."""

__author__ = """Wei Yu"""
__email__ = 'yuwei2005@gmail.com'

from datasurfer.datapool import Data_Pool
from datasurfer.datalake import Data_Lake
from datasurfer.lib_objects import DataInterface
from datasurfer.lib_objects import list_interfaces


__all__ = ['Data_Pool', 'Data_Lake', 'DataInterface', 'list_interfaces']



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
def read_string(s, name, **kwargs):
    
    from datasurfer.lib_objects.string_object import STRING_OBJECT
    
    return STRING_OBJECT(s, name, **kwargs)

#%%

