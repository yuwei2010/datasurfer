"""Top-level package for Data Structure."""

__author__ = """Wei Yu"""
__email__ = 'yuwei2005@gmail.com'

from datasurfer.datapool import DataPool
from datasurfer.datalake import DataLake
from datasurfer.datainterface import DataInterface
from datasurfer.datainterface import list_interfaces
from datasurfer.lib_poolobjects.financepool_object import StockPool


__all__ = ['DataPool', 'DataLake', 'DataInterface', 'list_interfaces', 'FinancePool']

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
    from datasurfer.lib_objects.string_object import StringObject
    
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
    from datasurfer.lib_objects.parquet_object import ParquetObject

    assert isinstance(df, pd.DataFrame), "The input data must be a pandas DataFrame."

    return ParquetObject(df, name=name, **kwargs)

#%%
def read_multisheets_excel(input, **kwargs):
    """
    Reads data from multiple sheets in an Excel file and returns an ExcelDataPool object.

    Parameters:
    - input: The input data source. It can be a DataInterface object, a DataPool object, or a file path.
    - **kwargs: Additional keyword arguments to customize the reading process.

    Returns:
    - An ExcelDataPool object containing the data from the input source.

    Example usage:
    ```
    data = read_multisheets_excel('data.xlsx', sheet_names=['Sheet1', 'Sheet2'], header_row=1)
    ```

    """
    from datasurfer.lib_poolobjects.excelmultisheet_object import ExcelDataPool
    
    if isinstance(input, DataInterface):
        
        return ExcelDataPool(input.path, **kwargs)
    
    elif isinstance(input, DataPool):
        
        return DataLake([ExcelDataPool(path, **kwargs) for path in input.paths()])
    
    else:
        
        return ExcelDataPool(input, **kwargs)
    
    

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

    getattr(importlib.import_module(f'datasurfer.{path}'), cls)

    dict_map[ext] = [path, cls]

    json.dump(dict_map, open(json_file, 'w'), indent=4)

    return dict_map

#%%

def ProcessPool(n_workers=4, threads_per_worker=1, memory_limit='8GB'):
    """
    Initializes a new instance of the ProcessPool class.

    Args:
        n_workers (int, optional): The number of workers to use. Defaults to 4.
        threads_per_worker (int, optional): The number of threads per worker. Defaults to 1.
        memory_limit (str, optional): The memory limit per worker. Defaults to '8GB'.

    Returns:
        ProcessPool: The created ProcessPool object.
    """
    from datasurfer.util_multiproc import MultiProc

    return MultiProc(db=None, n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)

#%%
def ignore_error(fun, *args, **kwargs):
    """
    Calls a function and ignores any errors that occur.

    Parameters:
    - fun: The function to call.
    - *args: Positional arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.

    Returns:
    - The result of the function call, or None if an error occurs.
    """
    import traceback
    import warnings
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as err:

            errname = err.__class__.__name__
            tb = traceback.format_exc(limit=0, chain=False)
            warnings.warn(f'Exception "{errname}" is raised while processing "{fun.__name__}": "{tb}"')
            
    return wrapper
#%%
