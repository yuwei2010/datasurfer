import re
import pandas as pd

from collections import abc
from itertools import chain
from pathlib import Path
from datasurfer.datapool import DataPool
from difflib import SequenceMatcher

from datasurfer.datautils import collect_dirs, show_pool_progress

#%%
class DataLake(object):
    """
    Represents a data lake that contains multiple data pools.

    Parameters:
    - root: The root directory of the data lake.
    - **kwargs: Additional keyword arguments.
        - pattern: A regular expression pattern used to filter the data pools. Defaults to '.*'.

    Attributes:
    - objs: A list of DataPool objects representing the data pools in the data lake.
    """

    def __init__(self, root=None, **kwargs):
        """
        Initializes a new instance of the DataLake class.

        Parameters:
        - root: The root directory of the data lake.
        - **kwargs: Additional keyword arguments.
            - pattern: A regular expression pattern used to filter the data pools. Defaults to '.*'.
        """
        patts = kwargs.pop('pattern', r'.*')
        config = kwargs.pop('config', None)
        self.name = kwargs.pop('name', None)

        if not isinstance(patts, (list, tuple, set)):
            patts = [patts]
               
        if isinstance(root, (str, Path)):
            founds = sorted(collect_dirs(root, *patts))
            objs = [DataPool([d/f for f in fs], name=d.stem, config=config, **kwargs) for d, fs in founds]

            for obj, (d, _) in zip(objs, founds):
                obj.path = d
            self.path = Path(root)
            if self.name is None:
                self.name = self.path.stem
            
        elif isinstance(root, abc.Sequence) and all(isinstance(r, DataPool) for r in root):
            
            objs = root
            
        elif root is None:
            objs = []
            
        else:
            raise ValueError('root must be a string or a sequence of DataPool objects.')

        self.objs = [obj for obj in objs if len(obj)]
        assert all(self.keys()), 'Data pool names must be unique.'
        
    def __repr__(self):
        
        if self.name:
            return f'<DataLake"{self.name}"@{tuple(self.objs)}>'
        else:
            return f'<DataLake@{tuple(self.objs)}>'
        
              
    def __setitem__(self, name, obj):
        """
        Adds a data pool to the data lake.

        Args:
            name (str): The name of the data pool.
            obj (DataPool): The data pool to be added.
        """
        assert isinstance(obj, DataPool), 'The object must be a DataPool object.'
        obj.name = name
        self.objs.append(obj)    
        
    def __getitem__(self, inval):
        """
        Retrieve an item from the datalake.

        Args:
            inval: The key or keys to retrieve from the datalake. It can be a string, a list, tuple, or set.

        Returns:
            The retrieved item(s) from the datalake.

        Raises:
            KeyError: If the specified key(s) do not exist in the datalake.
        """
        if isinstance(inval, str):
            if '*' in inval:
                if inval.strip()[0] in '#@%':
                    patt = inval.strip()[1:]
                    out =  sorted(set(chain(*[obj.search_signal(patt) for obj in self.objs])))
                else:
                    out = self.__class__([self.get_pool(name) for name in self.search(inval)])
            else:
                out = self.get_pool(inval)
                
        elif isinstance(inval, (list, tuple, set)):
            out = self.__class__([self.get_pool(n) for n in inval])
            
        elif isinstance(inval, slice):
            out = self.__class__(self.objs[inval])
        else:
            out = self.objs[inval]
        return out
    
    def __iter__(self):
        """
        Returns an iterator object that iterates over the elements in the `objs` list.
        """
        return iter(self.objs)
    
    def __len__(self):
        
        return len(self.objs)
    
    def __sub__(self, dlk0):
        
        if isinstance(dlk0, DataPool) and dlk0.name in self.keys():
            return self.__class__(list(set(self.objs).difference([dlk0])))
            
        
        objs = list(set(self.objs).difference(dlk0.objs))
        
        return self.__class__(objs)
    
    def size(self):
        """
        Returns the total size of the objects in the datalake.

        Returns:
            int: The total size of the objects in the datalake.
        """
        return sum(len(obj) for obj in self.objs)
    
    
    def keys(self):
        """
        Returns a list of names of the data pools in the data lake.

        Returns:
        - A list of strings representing the names of the data pools.
        """
        return [obj.name for obj in self.objs]
    
    def items(self):
        """
        Returns a list of all items in the datalake.
        """
        return self.objs

    def paths(self):
        """
        Returns a list of paths to the data pools in the data lake.

        Returns:
        - A list of strings representing the paths to the data pools.
        """
        return [obj.path for obj in self.objs]
    
    def signals(self):
        """
        Returns a sorted set of signals from all the objects in the datalake.

        Returns:
            set: A sorted set of signals.
        """
        return sorted(set(chain(*[obj.list_signals() for obj in self.objs])))

    def get_pool(self, name):
        """
        Retrieves the data pool with the specified name from the data lake.

        Parameters:
        - name: The name of the data pool to retrieve.

        Returns:
        - The DataPool object with the specified name.

        Raises:
        - NameError: If no data pool with the specified name is found.
        """
        for obj in self.objs:
            if obj.name == name:
                return obj
        else:
            raise NameError(f'Can not find any "{name}"')
        
    def get_object(self, name):
        """
        Retrieves an object from the data lake based on its name.

        Parameters:
            name (str): The name of the object to retrieve.

        Returns:
            DataPool: A DataPool object containing the found objects.

        """
        from datasurfer.lib_objects.numpy_object import NumpyObject
        founds = [dp.get_object(name).to_object(cls=NumpyObject, name=dp.name) for dp in self.objs if name in dp.names()]
        
        return DataPool(founds)

        
    def get_signals(self, *signals):
        """
        Retrieves signals from the data objects in the datalake.

        Args:
            *signals: Variable number of signals to retrieve.

        Returns:
            dict: A dictionary containing the retrieved signals, where the keys are the names of the data objects and the values are the corresponding signals.

        """
        out = dict((obj.name, obj.get_signals(*signals)) for obj in self.objs)
        return out
    
    
    def get_signal1Ds(self, *signals, as_df=True):
        """
        Retrieves the signal1Ds for the specified signals.

        Args:
            *signals: Variable number of signals to retrieve.
            as_df (bool): Flag indicating whether to return the result as a DataFrame.

        Returns:
            dict or DataFrame: If `as_df` is True, returns a DataFrame with the signal1Ds.
                               Otherwise, returns a dictionary with the signal1Ds.

        """
        out = dict((obj.name, obj.get_signal1Ds(*signals)) for obj in self.objs)

        if as_df:
            reform = {(outerKey, innerKey): values for outerKey, innerDict 
                      in out.items() for innerKey, values in innerDict.items()}
            out = pd.DataFrame(reform)

        return out
        
    def iterobjs(self):
        """
        Returns an iterator that yields all objects in the datalake.
        """
        return chain(*self.items())
    
    def search(self, patt, ignore_case=True, raise_error=False):
        """
        Searches for data pools in the data lake that match the specified pattern.

        Parameters:
        - patt: The pattern to search for.
        - ignore_case: Whether to ignore case when matching the pattern. Defaults to True.
        - raise_error: Whether to raise an error if no matching data pools are found. Defaults to False.

        Returns:
        - A list of strings representing the names of the matching data pools.
        """
        found = []

        if ignore_case:
            patt = patt.lower()

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
     
    def search_object(self, pattern):
        """
        Search for objects in the datalake that match the given pattern.

        Args:
            pattern (str): The regular expression pattern to match against object names.

        Returns:
            list: A list of objects that match the given pattern.
        """
        return [obj for obj in self.iterobjs() if re.match(pattern, obj.name)]
    
    def search_signal(self, pattern, **kwargs):
        """
        Searches for a signal pattern in the datalake.

        Args:
            pattern (str): The signal pattern to search for.
            **kwargs: Additional keyword arguments to pass to the search method.

        Returns:
            list: A sorted list of unique signals matching the pattern.

        """
        return sorted(set(chain(*[obj.search_signal(pattern, **kwargs) for obj in self.objs])))
        
    
    def find_duplicated(self, as_df=False):
        """
        Finds duplicated items in the data structure.

        Returns a dictionary where the keys are the duplicated items and the values are lists of items that contain the duplicated item.

        Parameters:
        - as_df (bool): If True, the output will be returned as a pandas DataFrame.

        Returns:
        - dict or pandas.DataFrame: If `as_df` is False, returns a dictionary. If `as_df` is True, returns a pandas DataFrame.
        """
        names = [obj.name for obj in self.iterobjs()]

        duplicated = sorted(set([name for name in names if names.count(name) > 1]))

        out = {k: [p[k].meta_info for p in self.items() if k in p.names()] for k in duplicated}

        if as_df:
            out = pd.DataFrame(dict(((k, col), pd.DataFrame(v).loc[col])  for k, v in out.items() for col in pd.DataFrame(v).index))

        return out
    
    def compare_value(self, this, that, **kwargs):
        """
        Compare the values of two DataFrames.

        Args:
            this (DataFrame): The first DataFrame to compare.
            that (DataFrame): The second DataFrame to compare.
            **kwargs: Additional keyword arguments to be passed to the `equals` method.

        Returns:
            bool: True if the DataFrames have the same values, False otherwise.

        Examples:
            >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> compare_value(df1, df2)
            True

            >>> df3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> df4 = pd.DataFrame({'A': [1, 2, 3], 'B': [7, 8, 9]})
            >>> compare_value(df3, df4)
            False
        """
        return this.df.equals(that.df, **kwargs)
    
    def to_datapool(self, name=None, raise_error=False):
        """
        Merge multiple data objects in the pool.

        Args:
            raise_error (bool, optional): Whether to raise an error if there are duplicated data objects. 
                Defaults to False.

        Returns:
            DataObject: The merged data object.

        Raises:
            ValueError: If there are duplicated data objects and `raise_error` is True.
        """
        out = DataPool()
        
        for obj in self.objs:
            try:   
                out = out.merge(obj, raise_error=raise_error)
            except ValueError:
                objname = set(out.names()) & set(obj.names())
                raise ValueError(f'Cannot merge "{obj.name}" with "{out.name}" because of duplicated data object "{objname}".')    
        out.name = name 
        
        out.path = self.path.parent if hasattr(self, 'path') else None

        return out
    
    def pop(self, name):
        """
        Removes and returns the object with the given name from the datalake.

        Args:
            name (str): The name of the object to be removed.

        Returns:
            object: The removed object.

        Raises:
            ValueError: If the object with the given name does not exist in the datalake.
        """
        return self.objs.pop(self.objs.index(self.get_pool(name)))
    
    def save_def(self, name=None):
        """
        Save the definition of the data objects in the datalake to a file.

        Args:
            name (str): The name of the file to save the definition to. If not provided, the definition will not be saved.

        Returns:
            dict: A dictionary containing the definition of the data objects in the datalake.

        Raises:
            None

        Example usage:
            datalake = DataLake()
            datalake.save_def('definition.json')
        """
        out = dict()

        for dp in self.objs:
            out[dp.name] = dict()
            for obj in dp:
                out[dp.name][obj.name] = dict()
                out[dp.name][obj.name]['path'] = str(obj.path)
                if obj.comment:
                    out[dp.name][obj.name]['comment'] = obj.comment

        if name:
            if name.lower().endswith('.json'):
                import json
                with open(name, 'w') as file:
                    json.dump(out, file, indent=4)
            elif name.lower().endswith('.yaml') or name.lower().endswith('.yml'):
                import yaml
                with open(name, 'w') as file:
                    yaml.safe_dump(out, file)
        return out
    
    @staticmethod
    def from_def(path, **kwargs):
        """
        Load data definition from a file or a dictionary and create a DataLake object.

        Args:
            path (str or dict): The path to the file containing the data definition in JSON or YAML format,
                                or a dictionary representing the data definition.
            **kwargs: Additional keyword arguments to be passed to the DataPool.load_def method.

        Returns:
            DataLake: A DataLake object containing the loaded data definition.

        Raises:
            ValueError: If the input value is not a string or a dictionary.

        """
        if isinstance(path, str):
            
            if path.lower().endswith('.json'):
                import json
                with open(path, 'r') as file:
                    data = json.load(file)
                    
            elif path.lower().endswith('.yaml') or path.lower().endswith('.yml'):
                import yaml
                with open(path, 'r') as file:
                    data = yaml.safe_load(file)
                    
        elif isinstance(path, dict):
            data = path
        else:
            raise ValueError('Input value must be a string or a dictionary.')
        
        dlk = DataLake()    
        for name, values in data.items():
            dp = DataPool.from_def(values, name=name, **kwargs)
            dlk.objs.append(dp)
            
        return dlk

    @staticmethod
    def pipeline(*funs, hook=None, pbar=True, ignore_error=True, asiterator=True):
        """
        Applies a series of functions to each object in the data pool and yields the result.

        Args:
            *funs: Variable number of functions to be applied to each object.

        Yields:
            The modified object after applying all the functions.
        """
   
        def wrapper(dlk):
            
            @show_pool_progress('Processing', show=pbar)
            def fun(dlk):
                for dp in dlk.objs:   

                    yield DataPool.pipeline(*funs, hook=hook, pbar=False, ignore_error=ignore_error, asiterator=False)(dp)
                    
            if asiterator:
                return fun(dlk)
            else:
                return list(fun(dlk))
            
        return wrapper   
    
    def initialize(self, use_buffer=True, pbar=True):
        
        buffer = []
        
        for dp in self.objs:
            
            if use_buffer:
                buffer.extend(dp.initialize(buffer=buffer, pbar=pbar).objs)
                
                buffer = list(set(buffer))
            else:
                dp.initialize(pbar=pbar)
            
        return self
    
#%%       
if __name__ == '__main__':

    pass
        
        
        
        
        
# %%
