import os
import re
import warnings

from itertools import chain
from pathlib import Path


import sys

sys.path.insert(0, '.')
from .datapool import DataPool
from difflib import SequenceMatcher


#%%
def collect_dirs(root, *patts, patt_filter=None):
    """
    Collects directories and their corresponding files under the given root directory.

    Args:
        root (str or list or tuple or set): The root directory or a collection of root directories.
        *patts (str): Patterns to match against directory names.
        patt_filter (list or None): Patterns to filter out directory names. Defaults to None.

    Yields:
        tuple: A tuple containing the path of the directory and a list of files in that directory.

    Examples:
        >>> for path, files in collect_dirs('/path/to/root', 'dir*', patt_filter=['dir2']):
        ...     print(f"Directory: {path}")
        ...     print(f"Files: {files}")
        ...
        Directory: /path/to/root/dir1
        Files: ['file1.txt', 'file2.txt']
        Directory: /path/to/root/dir3
        Files: ['file3.txt', 'file4.txt']
    """

    patt_filter = patt_filter or [r'^\..*']

    if isinstance(root, (list, tuple, set)):
        root = chain(*root)

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
    """
    Represents a data lake that contains multiple data pools.

    Parameters:
    - root: The root directory of the data lake.
    - **kwargs: Additional keyword arguments.
        - pattern: A regular expression pattern used to filter the data pools. Defaults to '.*'.

    Attributes:
    - objs: A list of DataPool objects representing the data pools in the data lake.
    """

    def __init__(self, root, **kwargs):
        """
        Initializes a new instance of the DataLake class.

        Parameters:
        - root: The root directory of the data lake.
        - **kwargs: Additional keyword arguments.
            - pattern: A regular expression pattern used to filter the data pools. Defaults to '.*'.
        """
        patts = kwargs.pop('pattern', r'.*')
        config = kwargs.pop('config', None)

        if not isinstance(patts, (list, tuple, set)):
            patts = [patts]

        founds = sorted(collect_dirs(root, *patts))
        objs = [DataPool([d/f for f in fs], name=d.stem, config=config) for d, fs in founds]

        for obj, (d, _) in zip(objs, founds):
            obj.path = d

        self.objs = [obj for obj in objs if len(obj)]
        
        
    def __getitem__(self, inval):

        if isinstance(inval, str):
            
            if '*' in inval:
                
                if inval.strip()[0] in '#@%':
                    
                    patt = inval.strip()[1:]
                    
                    out = self.search(patt)
                else:
                
                    out = [self.get_pool(name) for name in self.search(inval)]
            else:

                out = self.get_pool(inval)

            
        elif isinstance(inval, (list, tuple, set)):
            

            out = [self.__getitem__(n) for n in inval]
                

            
        else:
            out = self.objs[inval]
            
        return out
    
    
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
        
    def iterobjs(self):
        
        return chain(*self.items())
    
    def search_object(self, pattern):
        
        return [obj for obj in self.iterobjs() if re.match(pattern, obj.name)] 
    
    
    def find_duplicated(self):
        """
        Finds duplicated items in the data structure.

        Returns a dictionary where the keys are the duplicated items and the values are lists of items that contain the duplicated item.
        """
        names = [obj.name for obj in self.iterobjs()]
        duplicated = sorted(set([name for name in names if names.count(name) > 1]))
        return {k: [p for p in self.items() if k in p] for k in duplicated}
    
    def compare_value(self, this, that, **kwargs):
        
        return this.df.equals(that.df, **kwargs)
    
    def merge_pools(self, raise_error=False):
        
        out = self.objs[0]
        
        for obj in self.objs[1:]:
            try:   
                out = out.merge(obj, raise_error=raise_error)
            except ValueError:
                
                objname = set(out.names()) & set(obj.names())
                raise ValueError(f'Cannot merge "{obj.name}" with "{out.name}" because of data object "{objname}".')    
            
        return out
        
        

        

    

        
#%%       
if __name__ == '__main__':

    pass
        
        
        
        
        
# %%
