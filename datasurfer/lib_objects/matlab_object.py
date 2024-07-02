#%% Import Libraries
import pandas as pd
import numpy as np
import warnings
import scipy.io
import h5py
from datasurfer.datainterface import DataInterface

#%% MATLAB_OJBECT

class MatlabObject(DataInterface):
    """
    Represents a MATLAB object that can be loaded from a file.

    Args:
        path (str): The path to the MATLAB file.
        config (dict, optional): Configuration parameters for the object. Defaults to None.
        key (str, optional): The key to access the desired data in the MATLAB file. Defaults to None.
        name (str, optional): The name of the object. Defaults to None.
        comment (str, optional): Additional comment for the object. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        matkey (str): The key to access the desired data in the MATLAB file.

    Properties:
        fhandler (numpy.ndarray): The loaded data from the MATLAB file.
        t (numpy.ndarray): The time values associated with the data.

    Methods:
        get_df(): Returns a pandas DataFrame containing the loaded data.

    """
    exts = ['.mat']
    def __init__(self, path, config=None, key=None, name=None, comment=None, **kwargs):
        super().__init__(path, config, comment=comment, name=name)
        self.matkey = key

    @property
    def fhandler(self):
        """
        The loaded data from the MATLAB file.

        Returns:
            numpy.ndarray: The loaded data.

        Raises:
            ValueError: If the data area cannot be found in the MATLAB file.

        """

        if not hasattr(self, '_fhandler'):
            mat = scipy.io.loadmat(self.path)
            if self.matkey:
                self._fhandler = mat[self.matkey]
            else:
                for v in mat.values():
                    if isinstance(v, np.ndarray) and v.size > 0:
                        self._fhandler = v
                        break
                else:
                    raise ValueError('Can not find data area')
        return self._fhandler

    @property
    def t(self):
        """
        The time values associated with the data.

        Returns:
            numpy.ndarray: The time values.

        """
        key = None
        if 't' in self.df.columns:
            key = 't'
        elif 'time' in self.df.columns:
            key = 'time'
        if key:
            return self.df[key].to_numpy()
        else:
            return self.df.index.to_numpy()

    def get_df(self):
        """
        Returns a pandas DataFrame containing the loaded data.

        Returns:
            pandas.DataFrame: The DataFrame containing the loaded data.

        """        
        dat = dict((k, self.fhandler[k].ravel()[0].ravel()) for k in self.fhandler.dtype.fields.keys())
        df = pd.DataFrame()

        for key, value in dat.items():
            try:
                df[key] = value
            except ValueError:
                if len(value) == 1:
                    df[key] = value[0]
                else:                   
                    warnings.warn(f'Can not convert "{key}" to DataFrame')

        return df
    
    def save(self, key, path=None, **kwargs):
        
        path = None or str(self.name) + '.mat'
        
        struct = self.df.to_dict('list')
        
        for k, v in struct.items():
            struct[k] = np.array(v).reshape(-1, 1)
            
        struct.update(kwargs)
        
        scipy.io.savemat(path, {key: struct})
        
        return self
        
        
    
# %%

class H5pyObject(DataInterface):
    """
    Represents an H5py object that provides access to data stored in an HDF5 file.
    
    Args:
        path (str): The path to the HDF5 file.
        root (str): The root path within the HDF5 file where the data is located.
        config (dict, optional): Configuration settings for the object. Defaults to None.
        name (str, optional): The name of the object. Defaults to None.
        comment (str, optional): Any additional comments about the object. Defaults to None.
    """
    
    def __init__(self, path, root, config=None,  name=None, comment=None):
        super().__init__(path, config=config, comment=comment, name=name)
        self.root = root
        
    @property
    def fhandler(self):
        """
        Property that returns the HDF5 file handler for the specified root path.
        
        Returns:
            h5py.Group: The HDF5 file handler for the specified root path.
        """
        if not hasattr(self, '_fhandler'):
            rts = self.root.split('.')
            fobj = h5py.File(self.path, 'r') 
            root = fobj[rts[0]]
            for r in rts[1:]:  
                root = root[r]
            self._fhandler = root
        return self._fhandler   
    
    def get_df(self):
        """
        Retrieves the data from the HDF5 file as a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: The data from the HDF5 file as a DataFrame.
        """
        df = pd.DataFrame.from_dict(
            {k: np.array(v).ravel() for k, v in self.fhandler.items()})
        return df
        
        