#%% Import Libraries


import pandas as pd
import numpy as np

import scipy.io
from datasurfer.datainterface import DataInterface




#%% MATLAB_OJBECT

class MATLAB_OBJECT(DataInterface):
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
                    raise ValueError(f'Can not convert "{key}" to DataFrame')

        return df
# %%
