import pandas as pd
import numpy as np
from datasurfer import DataInterface
from pathlib import Path


#%% DATA_OBJECT
class DATA_OBJECT(DataInterface):
    """
    Represents a data object.

    Args:
        path (str): The path to the data file.
        config (dict): Configuration settings for the data object.
        name (str): The name of the data object.
        comment (str): Additional comment for the data object.
        df (pandas.DataFrame): The data as a pandas DataFrame.

    Attributes:
        t (numpy.ndarray): The index of the DataFrame as a numpy array.

    Methods:
        save(name=None): Save the data object to a file.
        load(path): Load the data object from a file.

    """

    def __init__(self, path=None, config=None, name=None, comment=None, df=None):
        """
        Initializes a new instance of the DATA_OBJECT class.

        If `df` is None, the data object will be loaded from the specified `path`.
        Otherwise, the data object will be created with the provided arguments.

        Args:
            path (str): The path to the data file.
            config (dict): Configuration settings for the data object.
            name (str): The name of the data object.
            comment (str): Additional comment for the data object.
            df (pandas.DataFrame): The data as a pandas DataFrame.

        """
        if df is None:
            if isinstance(path, pd.DataFrame):
                super().__init__(None, config=config, name=name, comment=comment)
                self._df = path
            elif path is not None:   
                self.load(path)
        else:
            super().__init__(path, config=config, name=name, comment=comment)
            self._df = df
   
    def get_df(self):
        
        raise NotImplementedError("A Data object doesn't need get_df.")
    
    def save(self, name=None):
        """
        Save the data object to a file.

        If `name` is not provided, the data object will be saved with its current name.

        Args:
            name (str, optional): The name of the saved file. Defaults to None.

        Returns:
            DATA_OBJECT: The current instance of the data object.

        """
        if name is None:
            name = self.name
        np.savez(name, **self.to_dict())
        return self
    
    @staticmethod
    def from_other(other):
        
        assert isinstance(other, DataInterface)
        dat = other.to_dict()
        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
        obj = DATA_OBJECT(path=dat['path'],
                    config=dat['config'],
                    comment=dat['comment'],
                    name=dat['name'],
                    df=df)  
        return obj  
    @staticmethod
    def load(path):
        """
        Load the data object from a file.

        Args:
            path (str): The path to the data file.

        Returns:
            DATA_OBJECT: The current instance of the data object.

        """
        with np.load(path, allow_pickle=True) as dat:
            dat = np.load(path, allow_pickle=True)
            df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
            obj = DATA_OBJECT(path=Path(path).absolute(),
                          config=dat['config'].item(),
                          comment=dat['comment'].item(),
                          name=dat['name'].item(),
                          df=df)
        return obj
    
# %%
