
import numpy as np
import pandas as pd

from datasurfer.datainterface import DataInterface
from datasurfer.datautils import translate_config


#%%
class PandasObject(DataInterface):
    """
    A class representing a Pandas object.

    Attributes:
        dict_fun (dict): A dictionary mapping file extensions to Pandas read functions.
    """

    dict_fun = {
        '.csv':  pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls':  pd.read_excel,        
    }
    
    exts = ['.xlsx', '.csv', '.xls']
    
    def __init__(self, path=None, config=None, name=None, comment=None, **kwargs):
        """
        Initializes a new instance of the PandasObject class.

        Args:
            path (str): The path to the data file.
            config (dict): A dictionary containing configuration options.
            name (str): The name of the object.
            comment (str): A comment or description for the object.
        """
        super().__init__(path, config=config, name=name, comment=comment)
        self.kwargs = kwargs
        
        if self.path and self.path.suffix.lower() in ('.xlsx', '.xls'):
            self.sheet_names = pd.ExcelFile(self.path).sheet_names
            

        
    @property   
    def t(self):
        """
        Returns the index of the DataFrame as a NumPy array.
        """
        return np.asarray(self.df.index)
    
    @translate_config()
    def get_df(self):
        """
        Reads the data file using the appropriate Pandas read function based on the file extension.

        Returns:
            DataFrame: The loaded data as a Pandas DataFrame.
        """
        fun = self.__class__.dict_fun[self.path.suffix.lower()]
       
        kwargs = dict(index_col=0)  
        kwargs.update(self.kwargs)        
        df = fun(self.path, **self.kwargs)
            
        return df
    
    @staticmethod
    def from_other(other):
        
        assert isinstance(other, DataInterface)
        dat = other.to_dict()
        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
        obj = PandasObject(path=dat['path'],
                    config=dat['config'],
                    comment=dat['comment'],
                    name=dat['name'],)
        obj._df = df

        return obj 
    


#%%
class FinanceObject(PandasObject):
    """
    A class representing a finance object.
    
    Attributes:
        path (str): The path to the finance object.
        config (dict): The configuration settings for the finance object.
        name (str): The name of the finance object.
        comment (str): Any additional comments about the finance object.
        df (pandas.DataFrame): The data stored in the finance object.
        time_format (str): The format of the time values in the finance object.
    """
    exts = ['.csv', '.xlsx', '.xls']
    def __init__(self, path=None, config=None, name=None, comment=None, time_format='%Y%m%d'):
        """
        Initializes a new instance of the FinanceObject class.
        
        Args:
            path (str, optional): The path to the finance object. Defaults to None.
            config (dict, optional): The configuration settings for the finance object. Defaults to None.
            name (str, optional): The name of the finance object. Defaults to None.
            comment (str, optional): Any additional comments about the finance object. Defaults to None.
            df (pandas.DataFrame, optional): The data stored in the finance object. Defaults to None.
            time_format (str, optional): The format of the time values in the finance object. Defaults to '%Y%m%d'.
        """
        
        super().__init__(path, config=config, name=name, comment=comment)
        self.time_format = time_format
    
    @translate_config()
    def get_df(self):
        """
        Retrieves the data stored in the finance object.
        
        Returns:
            pandas.DataFrame: The data stored in the finance object.
        """
        
        fun = self.__class__.dict_fun[self.path.suffix.lower()]
                
        df = fun(self.path)
                
        cols_dat = df.columns[df.columns.str.contains('date')]
               
        for c in cols_dat:
            
            df[c] = pd.to_datetime(df[c], format=self.time_format)
        
        return df