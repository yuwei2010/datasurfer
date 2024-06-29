
import numpy as np
import pandas as pd
from pathlib import Path
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
        
        if isinstance(self.path, Path) and self.path.suffix.lower() in ('.xlsx', '.xls'):
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
    
    @classmethod
    def from_other(cls, other):
        """
        Create a new instance of the class from another DataInterface object.

        Parameters:
            other (DataInterface): The DataInterface object to create the new instance from.

        Returns:
            obj (cls): The new instance of the class.

        """
        dat = other.to_dict()

        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
        obj = cls(path=dat['path'],
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
    def __init__(self, path=None, config=None, name=None, comment=None):
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

        
    def date2index(self, col_date='date', drop=False, time_format='%Y%m%d'):
        """
        Convert a column of dates to an index in the DataFrame.

        Args:
            col_date (str, optional): The name of the column containing the dates. Defaults to 'date'.
            drop (bool, optional): Whether to drop the original column after converting it to an index. Defaults to False.
            time_format (str, optional): The format of the dates in the column. Defaults to '%Y%m%d'.

        Returns:
            self: The modified DataFrame object with the date column converted to an index.
        """
        self.df[col_date] = pd.to_datetime(self.df[col_date], format=time_format)
        self.df.sort_values(by=col_date, inplace=True)                
        self.col2index(col_date, drop=drop)
                
        return self
    
