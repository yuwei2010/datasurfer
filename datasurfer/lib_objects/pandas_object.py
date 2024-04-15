
import numpy as np
import pandas as pd
from datasurfer.lib_objects import DataInterface
from datasurfer.datautils import translate_config


#%%
class PANDAS_OBJECT(DataInterface):
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
        Initializes a new instance of the PANDAS_OBJECT class.

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