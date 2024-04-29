import pandas as pd
from datasurfer.datautils import translate_config
from datasurfer.lib_objects.pandas_object import PandasObject

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
    
    @staticmethod
    def from_other(other):
        
        assert isinstance(other, PandasObject)
        
        obj = FinanceObject(path=other.path,
                    config=other.config,
                    comment=other.comment,
                    name=other.name)

  
        return obj  