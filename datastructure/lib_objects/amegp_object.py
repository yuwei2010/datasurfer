#%% Import Libraries

import re
import pandas as pd
from pathlib import Path
from .data_interface import Data_Interface


#%% AMEGP_OJBECT
class AMEGP_OBJECT(Data_Interface):
    """
    Represents an object for handling AMEGP data.

    Args:
        path (str): The path to the data file.
        config (dict): Configuration settings for the object.
        name (str): The name of the object.
        comment (str): Additional comment for the object.

    Attributes:
        path (str): The path to the data file.
        config (dict): Configuration settings for the object.
        name (str): The name of the object.
        comment (str): Additional comment for the object.
        df (pd.DataFrame): The data stored in a pandas DataFrame.

    Methods:
        __init__(self, path=None, config=None, name=None, comment=None):
            Initializes a new instance of the AMEGP_OBJECT class.
        __setitem__(self, name, value):
            Sets the value of a specific item in the object.
        get_df(self):
            Retrieves the data as a transposed DataFrame.
        set_value(self, name, value):
            Sets the value of a specific item in the DataFrame.
        save(self, name=None):
            Saves the object's data to a file.

    """

    def __init__(self, path=None, config=None, name=None, comment=None):
        """
        Initializes a new instance of the AMEGP_OBJECT class.

        Args:
            path (str): The path to the data file.
            config (dict): Configuration settings for the object.
            name (str): The name of the object.
            comment (str): Additional comment for the object.

        """
        if name is None:
            name = Path(path).stem[:-1]
        
        super().__init__(path, config=config, name=name, comment=comment)  
    
    def __setitem__(self, name, value):
        """
        Sets the value of a specific item in the object.

        Args:
            name (str): The name of the item.
            value: The value to be set.

        """
        self.set_value(name, value)
        
    def get_df(self):
        """
        Retrieves the data as a transposed DataFrame.

        Returns:
            pd.DataFrame: The data stored in a pandas DataFrame.

        """
        df = pd.read_xml(self.path)
        df.set_index('VARNAME', inplace=True)
        return df.transpose()
    
    def set_value(self, name, value):
        """
        Sets the value of a specific item in the DataFrame.

        Args:
            name (str): The name of the item.
            value: The value to be set.

        Returns:
            self: The updated AMEGP_OBJECT instance.

        """
        self.df.at['VALUE', name] = value
        return self
    
    def save(self, name=None):
        """
        Saves the object's data to a file.

        Args:
            name (str, optional): The name of the file to save the data to. If not provided, the original file will be overwritten.

        Returns:
            self: The AMEGP_OBJECT instance.

        """
        name = name if name is not None else self.path
        
        with open(self.path, 'r') as fobj:
            lines = fobj.readlines()
        
        newlines = []
        
        for l in lines:
            if re.match(r'^<VARNAME>.+</VARNAME>$', l.strip()):
                varname, = re.findall(r'^<VARNAME>(.+)</VARNAME>$', l.strip())
                item = self.df[varname]
                
            if re.match(r'^<VALUE>.+</VALUE>$', l.strip()):  
                str_old, = re.findall(r'^<VALUE>.+</VALUE>$', l.strip())
                str_new = '<VALUE>{}</VALUE>'.format(item['VALUE'])
                l = l.replace(str_old, str_new, 1)
                    
            newlines.append(l)
            
        with open(name, 'w') as fobj:
            fobj.writelines(newlines)
            
        return self