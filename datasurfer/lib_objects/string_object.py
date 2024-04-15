import io
import pandas as pd
import re
from datasurfer import DataInterface
from datasurfer.datautils import str2df


class STRING_OBJECT(DataInterface):
    """
    Represents a string object.

    Args:
        path (str): The path to the string object.
        config (dict): Configuration settings for the string object.
        name (str): The name of the string object.
        comment (str): Additional comment for the string object.

    Attributes:
        path (str): The path to the string object.
        config (dict): Configuration settings for the string object.
        name (str): The name of the string object.
        comment (str): Additional comment for the string object.
    """
    
    pattern = re.compile(r'([0-9]+)[,.]{1}([0-9]+)')
    exts = ['-']

    def __init__(self, s, name=None, comment=None, config=None, decimal='.', **kwargs):
        """
        Initializes a new instance of the STRING_OBJECT class.

        Args:
            path (str): The path to the string object.
            config (dict): Configuration settings for the string object.
            name (str): The name of the string object.
            comment (str): Additional comment for the string object.
        """
        super().__init__(path=None, config=config, name=name, comment=comment)

        self.kwargs = kwargs
        self.string = s
        self.decimal_delimiter = decimal

    @staticmethod
    def replace_delimiter(string, delimiter):
        """
        Replaces the delimiter in the given string with the specified delimiter.

        Args:
            string (str): The input string.
            delimiter (str): The new delimiter to be used.

        Returns:
            str: The modified string with the new delimiter.

        """
        s = re.sub(STRING_OBJECT.pattern, rf'\1{delimiter}\2', string)
        return s
        

    def get_df(self):
        """
        Converts the string object to a DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame representation of the string object.
        """
        s = STRING_OBJECT.replace_delimiter(self.string, self.decimal_delimiter)            
        return str2df(s, **self.kwargs)
    
    
    def dfstring(self):
        """
        Returns a string representation of the DataFrame.

        Returns:
            str: A string representation of the DataFrame.
        """
        return self.df.to_string()
    
    
    @staticmethod
    def from_other(other):
        """
        Create a STRING_OBJECT instance from another DataInterface object.

        Parameters:
            other (DataInterface): The DataInterface object to create the STRING_OBJECT from.

        Returns:
            STRING_OBJECT: The created STRING_OBJECT instance.

        Raises:
            AssertionError: If the `other` object is not an instance of DataInterface.
        """
        assert isinstance(other, DataInterface)
        dat = other.to_dict()
        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])

        obj = STRING_OBJECT(s=df.to_string(),
                    config=dat['config'],
                    comment=dat['comment'],
                    name=dat['name'], index_col=0, delim_whitespace=True)  
        return obj
    
    
    def to_clipboard(self, decimal='.', **kwargs):
        """
        Copies the DataFrame to the clipboard.

        Args:
            delimiter (str, optional): The delimiter to use when converting the DataFrame to a string. Defaults to '.'.
            **kwargs: Additional keyword arguments to pass to the `to_string` method.

        Returns:
            self: The StringObject instance.

        """

        # index's name causes parse error
        name = self.df.index.name    
        self.df.index.name = None 
                        
        s = self.df.to_string(**kwargs)        
        df = pd.read_csv(io.StringIO(s), delim_whitespace=True)   
        df.index.name = name  
        df.to_clipboard(decimal=decimal)

        return self
    
    def save(self, name=None, decimal='.'):
        """
        Save the string object as a CSV file.

        Args:
            name (str, optional): The name of the CSV file. If not provided, the name will be generated based on the object's name. Defaults to None.
            decimal (str, optional): The decimal delimiter to use in the CSV file. Defaults to '.'.

        Returns:
            self: The string object itself.

        """
        if name is None:
            name = f"{self.name}.csv"
        
        idxname = self.df.index.name    
        self.df.index.name = None 
        s = STRING_OBJECT.replace_delimiter(self.dfstring(), decimal)    
       
        df = str2df(s, index_col=0, delim_whitespace=True)
        df.index.name = idxname
            
        df.to_csv(name)
        
        return self