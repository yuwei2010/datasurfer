import pandas as pd
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

    def __init__(self, s, name, comment=None, config=None, **kwargs):
        """
        Initializes a new instance of the STRING_OBJECT class.

        Args:
            path (str): The path to the string object.
            config (dict): Configuration settings for the string object.
            name (str): The name of the string object.
            comment (str): Additional comment for the string object.
        """
        super().__init__(path=None, config=config, name=name, comment=comment)
        
        
        self.string = s
        self.kwargs = kwargs
        

    def get_df(self):

        return str2df(self.string, **self.kwargs)
    
    
    @staticmethod
    def from_other(other):
        assert isinstance(other, DataInterface)
        dat = other.to_dict()
        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])

        obj = STRING_OBJECT(s=df.to_string(),
                    config=dat['config'],
                    comment=dat['comment'],
                    name=dat['name'], index_col=0, delim_whitespace=True)  
        return obj  
    