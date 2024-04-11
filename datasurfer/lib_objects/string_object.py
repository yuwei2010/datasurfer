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
    
    patt_de2en = (re.compile(r'([0-9]+),([0-9]+)'), r'\1.\2')
    patt_en2de = (re.compile(r'([0-9]+).([0-9]+)'), r'\1,\2')

    def __init__(self, s, name=None, comment=None, config=None, **kwargs):
        """
        Initializes a new instance of the STRING_OBJECT class.

        Args:
            path (str): The path to the string object.
            config (dict): Configuration settings for the string object.
            name (str): The name of the string object.
            comment (str): Additional comment for the string object.
        """
        
        self.de2en = kwargs.pop('de2en', False)
        self.en2de = kwargs.pop('en2de', False)
        
        assert not (self.de2en and self.en2de), 'Only one of `de2en` or `en2de` can be True.'
        
        super().__init__(path=None, config=config, name=name, comment=comment)

        self.kwargs = kwargs
        self.string = s

    @staticmethod
    def covert_format(string,  de2en=False, en2de=False):   
         
        if de2en:
            s = re.sub(*STRING_OBJECT.patt_de2en, string)
        elif en2de:
            s = re.sub(*STRING_OBJECT.patt_en2de, string)
        else:
            s = string            
        return s
        

    def get_df(self):

        s = STRING_OBJECT.covert_format(self.string, self.de2en, self.en2de)            
        return str2df(s, **self.kwargs)
    
    
    def dfstring(self):
        
        return self.df.to_string()
    
    
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
    
    def to_clipboard(self, **kwargs):
        import io
        self.df.index.name = None               
        s = self.df.to_string(**kwargs)        
        df = pd.read_csv(io.StringIO(s), delim_whitespace=True)        
        df.to_clipboard()

        return self
    
    def save(self, name=None):
        
        if name is None:
            name = f"{self.name}.csv"
        self.df.to_csv(name)
        
        return self