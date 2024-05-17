import numpy as np
import pandas as pd

from datasurfer.datainterface import DataInterface
from datasurfer.datautils import translate_config

class HDFObject(DataInterface):
   
    exts = ['.h5']
    
    def __init__(self, path, config=None, name=None, comment=None, **kwargs):

        super().__init__(path, config=config, name=name, comment=comment)
        self.kwargs = kwargs
        
    @translate_config()    
    def get_df(self):
        
        defaults = dict(key='df')
        defaults.update(self.kwargs)
        return pd.read_hdf(self.path, **defaults)
    
    def save(self, path, **kwargs):
        
        defaults = dict(key='df', mode='w', format='table', data_columns=True)
        defaults.update(kwargs)        
        self.df.to_hdf(path, **defaults)
        
        return self
    
    
    @staticmethod
    def from_other(other):
        
        assert isinstance(other, DataInterface)
        dat = other.to_dict()
        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
        obj = HDFObject(path=dat['path'],
                    config=dat['config'],
                    comment=dat['comment'],
                    name=dat['name'],)
        obj._df = df

        return obj 
        
