import pyarrow.parquet as pq
import pyarrow as pa 
import pandas as pd
import json
from datasurfer.datainterface import DataInterface
from datasurfer.datautils import translate_config, parse_config

class ParquetObject(DataInterface):
    """
    https://arrow.apache.org/docs/python/parquet.html
    """
   
    exts = ['.parquet']
    
    def __init__(self, path=None, config=None, name=None, comment=None, **kwargs):
        
        """
            kwargs: 
                Columns: List of columns to read
        
        """

        super().__init__(path, name=name, comment=comment, config=config)
        
        self.kwargs = kwargs
    
    @property
    def comment(self):
        if self._comment is None:
            metadata = self.fhandler.schema.metadata
            if b'comment' in metadata:
                self._comment = json.loads(metadata[b'comment'].decode('utf-8'))       
        return self._comment
    
    @comment.setter
    def comment(self, value):
        
        self._comment = value
        
    @property
    def config(self):
        if self._config is None:
            metadata = self.fhandler.schema.metadata
            if b'config' in metadata:
                self._config = json.loads(metadata[b'config'].decode('utf-8'))     
        return self._config
    
    @config.setter
    def config(self, val):                
        self._config = parse_config(val)  
        if hasattr(self, '_df'):
            del self._df
        
    @property
    def fhandler(self):

        if not hasattr(self, '_fhandler'):
 
            self._fhandler = pq.read_table(self.path, **self.kwargs)
                
        return self._fhandler
    
    @translate_config()    
    def get_df(self):
        
        return self.fhandler.to_pandas()
    
    def add_metadata(self, key, value):
        
        assert isinstance(key, (str, dict)), "Key must be a string or a dictionary."

        metadata = self.fhandler.schema.metadata     
                
        str_attr = value if isinstance(value, str) else json.dumps(value)            
        metadata.update({key: str_attr})
        
        new_schema = self.fhandler.schema.with_metadata(metadata)
        new_table = self.fhandler.cast(new_schema)  
        
        return new_table     
    
    
    def save(self, path, **kwargs):
        
        new_table = self.fhandler
        
        if self.comment is not None:
            new_table = self.add_metadata('comment', self.comment)
            
        if self.config is not None:
            
            new_table = self.add_metadata('config', self.config)

        pq.write_table(new_table, path, **kwargs)
        
        return self
    
    
    @staticmethod
    def from_other(other):
        
        assert isinstance(other, DataInterface)
        dat = other.to_dict()
        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
        obj = ParquetObject(path=dat['path'],
                    config=dat['config'],
                    comment=dat['comment'],
                    name=dat['name'],)
        obj._fhandler = pa.Table.from_pandas(df)


        return obj     
    
    
        
        