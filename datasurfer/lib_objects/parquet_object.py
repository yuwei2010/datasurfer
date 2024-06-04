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
    def orgpath(self):
        if not hasattr(self, '_orgpath'):
            metadata = self.fhandler.schema.metadata
            if b'orgpath' in metadata:  
                self._orgpath = json.loads(metadata[b'orgpath'].decode('utf-8'))       
            else:
                self._orgpath = None
            
        return self._orgpath    
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
            """
            Returns the DataFrame representation of the Parquet object.
            
            Returns:
                pandas.DataFrame: The DataFrame representation of the Parquet object.
            """
            return self.fhandler.to_pandas()
    
    def add_metadata(self, **kwargs):
        """
        Adds metadata to the Parquet file.

        Args:
            **kwargs: Keyword arguments representing the metadata key-value pairs.
                        The values can be either a string or a dictionary.

        Returns:
            A new Parquet table with the updated metadata.
        """
        assert all(isinstance(value, (str, dict)) for value in kwargs.values()), "Value must be a string or a dictionary."

        metadata = self.fhandler.schema.metadata     
        for key, value in kwargs.items():    
            str_attr = value if isinstance(value, str) else json.dumps(value)            
            metadata.update({key: str_attr})
        
        new_schema = self.fhandler.schema.with_metadata(metadata)
        new_table = self.fhandler.cast(new_schema)  
        
        return new_table
    
    
    def save(self, path, **kwargs):
        """
        Save the Parquet table to a specified path.

        Args:
            path (str): The path where the Parquet table will be saved.
            **kwargs: Additional keyword arguments to be passed to the `pq.write_table` function.

        Returns:
            self: The Parquet object itself.

        """
        new_table = self.fhandler
        metadata = dict()

        if self.comment is not None:
            metadata['comment'] = self.comment

        if self.config is not None:
            metadata['config'] = self.config
            
        if hasattr(self, '_orgpath') and self._orgpath is not None:
            metadata['orgpath'] = self.orgpath

        if metadata:
            new_table = self.add_metadata(**metadata)

        pq.write_table(new_table, path, **kwargs)

        return self
    
    
    @staticmethod
    def from_other(other):
        """
        Create a ParquetObject from another DataInterface object.

        Parameters:
        - other: DataInterface
            The other DataInterface object to create the ParquetObject from.

        Returns:
        - obj: ParquetObject
            The created ParquetObject.

        Raises:
        - AssertionError: If the input object is not an instance of DataInterface.
        """

        assert isinstance(other, DataInterface)
        dat = other.to_dict()
        df = pd.DataFrame(dat['df'], index=dat['index'], columns=dat['columns'])
        obj = ParquetObject(path=None,
                            config=dat['config'],
                            comment=dat['comment'],
                            name=dat['name'])
        obj._orgpath = other.path
        obj._fhandler = pa.Table.from_pandas(df)

        return obj

#%%    
    
        
        