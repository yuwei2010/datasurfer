
from pathlib import Path
from datasurfer.lib_objects.amevari_object import AMEResObject
from datasurfer.lib_objects.ameparam_object import AMEGPObject, AMEParamObject
from datasurfer.datapool import DataPool


#%%
class AMESingleResObject(AMEResObject):
    """
    Represents a single result object in the AME (Adaptive Mesh Refinement) simulation.

    Args:
        path (str): The path to the result file.
        config (dict, optional): Configuration parameters for the object. Defaults to None.
        name (str, optional): The name of the object. Defaults to None.
        comment (str, optional): Additional comment for the object. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        gp (AMEGPObject): The global parameter object associated with the result object.

    Properties:
        name (str): The name of the object.
        df_gp (pandas.DataFrame): The DataFrame of the global parameters.

    """

    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        
        super().__init__(path, config=config, name=name, comment=comment)
    
        f_amegp = kwargs.pop('amegp', None) or Path(str(self.path).replace('results', 'amegp'))
        f_param = kwargs.pop('param', None) or Path(str(self.path).replace('results', 'param'))

        f_amegp = f_amegp if f_amegp.is_file() else None
        f_param = f_param if f_param.is_file() else None
                   
        self.gp = self.global_param = AMEGPObject(f_amegp, name=name, comment=comment) if f_amegp else None 
        self.param = AMEParamObject(f_param, name=name, comment=comment) if f_param else None

    @AMEResObject.name.setter
    def name(self, value):
        """
        Set the name of the object.

        Args:
            value (str): The name of the object.

        """
        self._name = value
        if self.gp:
            self.gp.name = value
        if self.param:
            self.param.name = value

            
    @property    
    def df_gp(self):
        """
        Get the DataFrame of the global parameters.

        Returns:
            pandas.DataFrame: The DataFrame of the global parameters.

        """
        return self.amegp.df
    
      

#%%
class AMEMultiResPool(DataPool):
    """
    A class representing a pool of AMESingleResObject instances.

    Args:
        path (str): The path to the data pool.
        config (dict, optional): Configuration parameters for the data pool. Defaults to None.
        name (str, optional): The name of the data pool. Defaults to None.
        comment (str, optional): Additional comments about the data pool. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        gp (DataPool): The global parameters of the data pool.

    """

    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        """
        Initializes a new instance of the AMEMultiResPool class.

        Args:
            path (str): The path to the data pool.
            config (dict, optional): Configuration parameters for the data pool. Defaults to None.
            name (str, optional): The name of the data pool. Defaults to None.
            comment (str, optional): Additional comments about the data pool. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        pattern = kwargs.pop('pattern', None) or r'.*\.results[.]{0,1}.*'
        super().__init__(path, interface=AMESingleResObject, pattern=pattern, config=config, name=name, comment=comment)
        self.gp = self.global_params = DataPool([obj.gp for obj in self.objs], keep_df_index=True, name=self.name, comment=comment)
        self.params = DataPool([obj.param for obj in self.objs], keep_df_index=True, name=self.name, comment=comment)


