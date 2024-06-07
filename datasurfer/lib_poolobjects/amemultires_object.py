import re

from datasurfer.lib_objects.ameres_object import AMEResObject
from datasurfer.lib_objects.amegp_object import AMEGPObject
from datasurfer.datapool import DataPool

#%%
class AMESingleResObject(AMEResObject):
    
    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        
        super().__init__(path, config=config, name=name, comment=comment)
        
        path_amegp = kwargs.pop('amegp', None) or self.path.parent / (self.stem + self.ext.replace('results', 'amegp'))
        
        self.amegp = AMEGPObject(path_amegp, name=name, comment=comment)
    
    @property    
    def df_gp(self):
        
        return self.amegp.df
    
    
    
        

#%%
class AMEMultiResPool(DataPool):
       
    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        
        pattern = r'.*\.results[.]{0,1}.*'
        
        super().__init__(path, interface=AMESingleResObject, pattern=pattern, config=config, name=name, comment=comment)
        
        self.gp = self.global_params = DataPool([obj.amegp for obj in self.objs])


