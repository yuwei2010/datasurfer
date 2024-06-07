import re

from datasurfer.lib_objects.ameres_object import AMEResObject
from datasurfer.lib_objects.amegp_object import AMEGPObject
from datasurfer.datapool import DataPool

#%%
class AMERunObject(AMEResObject):
    
    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        
        super().__init__(path, config=config, name=name, comment=comment)
        
        self.amegp = AMEGPObject(path, name=name, comment=comment)
        

#%%
class AMEMultiResPool(DataPool):
       
    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
        
        pass


