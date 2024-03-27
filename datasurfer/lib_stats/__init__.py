import numpy as np
from datasurfer.datautils import arghisto, parse_data

#%%

class Stats(object):
    
    def __init__(self, dp) -> None:
        
        self.dp = dp   
        
    
    @parse_data   
    def arghisto(self, val, *, bins, **kwargs):
        
        return arghisto(val, bins)
        
 
        
    
# %%
