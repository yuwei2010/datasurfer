import pandas as pd
from pathlib import Path
from datasurfer.datapool import DataPool
from datasurfer.lib_objects.pandas_object import PandasObject

#%%
class ExcelDataPool(DataPool):
       
    def __init__(self, path, config=None, name=None, comment=None, sheets=None, **kwargs):
        
        shnames = pd.ExcelFile(path).sheet_names
        
        sheet_names = sheets or shnames
        
        sheet_names = [sh for sh in sheet_names if sh in shnames]
        
        objs = [PandasObject(path, config=config, comment=comment, name=sname,
                             sheet_name=sname, **kwargs) for sname in sheet_names]
                
        super().__init__(objs)
        
        self.name = name or Path(path).stem
        


#%%
if __name__ == '__main__':
    
    pass
    
    