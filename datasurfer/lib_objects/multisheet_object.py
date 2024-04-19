import pandas as pd
from pathlib import Path
from datasurfer.datapool import DataPool
from datasurfer.lib_objects.pandas_object import PandasObject

#%%
class ExcelDataPool(DataPool):
       
    def __init__(self, path=None, config=None, name=None, comment=None, **kwargs):
        
        sheet_names = pd.ExcelFile(path).sheet_names
        
        objs = [PandasObject(path, config=config, comment=comment, name=sname, sheet_name=sname, **kwargs) for sname in sheet_names]
                
        super().__init__(objs)
        
        self.name = Path(path).stem
        


#%%
if __name__ == '__main__':
    
    path = r'C:\30_eATS1p6\80_Lifetime\30_HeatDump\40_Export_Histograms\10_eATS1p6_V2C\last_Temp\Histogram_tACScrewU_V2C.xlsx'
    
    dp = ExcelDataPool(path)
    
    print(dp)
    
    