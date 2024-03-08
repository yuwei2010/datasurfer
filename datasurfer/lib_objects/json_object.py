
import json
import pandas as pd
from datasurfer.datainterface import DataInterface, translate_config


class JSON_OBJECT(DataInterface):
   
    def __init__(self, path=None, config=None, name=None, comment=None):

        super().__init__(path, config=config, name=name, comment=comment)
        
    @translate_config()
    def get_df(self):

        dat = json.load(open(self.path, 'r'))

        df = pd.DataFrame(dat).transpose()
        return df       
        
if __name__ == '__main__':
    
    obj = JSON_OBJECT(r'C:\30_eATS1p6\33_Measurement_Evaluation\45_Winter\10_Schweden\50_Master\Valid_Measurement_V2.json')
    print(obj.df.reset_index())