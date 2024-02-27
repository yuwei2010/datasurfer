import sys

sys.path.insert(0, r'C:\95_Programming\10_Data_Related\20_Projects\10_Git\10_Datastructure\datastructure')


import pandas as pd

from lib_objects import DATA_OBJECT, PANDAS_OBJECT, FINANCE_OBJECT, ASAMMDF_OBJECT, MATLAB_OBJECT, MDF_OBJECT, AMERES_OBJECT, AMEGP_OBJECT
from pathlib import Path



obj = DATA_OBJECT(df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
obj = FINANCE_OBJECT(path=Path(r'C:\95_Programming\10_Data_Related\10_test_files\tushare_csv\000001.SZ.csv'))
obj = ASAMMDF_OBJECT(path=Path(r'C:\95_Programming\10_Data_Related\10_test_files\eATS16_2023-07-04_13-40-44_107.mf4'))

print(obj.channels)