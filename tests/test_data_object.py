

import sys

sys.path.insert(0, r'C:\95_Programming\10_Data_Related\20_Projects\10_Git\10_Datastructure\datastructure\lib_objects')


import pandas as pd

from data_object import DATA_OBJECT
from pandas_object import PANDAS_OBJECT

#%%

obj = DATA_OBJECT(df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))

print(obj.df)