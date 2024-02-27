
import sys
sys.path.insert(0, r'C:\95_Programming\10_Data_Related\20_Projects\10_Git\10_Datastructure')
import importlib

import datastructure as ds



dp = ds.DataPool(r'C:\95_Programming\10_Data_Related\10_test_files\tushare_csv', pattern=r'.*SZ.*\.csv', name='SZ', comment='Shenzhen Stock Exchange')

print(dp.describe())
