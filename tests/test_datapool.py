
import sys
sys.path.insert(0, r'C:\95_Programming\10_Data_Related\20_Projects\10_Git\10_Datastructure')


import datastructure as ds
from multiprocessing import Pool



dp = ds.DataPool(r'C:\95_Programming\10_Data_Related\10_test_files\tushare_csv', 
                 pattern=r'.*SH.*', interface=ds.FINANCE_OBJECT)[:2]
def fun(x):

     return x['close']
 
if __name__ == '__main__':


    p = Pool(4)

    print(p.map(fun, dp))