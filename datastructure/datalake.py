Data Lake

class DataLake(object):
    
    
    def __init__(self, items):
        
        
        self.objs = items
        
        
    def keys(self):
        
        zcount = len(str(len(self.objs))) + 1
        
        strfmt = f':0{zcount}'        
        
        fmt = 'DataPool_' + '{' + strfmt + '}'
                
        def get():
            
            for idx, dp in enumerate(self.objs):
                
                if dp.name is None:
                    
                    yield fmt.format(idx)
                else:
                    
                    yield dp.name
                
        return list(get())

    def search(self, patt, ignore_case=True, raise_error=False):
                
        pass