import dask
import dask.compatibility
import dask.delayed
import dask.distributed

#%%
class MultiProc(object):
    
    def __init__(self, db=None, n_workers=4):
        
        self.db = db
        self._nworkers = n_workers
   
    @property
    def n_workers(self):
        return len(self.client.get_worker_logs())
    
    @n_workers.setter
    def n_workers(self, value):
        if not hasattr(self, '_client'):
            self._nworkers = value
        
        elif self.n_workers != value:
            self._nworkers = value
            self.client.close()
            del self._client
            
    
    @property        
    def client(self):   
        
        if not hasattr(self, '_client'):   
            from dask.distributed import Client
            self._client = Client(n_workers=self._nworkers, threads_per_worker=1) #, memory_limit='8GB'

        return self._client
    
    def map(self, func, items=None):
        
        if items is None:
            items = self.db.items()
                
        return self.client.map(func, items)
    
    def scheduler_info(self):

        return self.client.scheduler_info()        

            
    def close(self):
        self.client.close()
        #del self.db._client
        
        if self.db is not None:
            del self.db._multiproc
            
    


#%%
if __name__ == '__main__':

    client = MultiProc(nworkers=4)