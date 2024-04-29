import dask
import dask.compatibility
import dask.delayed
import dask.distributed

#%%
class MultiProc(object):
    
    def __init__(self, db=None, n_workers=4, threads_per_worker=1, memory_limit='8GB'):
        
        self.db = db
        self._nworkers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        
    def __call__(self, n_workers=None, threads_per_worker=None, memory_limit=None):
        
        self.threads_per_worker = threads_per_worker or self.threads_per_worker
        self.memory_limit = memory_limit or self.memory_limit
        self.n_workers = n_workers or self.n_workers
        return self
   
    @property
    def n_workers(self):
        return len(self.client.get_worker_logs())
    
    @n_workers.setter
    def n_workers(self, value):
        if not hasattr(self, '_client'):
            self._nworkers = value
        
        elif self.n_workers != value:
            self._nworkers = value
            self.restart()
            
    
    @property        
    def client(self):   
        
        if not hasattr(self, '_client'):   
            from dask.distributed import Client
            self._client = Client(n_workers=self._nworkers, threads_per_worker=self.threads_per_worker,
                                  memory_limit=self.memory_limit) 

        return self._client
    
    def map(self, func, items=None):
        
        if items is None:
            items = self.db.items()
                
        L =  self.client.map(func, items)
        
        return self.client.gather(L)
       
    def scheduler_info(self):

        return self.client.scheduler_info()        

    def restart(self):
        if hasattr(self, '_client'):
            self.client.close()
            del self._client
                     
    def close(self):
        if hasattr(self, '_client'):
            self.client.close()
            del self._client
        
        if self.db is not None and hasattr(self.db, '_multiproc'):
            del self.db._multiproc
            
    


#%%
if __name__ == '__main__':

    client = MultiProc(nworkers=4)