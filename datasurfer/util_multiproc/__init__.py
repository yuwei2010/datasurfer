from typing import Any
import dask
import dask.compatibility
import dask.delayed
import dask.delayed
import dask.delayed
import dask.distributed


class MultiProc(object):
    """
    A class for managing multiprocessing tasks using Dask.

    This class provides functionality for executing functions asynchronously using Dask,
    managing the number of workers, and retrieving information about the scheduler.

    Attributes:
        db (str): The database to be used. Defaults to None.
        n_workers (int): The number of workers to be used. Defaults to 4.
        threads_per_worker (int): The number of threads per worker. Defaults to 1.
        memory_limit (str): The memory limit for each worker. Defaults to '8GB'.
    """    

    def __init__(self, db=None, n_workers=4, threads_per_worker=1, memory_limit='8GB'):
        """
        Initialize MyClass object.

        Args:
            db (str): The database to be used. Defaults to None.
            n_workers (int): The number of workers to be used. Defaults to 4.
            threads_per_worker (int): The number of threads per worker. Defaults to 1.
            memory_limit (str): The memory limit for each worker. Defaults to '8GB'.
        """
        self.db = db
        self._nworkers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        
    def __call__(self, n_workers=None, threads_per_worker=None, memory_limit=None):
        """
        Executes the object as a function.

        Args:
            n_workers (int, optional): The number of workers to use. Defaults to None.
            threads_per_worker (int, optional): The number of threads per worker. Defaults to None.
            memory_limit (int, optional): The memory limit per worker. Defaults to None.

        Returns:
            self: The modified object.
        """
        self.threads_per_worker = threads_per_worker or self.threads_per_worker
        self.memory_limit = memory_limit or self.memory_limit
        self.n_workers = n_workers or self.n_workers
        return self
    
    def __getattr__(self, name: str) -> Any:
        """
        Retrieve the value of an attribute dynamically.

        This method is called when an attribute is accessed that doesn't exist
        in the current object. It attempts to retrieve the attribute from the
        underlying database object and returns it.

        Parameters:
        - name (str): The name of the attribute to retrieve.

        Returns:
        - Any: The value of the attribute.

        Raises:
        - AttributeError: If the attribute doesn't exist in the current object
                            or the underlying database object.
        """
        try:
            return self.__getattribute__(name)
        
        except AttributeError:
            return dask.delayed(getattr(self.db, name))
        
   
    @property
    def n_workers(self):
        """
        Returns the number of workers currently active.

        Returns:
            int: The number of active workers.
        """
        return len(self.client.get_worker_logs())
    
    @n_workers.setter
    def n_workers(self, value):
        """
        Sets the number of workers for the multiprocessing pool.

        Args:
            value (int): The number of workers to set.

        Returns:
            None
        """
        if not hasattr(self, '_client'):
            self._nworkers = value
        
        elif self.n_workers != value:
            self._nworkers = value
            self.restart()
            
    
    @property        
    def client(self):
        """
        Returns a Dask distributed client.

        If the client has not been created yet, it creates a new client with the specified number of workers,
        threads per worker, and memory limit. Otherwise, it returns the existing client.

        Returns:
            dask.distributed.Client: The Dask distributed client.
        """
        if not hasattr(self, '_client'):
            from dask.distributed import Client
            self._client = Client(n_workers=self._nworkers, threads_per_worker=self.threads_per_worker,
                                  memory_limit=self.memory_limit)

        return self._client
    
    def run(self, fun, *args, **kwargs):
        """
        Run a function asynchronously using Dask.

        Parameters:
        - fun: The function to be executed.
        - args: Positional arguments to be passed to the function.
        - kwargs: Keyword arguments to be passed to the function.

        Returns:
        - A Dask delayed object representing the execution of the function.
        """
        return dask.delayed(fun)(*args, **kwargs)
    
    def start(self, lzfun):
        """
        Starts the computation of a delayed object using Dask.

        Parameters:
            lzfun (dask.delayed.Delayed): The delayed object to compute.

        Returns:
            The computed result of the delayed object.
        """
        assert isinstance(lzfun, dask.delayed.Delayed), 'The input must be a delayed object.'
        
        return dask.compute(lzfun)
    
    def submit(self, fun, *args, **kwargs):
        """
        Submits a function to be executed asynchronously.

        Args:
            fun: The function to be executed.
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            The result of the function execution.

        """
        return self.client.submit(fun, *args, **kwargs)
               
    
    def map(self, func, items=None):
        """
        Applies the given function to each item in the specified list of items.
        
        Args:
            func: The function to apply to each item.
            items: The list of items to apply the function to. If None, the items from the database will be used.
        
        Returns:
            A list of results after applying the function to each item.
        """
        if items is None:
            items = self.db
                
        res =  self.client.map(func, items)
        
        return self.client.gather(res)
       
    def scheduler_info(self):
        """
        Retrieves information about the scheduler.

        Returns:
            dict: A dictionary containing information about the scheduler.
        """
        return self.client.scheduler_info()

    def restart(self):
        """
        Restarts the client connection.

        If the `_client` attribute exists, it closes the client connection and deletes the attribute.
        """
        if hasattr(self, '_client'):
            self.client.close()
            del self._client
                     
    def close(self):
        """
        Closes the connection and cleans up any resources used by the multiproc object.
        """
        if hasattr(self, '_client'):
            self.client.close()
            del self._client
        
        if self.db is not None and hasattr(self.db, '_multiproc'):
            del self.db._multiproc
            
    


#%%
if __name__ == '__main__':

    client = MultiProc(nworkers=4)