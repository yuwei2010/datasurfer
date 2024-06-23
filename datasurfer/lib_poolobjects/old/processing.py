# -*- coding: utf-8 -*-

import os
import time
import datetime
import threading
import queue
import logging
import multiprocessing
import subprocess
import pandas as pd
#%%---------------------------------------------------------------------------#
def bjobs():
    
    
    
    column, *values = SPopen('bjobs', shell=False).check_output().decode('utf-8').split('\n')
    
    if values:
        
        column = column.split()
        column.extend(['DAY', 'TIME'])
        

        
        values = [v.split() for v in values]
        
        df = pd.DataFrame(values, columns=column).dropna()
        
        print(datetime.datetime.now().year
              )
        submit_time = [pd.to_datetime(datetime.datetime.strptime('{} {} {} {}'.format(b, d, datetime.datetime.now().year, t), 
                       '%b %d %Y %H:%M')) for b, d, t in 
                                        zip(df['SUBMIT_TIME'], df['DAY'], df['TIME'])]
        df['SUBMIT_TIME'] = submit_time
        
        df.drop(['DAY', 'TIME'], axis=1, inplace=True)
        return df        
#%%---------------------------------------------------------------------------#
def bkill(ids, r=True):
    
    if isinstance(ids, pd.DataFrame):
        
        ids = ids['JOBID']
    
    ids = ' '.join([str(s) for s in ids])
    
    if r == True:
        
        cmd = f'bkill -r {ids}'
    
    else:
       
        cmd = f'bkill {ids}'

    
    return SPopen(cmd).check_output()
    
#%%---------------------------------------------------------------------------#

class SubCMD(object):
    
    cmd_prefix = '--'
    
    #%%-----------------------------------------------------------------------#    
    def __init__(self, cmd, argv='', **kwargs):
        
        prefix = kwargs.pop('prefix', SubCMD.cmd_prefix)

        self._cmd = [cmd]
        self._argv = '"{}"'.format(argv) if ' ' in argv.strip() else argv
        
        for k, v in kwargs.items():
            
            if isinstance(v, bool):
                if v is True:
                    self._cmd.append('{}{}'.format(prefix, k))
            else:
                self._cmd.append('{}{}={}'.format(prefix, k, v))
                
    
    #%%-----------------------------------------------------------------------#   
    def start(self, *args, **kwargs):
        
            
        cmd = ' '.join([' '.join(self._cmd), self._argv])
        
        if not args:
            
            argin = ''
        
        else:
            
            argin = ' '.join([str(v) for v in args])
        
        return SPopen(cmd.format(argin), **kwargs).check_output()
        
            
                        
#%%---------------------------------------------------------------------------#

class SubPY(SubCMD):
    
    def __init__(self, pyfile, **kwargs):
        
        which = kwargs.pop('which', 'python')
        spath = kwargs.pop('syspath', None)
        
        
        syspath = '' if spath is None else 'PYTHONPATH={}'.format(spath.strip())

        settings = dict(mpi='NOMPI', interactive=True)
        
        settings.update(kwargs)
        
        cmd = 'subbin'
        
        argv = '{} {} {} {{}}'.format(syspath, which, pyfile)
        
        super().__init__(cmd, argv, **settings)

#%%---------------------------------------------------------------------------#
class SPopen(object):
    def __init__(self, cmd, timeout=None, **kwargs):

        self.cmd = cmd
        self.shell = kwargs.pop('shell', True)
        self.kwargs = kwargs
        self.timeout = timeout
        self.start()
        
    #%%-----------------------------------------------------------------------#
    def call(self):
        
        return self.proc.wait()
    
    #%%-----------------------------------------------------------------------#
    def check_call(self):
        
        retcode=self.call()
        if retcode:
            raise subprocess.CalledProcessError(retcode, self.cmd)
        return 0
    
    #%%-----------------------------------------------------------------------#
    def check_output(self):
        retcode = self.proc.poll()
        if retcode:
            raise subprocess.CalledProcessError(retcode, self.cmd)
        return self.stdout
    
    #%%-----------------------------------------------------------------------#
    def start(self):
        
        def target():
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, 
                                         shell=self.shell, **self.kwargs)
            self.stdout, _ = self.proc.communicate()

        thread=threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            self.proc.terminate()
            thread.join()
            
#%%---------------------------------------------------------------------------#
class Exit(Exception): pass

#%%---------------------------------------------------------------------------#
class Input_Thread(threading.Thread):
    
    def __init__(self, inqueue, args):
        
        super().__init__()
        
        self.inqueue = inqueue
        self.args = args

        self.start()
        self.join()

    #%%-----------------------------------------------------------------------#          
    def run(self):
        
        for arg in self.args:

            self.inqueue.put(arg)
            

#%%---------------------------------------------------------------------------#
class Process(multiprocessing.Process):
    
    def __init__(self, inqueue, outqueue, name):
        
        super().__init__(name=name)
        
        self.func = None
        self.inqueue = inqueue
        self.outqueue = outqueue

    #%%-----------------------------------------------------------------------#
    def run(self):
        
        while 1:

            try:
                args = self.inqueue.get(block=False)
            except queue.Empty:
                break
            
            try:
                
                res = self.func(*args)            
                self.outqueue.put(res)
                
            except Exception as err:
                
                self.outqueue.put(Exception(
                        'Thread-{:>02}: "{}: {}".'.format(int(self.name), type(err).__name__, err)))
            
        self.outqueue.put(Exit(
                            'Thread-{:>02} is closed.'.format(int(self.name))))

        
#%%---------------------------------------------------------------------------#
class ProcessPool(object):
    
    def __init__(self, num_worker=1):
 
        self.logger = logging.getLogger(__name__)
        self.outqueue = multiprocessing.Manager().Queue(maxsize=0)
                
        self.inqueue = multiprocessing.Manager().Queue(maxsize=0)
                    
        self.processes = []
        self.add(num_worker)

    #%%-----------------------------------------------------------------------#   
    def add(self, num=1):
                
        for n in range(num):
            
            proc = Process(self.inqueue, self.outqueue, name=str(len(self.processes)))
            self.processes.append(proc)

            
        return self  

    #%%-----------------------------------------------------------------------#   
    def put(self, args):
        
        Input_Thread(self.inqueue, args)     
                      
        return self

    #%%-----------------------------------------------------------------------#        
    def start(self, func):
        
        for process in self.processes_waiting():
                            
            process.func = func
            
            process.start() 
            
        
        for process in self.processes:
            process.join()
                
        return self
                        
    #%%-----------------------------------------------------------------------#        
    def run(self, func):
        

                                                        
        return list(self.poll(func))

    #%%-----------------------------------------------------------------------#   
    def wait(self, func):
        
        for res in self.poll(func):
            pass
        
        return self
    #%%-----------------------------------------------------------------------#    
    def processes_alive(self):
        
        return [process for process in self.processes if process.is_alive()]
            
    #%%-----------------------------------------------------------------------#    
    def processes_waiting(self):
        
        return [process for process in self.processes if process.ident is None]  
    
    #%%-----------------------------------------------------------------------#  
    def poll(self, func):
        
        self.start(func)
            
        while self.outqueue.qsize() or self.processes_alive():
            
            out = self.outqueue.get()
            
            if isinstance(out, Exit):
                
                self.logger.debug(out)
                            
            elif isinstance(out, Exception):
                
                self.logger.exception('{}'.format(str(out)))
                
            else:
                
                yield out
#%%---------------------------------------------------------------------------#
class Thread(threading.Thread):
    
    def __init__(self, inqueue, outqueue, infoqueue, name):
        
        super().__init__(name=name)
        
        self.func = None
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.infoqueue = infoqueue
        

    #%%-----------------------------------------------------------------------#
    def run(self):
        
        while 1:
            
            if self.infoqueue:
                info = self.infoqueue.get()
                
                self.infoqueue.put(info)
                              
                if self.name not in info:
                    if not self.inqueue.qsize():
                        break
                    else:
                        time.sleep(0.1)
                        continue
            try:
                args = self.inqueue.get(block=False)
            except queue.Empty:
                break            
            
            try:
                
                res = self.func(*args)
                               
                self.outqueue.put(res)
                
            except Exception as err:
                
                self.outqueue.put(Exception(
                        'Thread-{:>02}: "{}: {}".'.format(int(self.name), type(err).__name__, err)))
    
        self.outqueue.put(Exit(
                            'Thread-{:>02} is closed.'.format(int(self.name))))
                    

#%%---------------------------------------------------------------------------#

class ThreadPool(object):
    
    def __init__(self, total_workers, notify_file=None):
                

        self.logger = logging.getLogger(__name__)
        self.outqueue = queue.Queue(maxsize=1)
        self.inqueue = queue.Queue(maxsize=0)

        self.notify = notify_file
        
        
        if self.notify is not None:
            
            init_workers = self.read_notify()
        
        else:
            
            init_workers = total_workers
        
        self.workers = init_workers
        
        self.threads = [] 
        
        if self.notify is None:
            
            self.infoqueue = None
            self.add(total_workers)
        
        else:
            self.infoqueue = queue.Queue(maxsize=1)
            self.add(total_workers)
            tids = self.thread_ids()
                
            self.infoqueue.put(tids[:self.workers])
        
    #%%-----------------------------------------------------------------------#
    def read_notify(self):

        with open(self.notify, 'r') as fobj:
            
            line, *_ = fobj.readlines()
            
            return int(line.strip())
        
    #%%-----------------------------------------------------------------------#   
    def add(self, num):
                
        for n in range(num):
            
            thread = Thread(self.inqueue, self.outqueue, 
                            self.infoqueue, name=str(len(self.threads)))
            self.threads.append(thread)
            
        return self  
    
    #%%-----------------------------------------------------------------------#   
    def put(self, args):
        for arg in args:

            self.inqueue.put(arg)
        return
    #%%-----------------------------------------------------------------------# 
    def start(self, func):
        

        for thread in self.threads_waiting():
                            
            thread.func = func
            
            thread.start() 
                
        return self
    #%%-----------------------------------------------------------------------#    
    def thread_ids(self, only_alive=False):
        
        if only_alive:
            return [thread.name for thread in self.threads_alive()]
        else:
            
            return [thread.name for thread in self.threads]
        
    #%%-----------------------------------------------------------------------#    
    def threads_alive(self):
        
        return [thread for thread in self.threads if thread.is_alive()]   
    #%%-----------------------------------------------------------------------#    
    def threads_waiting(self):
        
        return [thread for thread in self.threads if thread.ident is None]
    
    #%%-----------------------------------------------------------------------#   
    def poll(self, func):
        
        self.start(func)
        while self.outqueue.qsize() or self.threads_alive():
            
            try:
                out = self.outqueue.get(timeout=1)
            except queue.Empty:
                continue
                        
            if isinstance(out, Exit):
                
                self.logger.debug(out)
            
            elif isinstance(out, Exception):
                
                self.logger.exception('{}'.format(str(out)))
                
            else:
                
                yield out  
                
                
            if self.notify and os.path.lexists(self.notify):
                
                new_worker = self.read_notify()
                                
                if new_worker != self.workers:
                    
                    self.workers = new_worker
                    self.infoqueue.get()
                    tids = self.thread_ids(only_alive=True)
                        
                    self.infoqueue.put(tids[:self.workers])
                
    #%%-----------------------------------------------------------------------# 
    def wait(self, func):
        
        
        for n in self.poll(func):
            pass
        return self               

    #%%-----------------------------------------------------------------------#                 
    def run(self, func):
        
        return list(self.poll(func))
                
#%%---------------------------------------------------------------------------#
if __name__ == '__main__':

    print(bjobs())
    
    
    
