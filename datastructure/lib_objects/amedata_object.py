import pandas as pd
import numpy as np
from ..datainterface import DataInterface


class AMEDATA_OBJECT(DataInterface):
    
    @property
    def fhandler(self):
        if not hasattr(self, '_fhandler'):
            with open(self.path, 'r') as fobj:
                lines = [l.strip() for l in fobj.readlines() if l.strip()]
                
            self._fhandler = lines
        return self._fhandler
    
    @property
    def ndim(self):
        
        for line in self.fhandler:
            if line.startswith('#'):
                match = re.match(r'# Table format: ([0-9]+)D', line)
                if match:
                    return int(match.group(1))
        else:
            return 1
        
    @property
    def shape(self):
        
        if self.ndim == 1:
            return [len(self.data1D)]
        
        else:
        
            out = []
            for line in self.fhandler:
                
                if line.startswith('#'):
                    continue
                elif line.isnumeric():
                    out.append(int(line))
                else:
                    break
            assert len(out) == self.ndim, "shape not match ndim"
            return out
    
    @property
    def headidx(self):
    
        for idx, line in enumerate(self.fhandler):
            if line.startswith('#') or line.isnumeric():
                continue
            else:
                return idx
            
    @property
    def axispoints(self):
        out = []
        if self.ndim == 1:
            return out.append(np.asarray([float(x) for l in self.fhandler[self.headidx:] for x in l.split()]).ravel())
        else: 
            
            for idx in range(self.headidx, self.headidx + self.ndim):
                out.append(np.asarray([float(x) for x in self.fhandler[idx].split()]))
                        
            return out
    
    @property
    def data1D(self):
        if self.ndim == 1:
            
            return self.axispoints
        
        else:
       
            return np.asarray([float(x) for l in self.fhandler[(self.headidx + self.ndim):] for x in l.split()])
    
    @property
    def data(self):
        
        return self.data1D.reshape(self.shape)
    
    def get_df(self):
        
        return pd.DataFrame(self.data1D, columns=['data'])