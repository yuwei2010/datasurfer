import re
import numpy as np
import pandas as pd
from datasurfer.lib_objects import DataInterface
from functools import reduce
#%% AMEDataObject

class AMEDataObject(DataInterface):
    """ 
    A class reads AMESim tables.

    Attributes:
        fhandler (list): The lines of the file represented by the object.
        ndim (int): The number of dimensions of the data.
        shape (list): The shape of the data.
        headidx (int): The index of the first non-header line in the file.
        axispoints (list): The axis points of the data.
        data1D (ndarray): The 1D data.
        data (ndarray): The reshaped data.
    """
    exts = ['.data']
    
    @property
    def fhandler(self):
        """
        Property that returns the lines of the file represented by the object.

        Returns:
            list: The lines of the file.
        """
        if not hasattr(self, '_fhandler'):
            with open(self.path, 'r') as fobj:
                lines = [l.strip() for l in fobj.readlines() if l.strip()]
                
            self._fhandler = lines
        return self._fhandler
    
    @property
    def ndim(self):
        """
        Property that returns the number of dimensions of the data.

        Returns:
            int: The number of dimensions.
        """
        for line in self.fhandler:
            if line.startswith('#'):
                match = re.match(r'# Table format: ([0-9]+)D', line)
                if match:
                    return int(match.group(1))
        else:
            return 1
        
    @property
    def shape(self):
        """
        Property that returns the shape of the data.

        Returns:
            list: The shape of the data.
        """
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
        """
        Property that returns the index of the first non-header line in the file.

        Returns:
            int: The index of the first non-header line.
        """
        for idx, line in enumerate(self.fhandler):
            if line.startswith('#') or line.isnumeric():
                continue
            else:
                return idx
            
    @property
    def axispoints(self):
        """
        Property that returns the axis points of the data.

        Returns:
            list: The axis points of the data.
        """
        out = []
        if self.ndim == 1:
            return out.append(np.asarray([float(x) for l in self.fhandler[self.headidx:] for x in l.split()]).ravel())
        else: 
            for idx in range(self.headidx, self.headidx + self.ndim):
                out.append(np.asarray([float(x) for x in self.fhandler[idx].split()]))
            return out
    
    @property
    def data1D(self):
        """
        Property that returns the 1D data.

        Returns:
            ndarray: The 1D data.
        """
        if self.ndim == 1:
            return self.axispoints
        else:
            return np.asarray([float(x) for l in self.fhandler[(self.headidx + self.ndim):] for x in l.split()])
    
    @property
    def data(self):
        """
        Property that returns the reshaped data.

        Returns:
            ndarray: The reshaped data.
        """
        return self.data1D.reshape(self.shape)
    
    def get_df(self):
        """
        Method that returns the data as a pandas DataFrame.

        Returns:
            DataFrame: The data as a DataFrame.
        """
        return pd.DataFrame(self.data1D, columns=['data'])
    
    @staticmethod
    def dump(*inputs, titles=None, units=None):
        
        axispoints = [np.unique(arr) for arr in inputs[:-1]]
        data = np.asarray(inputs[-1]).ravel()
        
        ncol= len(axispoints[0])
        ndim = len(axispoints)
        nrow = int(len(data)/ncol)
        titles = titles or []
        units = units or []
        
        assert ncol*nrow == len(data), 'Data length not match axispoints'
               
        lines = [f'# Table format: {ndim}D']
             
        if units:
            lines.append(f'# table_unit = {units[0]}')
            for idx, unit in enumerate(units[1:], 1):
                lines.append(f'# axis{idx}_unit = {unit}')   
                    
        for idx, title in enumerate(titles, 1):
            
            lines.append(f'# axis{idx}_title = {title}')
       
        if ndim == 1:
            for x, y in zip(axispoints[0], data):
                lines.append(f'{x:>30.14e} {y:>30.14e}')                
        else:
            for arr in axispoints:
                lines.append(f'{len(arr)}')
            for arr in axispoints:
                lines.append(''.join([f'{x:>30.14e}' for x in arr]))
            
            for row in range(nrow):               
                lines.append(''.join([f'{x:>30.14e}' for x in data[row*ncol:(row+1)*ncol]]))         
        return lines
    
if __name__ == '__main__':
    pass    

# %%
