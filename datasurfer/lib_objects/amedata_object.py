import re
import numpy as np
import pandas as pd
from datasurfer.datainterface import DataInterface
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
        """
        Dump the input data into a formatted string representation.

        Args:
            *inputs: Variable length arguments representing the input data arrays.
            titles (list, optional): List of titles for each axis. Defaults to None.
            units (list, optional): List of units for each axis. Defaults to None.

        Returns:
            list: A list of strings representing the formatted data.

        Raises:
            AssertionError: If the data length does not match the axis points.

        """
        axispoints = [np.unique(arr) for arr in inputs[:-1]]
        data = np.asarray(inputs[-1]).ravel()

        ncol = len(axispoints[0])
        ndim = len(axispoints)
        nrow = int(len(data) / ncol)
        titles = titles or []
        units = units or []

        assert ncol * nrow == len(data), 'Data length does not match axis points'

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
                lines.append(''.join([f'{x:>30.14e}' for x in data[row * ncol:(row + 1) * ncol]]))
        return lines
    
#%%
class AMETableXYObject(DataInterface):

    exts = ['.dat']
    
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
            assert lines[0] == '# Table format: XY', 'Table format not match, expect xy-table.'
        return self._fhandler    
    
    @property
    def titles(self):
        """
        Property that returns the shape of the data.

        Returns:
            list: The shape of the data.
        """

        r = re.compile(r'# axis[0-9]+_title = (.*)')
        lines = list(filter(r.match, self.fhandler))    
        
        titles = [r.match(line).group(1) for line in lines]
        return titles
    
    @property
    def data(self):
        """
        Property that returns the 1D data.

        Returns:
            ndarray: The 1D data.
        """
        titles = self.titles
        return np.asarray([float(x) for l in self.fhandler[len(titles)+1:] for x in l.split()]).reshape(-1, len(titles))
    
    def get_df(self):
        """
        Method that returns the data as a pandas DataFrame.

        Returns:
            DataFrame: The data as a DataFrame.
        """
        return pd.DataFrame(self.data, columns=self.titles)
    
    @staticmethod
    def dump(df):
        
        lines = ['# Table format: XY']
        
        for title in df.columns:
            lines.append(f'# axis{df.columns.get_loc(title)+1}_title = {title}')    
            
        for rowdata in df.values:
            lines.append(''.join([f'{x:<30.14e}' for x in rowdata]))
        
        return lines
    
if __name__ == '__main__':
    pass    

# %%
