import numpy as np
import pandas as pd
from datasurfer.lib_objects import DataInterface

#%%

class KF_OBJECT(DataInterface):

    def __init__(self, path, key_x='n', key_y='Trq', config=None, name=None, comment=None):       
        super().__init__(path, config, comment=comment, name=name)
        self.key_x = key_x
        self.key_y = key_y

    
    def get_df(self):
        
        with open(self.path, 'r') as fobj:
            lines = [l.strip() for l in fobj.readlines() if l.strip()]  
            
        for idx, l in enumerate(lines):
            if l.startswith('KF') and l.split()[0] == 'KF':
                break

        cols = lines[idx+1].split()
        data = []
        
        for l in lines[idx+2:]:
            s = l.split()
            if s[0].isnumeric():
                data.append(list(map(float, s)))
        
        data = np.array(data, dtype=float)
        df = pd.DataFrame(data, columns=cols)
        return df
    
    def getx(self):
        
        return np.unique(self.df[self.key_x].values)
    
    def gety(self):
        
        return np.unique(self.df[self.key_y][self.df[self.key_x]==self.getx().min()].values.ravel())
    
    def get_triangles(self):
        
        from datasurfer.lib_plots.plot_utils import trigrid
        x = self.df[self.key_x].values.ravel()
        y = self.df[self.key_y].values.ravel()
        tris = trigrid(x, y, axis=0)
        
        return tris
    
    def getXY(self):
        
        x = self.getx()
        y = self.gety()
        
        X, Y = np.meshgrid(x, y)
        
        return X, Y
    
    def getZ(self, key, **kwargs):
        """
        Interpolates the values of a given key in the DataFrame based on the x and y coordinates.
        
        Parameters:
            key (str): The column name of the values to be interpolated.
            **kwargs: Additional keyword arguments to be passed to the interpolation function.
            
        Returns:
            numpy.ndarray: The interpolated values reshaped to match the shape of the x and y coordinates.
        """
        from datasurfer.lib_signals.interp_methods import interp_linearND 
        import numpy as np
        import pandas as pd
        
        X = self.df[[self.key_x, self.key_y]].values
        z = self.df[key].values.ravel()
         
        f = interp_linearND(X, z, **kwargs)
        
        XX, YY = self.getXY()
        
        xx = XX.ravel()
        yy = YY.ravel()
        
        zz = f(np.column_stack((xx, yy)))
        
        df = pd.DataFrame({'x': xx, 'y': yy, key: zz})
        df.interpolate(limit_direction='both', inplace=True)
        
        return df[key].values.reshape(XX.shape)
    
    
        
                 
# %%
