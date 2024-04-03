import numpy as np
import pandas as pd
from datasurfer.lib_objects import DataInterface

#%%
class XKF_OBJECT(DataInterface):
    
    def __init__(self, path, key_speed='n', key_torque='Trq', config=None, name=None, comment=None):       
        super().__init__(path, config, comment=comment, name=name)
        self.key_speed = key_speed
        self.key_torque = key_torque

    
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
    
    def get_speed(self):
        
        return np.unique(self.df[self.key_speed].values)
    
    def get_torque(self):
        
        return np.unique(self.df[self.key_torque][self.df[self.key_speed]==self.get_speed().min()].values.ravel())
    
    def get_triangles(self):
        
        from datasurfer.lib_plots.plot_utils import trigrid
        x = self.df[self.key_speed].values.ravel()
        y = self.df[self.key_torque].values.ravel()
        tris = trigrid(x, y, axis=0)
        
        return tris
    
    def getXY(self):
        
        x = self.get_speed()
        y = self.get_torque()
        
        X, Y = np.meshgrid(x, y)
        
        return X, Y
    
    def getZ(self, key, **kwargs):
        
        from datasurfer.lib_stats.interp_methods import interp_linearND 
        
        X = self.df[[self.key_speed, self.key_torque]].values
        z = self.df[key].values.ravel()
         
        f = interp_linearND(X, z, **kwargs)
        
        XX, YY = self.getXY()
        
        xx = XX.ravel()
        yy = YY.ravel()
        
        zz = f(np.column_stack((xx, yy)))
        
        df = pd.DataFrame({'speed': xx, 'torque': yy, key: zz})
        df.interpolate(limit_direction='both', inplace=True)
        
        return df[key].values.reshape(XX.shape)
    
    
        
                 
# %%
