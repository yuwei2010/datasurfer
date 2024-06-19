#%% Import Libraries
import re
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path
from datasurfer.lib_objects.ameparam_object import AMEObject
from datasurfer.datautils import translate_config, extract_channels

#%%
class AMEVLObject(AMEObject):
    """
    Represents an AMEVL object.

    Args:
        path (str): The path to the object.
        name (str, optional): The name of the object. Defaults to None.
        comment (str, optional): Any additional comment for the object. Defaults to None.
    """
    ametype = 'vl'
    
    def __init__(self, path, idxstr='ref', name=None, comment=None):
        
        super().__init__(path, name=name, comment=comment)
        
        self.idxstr = str(idxstr)
        
    @property
    def ext_idx(self):
        return self.idxstr
        
    @property
    def fhandler(self):
        
        if not hasattr(self, '_fhandler'):
            rootlst = ET.parse(self.path)

            for param in rootlst.getroot():
                for attr in param:
                    if attr.attrib['dataset'] == self.ext_idx:
                        checksum = attr.attrib['checksum']
                        break
            vlpath = Path(self.path.parent / (self.stem+'.vl.crc_'+checksum))
            assert vlpath.is_file(), f'Cannot find any VL file for index {self.idxstr}.'
            self._fhandler = ET.parse(vlpath)
                                    
        return self._fhandler

    def get_df(self):
        """
        Parses the XML file and returns a DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame containing the parsed data.
        """
        root = self.fhandler.getroot()
        out = dict()
        varlst = root.find('VARS_LIST')

        for var in varlst:
            attrs = var.attrib

            if 'Data_Path' in attrs:
                # attrs['PARENT'] = None
                out[attrs['Data_Path']] = attrs

            tievars = var.findall('TIE_VAR')
            if tievars:
                for tievar in tievars:
                    attrs = tievar.attrib
                    #attrs['PARENT'] = var.attrib['Data_Path']
                    out[attrs['Data_Path']] = attrs

        df = pd.DataFrame(out)
        return df
    
    def save(self, name=None):
        
        path = name or self.path
        
        root = self.fhandler.getroot()
        varlst = root.find('VARS_LIST') 
        
        for var in varlst:
            attrs = var.attrib

            if 'Data_Path' in attrs:
                key = attrs['Data_Path']
                for k in attrs.keys():
                    var.set(k, str(self.df[key][k]))
                    
            tievars = var.findall('TIE_VAR')
            if tievars:
                for tievar in tievars:
                    attrs = tievar.attrib
                    key = attrs['Data_Path']
                    for k in attrs.keys():
                        tievar.set(k, str(self.df[key][k]))    
                        
        self.fhandler.write(path)   
        
        return self         
        
#%%    
class AMESSFObject(AMEObject):
    
    
    ametype = 'ssf'    
    
    def get_df(self):
        
        out = dict()

        with open(self.path, 'r') as fobj:
            lines = fobj.readlines()

        for idx, l in enumerate(lines, start=1):
            item = dict()
            l = l.strip()

            try:
                raw, = re.findall(r'Data_Path=\S+', l)
                l = l.replace(raw, '').strip()
                s, = re.findall(r'Data_Path=(.+)', raw)
                item['Data_Path'] = s

                raw, = re.findall(r'Param_Id=\S+', l)
                l = l.replace(raw, '').strip()
                s, = re.findall(r'Param_Id=(.+)', raw)
                item['Param_Id'] = s
            except ValueError:
                continue

            try:
                raw, = re.findall(r'\[\S+\]', l)
                l = l.replace(raw, '').strip()
                s, = re.findall(r'\[(\S+)\]', raw)
                item['Unit'] = s
            except ValueError:
                item['Unit'] = '-'

            raw, = re.findall(r'^[01]+\s+\S+\s+\S+\s+\S+', l)
            l = l.replace(raw, '').strip()
            s, = re.findall(r'^[01]+\s+(\S+\s+\S+\s+\S+)', raw)
            item['Label'] = s
            item['Description'] = l
            item['Row_Index'] = idx

            out[item['Data_Path']] = item

        df = pd.DataFrame(out)
        return df
           

#%% AMERES_OJBECT
class AMEResObject(AMEObject):
    """
    Represents an object for handling AMERES data.

    Args:
        path (str): The path to the AMERES data file.
        config (dict): Configuration parameters for data extraction.
        name (str): The name of the AMERES object.
        comment (str): Additional comment for the AMERES object.

    Attributes:
        params (dict): A dictionary containing the parameters of the AMERES data.
        t (numpy.ndarray): An array containing the time values of the AMERES data.
        channels (list): A sorted list of data channels in the AMERES data.

    Methods:
        get_channels: Retrieves the specified data channels from the AMERES data.
        get_df: Retrieves a DataFrame containing the specified data channels.
        get: Retrieves the specified data channels or time values from the AMERES data.
        get_results: Retrieves the raw data from the AMERES data file.
        keys: Retrieves the keys (data channels) of the AMERES data.
        search_channel: Searches for data channels that match a given pattern.
    """

    ametype = 'results'

    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
                
        super().__init__(path, config=config, name=name, comment=comment) 
        f_amevl = kwargs.pop('amevl', None)     

        if f_amevl is None:

            f_amevl = Path(self.path.parent / (self.stem+'.vl.active'))


                
        if f_amevl.is_file():            
            self.vl = AMEVLObject(f_amevl, self.ext_idx,  name=name, comment=comment)
        else:
            self.vl = None       
        
        f_amessf = kwargs.pop('amessf', None)
        
        if f_amessf is None:       
            f_amessf = str(self.path).replace('results', 'ssf')
            
        if Path(f_amessf).is_file():  
            self.ssf = AMESSFObject(f_amessf, name=name, comment=comment) 
        else:
            self.ssf = None
        
        if self.ssf is None and self.vl is None:
            raise ValueError('Please assign a valid SSF or VL file to the AMEResObject.')
        
          
    
    @property
    def t(self):
            """
            Returns the index values of the DataFrame stored in the object.
            If the DataFrame is empty and the config is None, it retrieves the results for the first row.
            
            Returns:
                numpy.ndarray: The index values of the DataFrame.
            """
            if not len(self.df) and self.config is None:
                t = self.get_results(rows=[0])
            else:
                t = self.df.index
            return np.asarray(t)

    @property
    def params(self):        
        if self.vl:
            return self.vl.df
        else:
            return self.ssf.df
        
    @property
    def channels(self):
        
        return self.params.columns.tolist()


    @translate_config()
    @extract_channels()
    def get_channels(self, *channels):
        """
        Retrieves the data for the specified channels.
        
        Parameters:
        *channels: Variable number of channel names
        
        Returns:
        df: Pandas DataFrame containing the data for the specified channels, with time as the index.
        """
        
        params = self.params
        
        key_row = 'VARNUM' if 'VARNUM' in params.index else 'Row_Index'
        row_indices = [int(params[c][key_row]) for c in channels]
                
        array = self.get_results(rows=row_indices)
        
        df = pd.DataFrame(dict(zip(channels, array[1:])))
        df.index = array[0]
        
        df.index.name = 'time'
        
        return df

    def get_df(self):
        """
        Returns a DataFrame based on the configuration settings.

        If the configuration is None, an empty DataFrame is returned.
        Otherwise, a DataFrame is returned based on the channels specified in the configuration.

        Returns:
            pandas.DataFrame: The resulting DataFrame.
        """
        if self.config is None:
            df = pd.DataFrame()
        else:
            df = self.get_channels(*self.config.keys())

        return df
    
    def get(self, *names):
        """
        Retrieve data from the data object.

        Parameters:
        *names: str
            Names of the data columns to retrieve.

        Returns:
        pandas.DataFrame or pandas.Series
            The requested data columns.

        Raises:
        None
        """
        if all(na in self.df.keys() for na in names):
            res = super().get(*names)
        elif len(names) == 1 and (names[0].lower() == 't' 
                                  or names[0].lower() == 'time' 
                                  or names[0].lower() == 'index'):
            if names[0] in self.df.keys():
                res = self.df[names]
            else:
                res = pd.DataFrame(self.t, index=self.t, columns= ['time'])
        else:
            res = self.get_channels(*names)
                                               
        return res

    def get_results(self, rows=None):
        """
        Retrieves the results from a file and returns them as a NumPy array.

        Parameters:
        - rows (list or None): Optional list of row indices to retrieve. If None, all rows are retrieved.

        Returns:
        - array (ndarray): NumPy array containing the retrieved results.
        """
        
        with open(self.path, "rb") as fobj:
        
            narray, = np.fromfile(fobj, dtype=np.dtype('i'), count=1)
            nvar, = abs(np.fromfile(fobj, dtype=np.dtype('i'), count=1))
            index = np.hstack([[0], np.fromfile(fobj, dtype=np.dtype('i'), count=nvar)+1])                   
            nvar = nvar + 1
            array = np.fromfile(fobj, dtype=np.dtype('d'), count=narray*nvar)
            array = array.reshape(narray, nvar).T
            
        if rows:
            rows = np.concatenate([np.where(index==r)[0] for r in rows])
                
        array = (array if rows is None 
                    else array[np.concatenate([[0], np.asarray(rows, dtype=int).ravel()])])
        
        return array
    
    def keys(self):
        
        if not len(self.df):
            
            res = self.channels
            
        else:
            
            res = list(self.df.keys())
        
        return res   
    
    def search_channel(self, patt):
        """
        Searches for channels in the `self.channels` list that match the given pattern.

        Parameters:
        patt (str): The pattern to search for.

        Returns:
        list: A list of channels that match the given pattern.
        """
        r = re.compile(patt)
        
        return list(filter(r.match, self.channels))