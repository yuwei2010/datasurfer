#%% Import Libraries
import re
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path

from datasurfer.datainterface import DataInterface 
from datasurfer.datautils import translate_config, extract_channels

#%%
class AMEVLObject(DataInterface):
    """
    Represents an AMEVL object.

    Args:
        path (str): The path to the object.
        name (str, optional): The name of the object. Defaults to None.
        comment (str, optional): Any additional comment for the object. Defaults to None.
    """

    def __init__(self, path, name=None, comment=None):
        super().__init__(path, name=name, comment=comment)

    @property
    def stem(self):
        """
        Returns the stem of the object's path.

        Returns:
            str: The stem of the object's path.
        """
        r = re.compile(r'(.+)(\.vl\..*)')
        return r.match(self.path.name).group(1)

    @property
    def name(self):
        """
        Returns the name of the object.

        If the name is not set, it returns the stem of the object's path.

        Returns:
            str: The name of the object.
        """
        if self._name is None:
            assert self.path is not None, 'Expect a name for data object.'
            return self.stem
        else:
            return self._name

    @name.setter
    def name(self, value):
        """
        Sets the name of the object.

        Args:
            value (str): The name to set.
        """
        self._name = value

    def get_df(self):
        """
        Parses the XML file and returns a DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame containing the parsed data.
        """
        root = ET.parse(self.path).getroot()
        out = dict()
        varlst = root.find('VARS_LIST')

        for var in varlst:
            attrs = var.attrib

            if 'Data_Path' in attrs:
                attrs['PARENT'] = None
                out[attrs['Data_Path']] = attrs

                tievar = var.find('TIE_VAR')
                if tievar is not None:
                    attrs = tievar.attrib
                    attrs['PARENT'] = var.attrib['Data_Path']
                    out[attrs['Data_Path']] = attrs

        df = pd.DataFrame(out)
        return df
#%%    
class AMESSFObject(DataInterface):
    
    def __init__(self, path, name=None, comment=None):
                
        super().__init__(path, name=name, comment=comment) 
        
    
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
class AMEResObject(DataInterface):
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

    exts = ['.results']

    def __init__(self, path, config=None, name=None, comment=None, **kwargs):
                
        super().__init__(path, config=config, name=name, comment=comment) 
        f_amevl = kwargs.pop('amevl', None)     

        if f_amevl is None:
            try:
                r = re.compile(rf'{self.stem}.vl.crc_\d+')
                f_amevl, = filter(r.match,  [s.name for s in self.path.parent.glob('*')])
                f_amevl = self.path.parent / f_amevl
            except ValueError:
                f_amevl = None
                
        if f_amevl:            
            self.vl = AMEVLObject(f_amevl, name=name, comment=comment)
        else:
            self.vl = None       
        
        f_amessf = kwargs.pop('amessf', None)
        
        if f_amessf is None:       
            f_amessf = self.path.parent / (self.stem+'.ssf.'+self.ext_idx).rstrip('.')  
            
        if Path(f_amessf).is_file():  
            self.ssf = AMESSFObject(f_amessf, name=name, comment=comment) 
        else:
            self.ssf = None 
        
    @property
    def name(self):
        if self._name is None:
            
            assert self.path is not None, 'Expect a name for data object.'
            return self.stem+self.ext_idx
        else:
            return self._name
        
    @name.setter
    def name(self, value):  
        self._name = value    
        
    @property
    def params(self):
        """
        Parses a parameter file and returns a dictionary of parameter information.

        Returns:
            dict: A dictionary containing parameter information. The keys are the data paths and the values are dictionaries
                  containing the parameter details such as 'Data_Path', 'Param_Id', 'Unit', 'Label', 'Description', and 'Row_Index'.
        """
        
        
        fparam = self.path.parent / (self.stem+'.ssf.'+self.ext_idx).rstrip('.')
        out = dict()

        with open(fparam, 'r') as fobj:
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

        return out
    
    @property
    def stem(self):
        
        r = re.compile(r'(.+)(\.results.*)')
        return r.match(self.path.name).group(1)
    
    @property
    def ext(self):
        
        r = re.compile(r'(.+)(\.results.*)')
        return r.match(self.path.name).group(2)      
    
    @property
    def ext_idx(self):
        r = re.compile(r'(.+)\.results[.]{0,1}(.*)')     
        return r.match(self.path.name).group(2)    
          
    
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
    def channels(self):
        
        return sorted(v['Data_Path'] for v in self.params.values())


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
        
        row_indices = [params[c]['Row_Index'] for c in channels]
        
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
            _ = np.hstack([[0], np.fromfile(fobj, dtype=np.dtype('i'), count=nvar)+1])                        
            nvar = nvar + 1
            array = np.fromfile(fobj, dtype=np.dtype('d'), count=narray*nvar)
            array = array.reshape(narray, nvar).T
                
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