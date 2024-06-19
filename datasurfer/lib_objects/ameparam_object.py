#%% Import Libraries

import re
import pandas as pd
import xml.etree.ElementTree as ET
from datasurfer.datainterface import DataInterface

#%%
class AMEObject(DataInterface):

    @property
    def name(self):
        if self._name is None:
            
            assert self.path is not None, 'Expect a name for data object.'
            
            if self.ext_idx.isnumeric():
                return self.stem+self.ext_idx
            else:
                return self.stem
       
        else:
            return self._name
        
    @name.setter
    def name(self, value):  
        self._name = value    
           
    @property
    def stem(self):
        
        r = re.compile(rf'(.+)(\.{self.__class__.ametype}.*)')
        return r.match(self.path.name).group(1)
    
    @property
    def ext(self):
        
        r = re.compile(rf'(.+)(\.{self.__class__.ametype}.*)')
        return r.match(self.path.name).group(2)      
    
    @property
    def ext_idx(self):
        r = re.compile(rf'(.+)\.{self.__class__.ametype}[.]?(\d*)')  
 
        idxstr = r.match(self.path.name).group(2)    
        if len(idxstr):
            return idxstr
        else:
            return 'ref'   
#%%
class AMEPLObject(AMEObject):
    
    ametype = 'pl'

    @property
    def fhandler(self):
        """
        Returns the XML file handler for the current object.
        
        If the XML file handler has not been initialized yet, it will be initialized by parsing the XML file specified by the `path` attribute.
        
        Returns:
            The XML file handler for the current object.
        """
        if not hasattr(self, '_fhandler'):
            etobj = ET.parse(self.path)           
            checksum = etobj.getroot().attrib['checksum'] 
            try:         
                self._fhandler = ET.parse(self.path.parent / f'{self.stem}.pl.crc_{checksum}')  
            except FileNotFoundError:
                self._fhandler = etobj        
        return self._fhandler
    
    def df_gp(self):
        def get():
            root = ET.parse(self.path).getroot()
            for param in root.find('GLOBAL_PARAMS_LIST'):                
                attrs = param.attrib
                yield attrs['Data_Path'], attrs
        return pd.DataFrame.from_dict(dict(get()), orient='index').T
        

    def get_df(self):
        def get():
            for grp in self.fhandler.getroot():
                grpname = grp.tag.lower().replace('list', '').rstrip('_')      
                for param in grp:
                    attris = param.attrib
                    attris['GROUP'] = grpname                   
                    if 'Data_Path' in attris:                       
                        yield attris['Data_Path'], attris
        return pd.DataFrame.from_dict(dict(get()), orient='index').T

#%%  
class AMEParamObject(AMEObject):
    
    ametype = 'param'
    
    def __init__(self, path=None, name=None, comment=None):

        super().__init__(path, name=name, comment=comment) 
        
        self.data = AMEParamDataObject(str(self.path).replace('param', 'data'), name=name, comment=comment)
        
    def __setitem__(self, name, value):
        self.df.loc['VALUE', name] = value
    
    @property
    def fhandler(self):
        
        if not hasattr(self, '_fhandler'):
            self._fhandler = [l.strip() for l in open(self.path).readlines()]
                       
        return self._fhandler
    
    def get_df(self):
        
        def get():
            r = re.compile(r'(\S+)=(\S+)')
            for index, line in enumerate(self.fhandler):
                if len(line):
                    found = r.findall(line)
                    if found: 

                        attrs = dict(found)
                        attrs['Line_Index'] = index
                        attrs['VALUE'] = self.data.df.loc[index, 'VALUE']
                        
                        yield attrs['Data_Path'], attrs
                
        return pd.DataFrame.from_dict(dict(get()), orient='index').T
    
    def save_value(self, name=None):
        
        for lineidx, val in self.df.T[['Line_Index', 'VALUE']].set_index('Line_Index')['VALUE'].to_dict().items():
            self.data.df.loc[lineidx, 'VALUE'] = str(val)
            
        self.data.save(name=name)
        return self
    
#%%
class AMEParamDataObject(AMEObject):
    
    ametype = 'data'
    
    def get_df(self):
        
        return pd.DataFrame.from_dict({index: line.strip() for index, line 
                                       in enumerate(open(self.path).readlines())}, orient='index', columns=['VALUE'])
    
    def save(self, name=None):
        
        path = name or self.path  
        
        with open(path, 'w') as fobj:
            fobj.write('\n'.join(self.df['VALUE'].tolist()))
            
        return self
        
          
   
#%%
class AMEGPObject(AMEObject):  
    
    ametype = 'amegp'
    
    def __setitem__(self, name, value):
        self.df.loc['VALUE', name] = value
        
    @property
    def fhandler(self):
        """
        Returns the XML file handler for the current object.
        
        If the XML file handler has not been initialized yet, it will be initialized by parsing the XML file specified by the `path` attribute.
        
        Returns:
            The XML file handler for the current object.
        """
        if not hasattr(self, '_fhandler'):
            self._fhandler = ET.parse(self.path)
            
        return self._fhandler
    
    def get_df(self):
        """
        Returns a pandas DataFrame containing the data from the AMEGP object.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the AMEGP object.
        """
        out = dict()
        for param in self.fhandler.getroot():
            attris = dict()
            for attr in param:
                attris[attr.tag] = attr.text

            out[attris['VARNAME']] = attris

        df = pd.DataFrame(out) 
        return df
    
    def save(self, name=None):
        """
        Save the data to a file.

        Args:
            name (str, optional): The name of the file to save the data to. If not provided, the data will be saved to the object's `path` attribute.

        Returns:
            self: The current instance of the object.

        """
        path = name or self.path
        for param in self.fhandler.getroot():
            key = param.find('VARNAME').text
            for attr in param:
                attr.text = str(self.df[key][attr.tag])
        self.fhandler.write(path)
        return self

#%% AMEGP_OJBECT
class _AMEGPObject_(DataInterface):
    """
    Represents an object for handling AMEGP data.

    Args:
        path (str): The path to the data file.
        config (dict): Configuration settings for the object.
        name (str): The name of the object.
        comment (str): Additional comment for the object.

    Attributes:
        path (str): The path to the data file.
        config (dict): Configuration settings for the object.
        name (str): The name of the object.
        comment (str): Additional comment for the object.
        df (pd.DataFrame): The data stored in a pandas DataFrame.

    Methods:
        __init__(self, path=None, config=None, name=None, comment=None):
            Initializes a new instance of the AMEGPObject class.
        __setitem__(self, name, value):
            Sets the value of a specific item in the object.
        get_df(self):
            Retrieves the data as a transposed DataFrame.
        set_value(self, name, value):
            Sets the value of a specific item in the DataFrame.
        save(self, name=None):
            Saves the object's data to a file.

    """
    exts = ['.amegp']
    
    def __init__(self, path=None, name=None, comment=None):
        """
        Initializes a new instance of the AMEGPObject class.

        Args:
            path (str): The path to the data file.
            config (dict): Configuration settings for the object.
            name (str): The name of the object.
            comment (str): Additional comment for the object.

        """
        super().__init__(path, name=name, comment=comment)     
        
    @property
    def stem(self):
        
        r = re.compile(r'(.+)(\.amegp.*)')
        return r.match(self.path.name).group(1)
    
    @property
    def ext(self):
        
        r = re.compile(r'(.+)(\.amegp.*)')
        return r.match(self.path.name).group(2)      
    
    @property
    def ext_idx(self):
        r = re.compile(r'(.+)\.amegp[.]{0,1}(.*)')     
        return r.match(self.path.name).group(2) 
         
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

    
    def __setitem__(self, name, value):
        """
        Sets the value of a specific item in the object.

        Args:
            name (str): The name of the item.
            value: The value to be set.

        """
        self.set_value(name, value)
        
    def get_df(self):
        """
        Retrieves the data as a transposed DataFrame.

        Returns:
            pd.DataFrame: The data stored in a pandas DataFrame.

        """
        df = pd.read_xml(self.path)
        df.set_index('VARNAME', inplace=True)
        return df.transpose()
    
    def set_value(self, name, value):
        """
        Sets the value of a specific item in the DataFrame.

        Args:
            name (str): The name of the item.
            value: The value to be set.

        Returns:
            self: The updated AMEGPObject instance.

        """
        self.df.at['VALUE', name] = value
        return self
    
    def save(self, name=None):
        """
        Saves the object's data to a file.

        Args:
            name (str, optional): The name of the file to save the data to. If not provided, the original file will be overwritten.

        Returns:
            self: The AMEGPObject instance.

        """
        name = name or self.path
        
        with open(self.path, 'r') as fobj:
            lines = fobj.readlines()
        
        newlines = []
        
        for l in lines:
            if re.match(r'^<VARNAME>.+</VARNAME>$', l.strip()):
                varname, = re.findall(r'^<VARNAME>(.+)</VARNAME>$', l.strip())
                item = self.df[varname]
                
            if re.match(r'^<VALUE>.+</VALUE>$', l.strip()):  
                str_old, = re.findall(r'^<VALUE>.+</VALUE>$', l.strip())
                str_new = '<VALUE>{}</VALUE>'.format(item['VALUE'])
                l = l.replace(str_old, str_new, 1)
                    
            newlines.append(l)
            
        with open(name, 'w') as fobj:
            fobj.writelines(newlines)
            
        return self

#%%