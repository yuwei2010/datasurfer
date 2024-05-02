#%% Import Libraries

import re
import pandas as pd
import numpy as np
import warnings
import mdfreader
from datasurfer.datainterface import DataInterface
from datasurfer.datautils import translate_config, extract_channels

#%% MDFObject
class MDFObject(DataInterface):
    """
    Represents an MDF object that provides access to MDF file data.

    Args:
        path (str): The path to the MDF file.
        config (dict, optional): Configuration dictionary specifying the channels to extract. Defaults to None.
        sampling (float, optional): The sampling rate for resampling the data. Defaults to 0.1.
        name (str, optional): The name of the MDF object. Defaults to None.
        comment (str, optional): The comment associated with the MDF object. Defaults to None.

    Attributes:
        sampling (float): The sampling rate for resampling the data.
        fhandler: The MDF file handler.
        info: Information about the MDF file.
        comment: The comment associated with the MDF object.
        channels: The list of available channels in the MDF file.
        t: The time axis of the data.
    """
    exts = ['.mf4']
    def __init__(self, path, config=None, sampling=0.1, name=None, comment=None):
        super().__init__(path, config, comment=comment, name=name)
        self.sampling = sampling

    @property
    def fhandler(self):
        """
        Lazily initializes and returns the MDF file handler.

        Returns:
            Mdf: The MDF file handler.
        """

        if not hasattr(self, '_fhandler'):
            self._fhandler = mdfreader.Mdf(self.path, no_data_loading=True)
        return self._fhandler

    @property
    def info(self):
        """
        Returns information about the MDF file.

        Returns:
            dict: Information about the MDF file.
        """
        return self.fhandler.info

    @property
    def comment(self):
        """
        Returns the comment associated with the MDF object.

        Returns:
            dict: The comment associated with the MDF object.
        """
        if self._comment is None:
            cmmt = self.info['HD']['Comment']
            if cmmt.get('TX') is not None:
                txt = dict(re.findall(r'(.+):\s+(.+)', cmmt.get('TX')))
                if txt:
                    cmmt.update(txt)
                else:
                    cmmt['comment'] = txt
            self._comment = cmmt
        return self._comment

    @property
    def channels(self):
        """
        Returns the list of available channels in the MDF file.

        Returns:
            list: The list of available channels.
        """
        return sorted(set(self.fhandler.keys()))

    @property
    def t(self):
        """
        Returns the time axis of the data.

        Returns:
            ndarray: The time axis.
        """
        if self.config is None:
            warnings.warn('Time axis may have deviation due to missing configuration.')
        if not len(self.df):
            df = self.get_channels(self.channels[0])
            t = df.index
        else:
            t = self.df.index
        return np.asarray(t)

    def keys(self):
        """
        Returns the list of keys.

        Returns:
            list: The list of keys.
        """
        if not hasattr(self, '_df'):
            res = self.channels
        else:
            res = list(self.df.keys())
        return res
    list_signals = keys
    def search_channel(self, patt):
        """
        Searches for channels that match the given pattern.

        Args:
            patt (str): The pattern to search for.

        Returns:
            list: The list of channels that match the pattern.
        """
        r = re.compile(patt)
        return list(filter(r.match, self.channels))

    @translate_config()
    @extract_channels()
    def get_channels(self, *channels):
        """
        Extracts the specified channels from the MDF file.

        Args:
            *channels: The channels to extract.

        Returns:
            DataFrame: The extracted channels as a DataFrame.
        """
        def get(chn):
            mname = mdfobj.get_channel_master(chn)
            if mname is None:
                raise ValueError
            dat = mdfobj.get_channel(chn)['data']
            t = mdfobj.get_channel(mname)['data']
            return pd.Series(dat, index=t-t.min(), name=chn)

        mdfobj = mdfreader.Mdf(self.path, channel_list=channels)
        if self.sampling is not None:
            mdfobj.resample(self.sampling)
        outs = []
        for chn in channels:
            try:
                res = get(chn)
                outs.append(res)
            except ValueError:
                raise
                warnings.warn(f'Channel "{chn}" not found.')
        return pd.concat(outs, axis=1)

    def get_df(self):
        """
        Returns the DataFrame representation of the MDF object.

        Returns:
            DataFrame: The DataFrame representation of the MDF object.
        """
        if self.config is None:
            df = pd.DataFrame()
        else:
            df = self.get_channels(*self.config.keys())
        return df

    def get(self, *names):
        """
        Returns the specified data from the MDF object.

        Args:
            *names: The names of the data to retrieve.

        Returns:
            DataFrame: The retrieved data as a DataFrame.
        """
        if all(na in self.df.keys() for na in names):
            res = self.df[list(names)]
        elif len(names) == 1 and (names[0].lower() == 't' or names[0].lower() == 'time' or names[0].lower() == 'index'):
            if names[0] in self.df.keys():
                res = self.df[names]
            else:
                res = pd.DataFrame(self.t, index=self.t, columns=['time'])
        else:
            res = self.get_channels(*names)
        return res