# -*- coding: utf-8 -*-

import datetime
import logging
import requests
import json
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#%%---------------------------------------------------------------------------#
URL_YAHOO = (
    'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?symbol={symbol}'
    '&period1={time_start}&period2={time_end}&interval={frequency}&'
    'includePrePost=true&events=div%7Csplit%7Cearn&lang=en-US&'
    'region=US&crumb=t5QZMhgytYZ&corsDomain=finance.yahoo.com')

FREQ_STRS = ['1m', '2m', '5m', '15m', '30m', '60m',
             '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

FREQ_ARR = np.array([1,
                     2,
                     5,
                     15,
                     30,
                     60,
                     90,
                     60,
                     60 * 24,
                     60 * 24 * 5,
                     60 * 24 * 7,
                     60 * 24 * 7 * 30,
                     60 * 24 * 7 * 30 * 3],
                    dtype=int)

#%%---------------------------------------------------------------------------#


class YahooFinanceError(Exception):

    def __init__(self, message):

        self.message = message

#%%---------------------------------------------------------------------------#


class FiDat(object):

    def __init__(self, symbol=None, days=None, freq=None,
                 start=None, end=None, js=None):

        self.symbol = symbol

        self.days = days

        if start is not None:
            start = start if isinstance(
                start, datetime.datetime) else datetime.datetime(
                *start)

        self.start = start

        if end is not None:
            end = end if isinstance(
                end, datetime.datetime) else datetime.datetime(
                *end)
        else:
            end = datetime.datetime.now()

        self.end = end

        if self.start is not None and self.end is not None:
            self.days = (self.end - self.start).days

        self.freq = freq  # default: 1 minute

        if js is None:
            js = requests.get(self.url).json()['chart']

        elif isinstance(js, str):

            js = self.load(js)

        self.json = js

        if self.json['error'] is not None:

            code = self.json['error']['code']
            description = self.json['error']['description']

            raise YahooFinanceError(f'{code}: {description}')

        self.timestamp = np.array(self.result['timestamp'], dtype=int)

        self.log = logging.Logger(None)

    #%%-----------------------------------------------------------------------#
    def __str__(self):

        symbol = self.result['meta']['symbol']
        start = datetime.datetime.fromtimestamp(
            self.timestamp.min()).strftime('%m/%d/%y')
        end = datetime.datetime.fromtimestamp(
            self.timestamp.max()).strftime('%m/%d/%y')

        size = self.timestamp.size

        return f'<"{symbol}", {size}, {start}-{end}>'

    __repr__ = __str__

    #%%-----------------------------------------------------------------------#
    @property
    def now(self):

        if self.end is None:

            return datetime.datetime.now()

        else:
            if self.end > datetime.datetime.now():

                self.log.warning('end time has been set to current time.')
                self.end = datetime.datetime.now()

            return self.end

    #%%-----------------------------------------------------------------------#
    @property
    def time_start(self):

        if self.start is not None:
            return int(self.start.timestamp())

        else:
            return int(
                (self.now - datetime.timedelta(days=self.days)).timestamp())

    #%%-----------------------------------------------------------------------#
    @property
    def time_end(self):

        return int(self.now.timestamp())

    #%%-----------------------------------------------------------------------#
    @property
    def frequency(self):

        if self.freq is None:
            self.freq = max(24 * 60 * self.days // 500, 1)

        return FREQ_STRS[np.abs(FREQ_ARR - self.freq).argmin()]

    #%%-----------------------------------------------------------------------#
    @property
    def url(self):

        return URL_YAHOO.format(
            symbol=self.symbol,
            time_start=self.time_start,
            time_end=self.time_end,
            frequency=self.frequency)

    #%%-----------------------------------------------------------------------#
    @property
    def result(self):

        return self.json['result'][0]

    #%%-----------------------------------------------------------------------#
    @property
    def time(self):

        return [datetime.datetime.fromtimestamp(x) for x in self.timestamp]

    #%%-----------------------------------------------------------------------#
    @property
    def open(self):

        return pd.Series(
            self.result['indicators']['quote'][0]['open'],
            index=self.timestamp,
            dtype=float,
            name='open').dropna()

    #%%-----------------------------------------------------------------------#
    @property
    def high(self):

        return pd.Series(
            self.result['indicators']['quote'][0]['high'],
            index=self.timestamp,
            dtype=float,
            name='high').dropna()

    #%%-----------------------------------------------------------------------#
    @property
    def low(self):

        return pd.Series(
            self.result['indicators']['quote'][0]['low'],
            index=self.timestamp,
            dtype=float,
            name='low').dropna()

    #%%-----------------------------------------------------------------------#
    @property
    def close(self):

        return pd.Series(
            self.result['indicators']['quote'][0]['close'],
            index=self.timestamp,
            dtype=float,
            name='close').dropna()

    #%%-----------------------------------------------------------------------#
    @property
    def volume(self):

        return pd.Series(
            self.result['indicators']['quote'][0]['volume'],
            index=self.timestamp,
            dtype=float,
            name='volume').dropna()

    #%%-----------------------------------------------------------------------#
    @property
    def dataframe(self):

        return pd.concat([self.open, self.high, self.low, self.close], axis=1)
    #%%-----------------------------------------------------------------------#
    @classmethod
    def index2date(self, s):

        s.index = [datetime.datetime.fromtimestamp(int(x)) for x in s.index]

        return s

    #%%-----------------------------------------------------------------------#
    def roundtime(self, intval):

        intval = intval * 60
        ts = self.timestamp

        scale = np.arange(self.time_start, self.time_end + intval, intval)

        for x in scale:
            mask = np.abs(ts - x) <= intval / 2
            ts[mask] = x

        self.timestamp = ts

        return scale

    #%%-----------------------------------------------------------------------#
    def grouptime(self, key, interval=None):

        if interval is None:
            interval = FREQ_ARR[FREQ_STRS.index(self.frequency)]

        interval *= 60  # second to minute

        arr = getattr(self, key)

        ts = arr.index
        tdiff = np.diff(ts)

        return np.split(arr, np.where(tdiff > interval)[0] + 1)

    #%%-----------------------------------------------------------------------#
    @classmethod
    def round2(self, key, periode):

        def get(ts, n):

            dt = datetime.datetime.fromtimestamp(ts)

            lst = [dt.year, dt.month, dt.day, dt.hour]

            dates = lst[:n]

            if len(dates) < 3:

                dates = dates + [1] * int(3 - len(dates))

            return datetime.datetime(*dates).timestamp()

        df = getattr(self, key) if isinstance(key, str) else key

        tbl = dict(y=1, m=2, d=3, h=4, year=1, month=2, day=3, hour=4)

        timestamps = np.fromiter(
            (get(ts, tbl[periode]) for ts in df.index), dtype=int)

        if isinstance(df, pd.Series):
            return pd.Series(df.values, index=timestamps)
        else:
            return pd.DataFrame(df.values, index=timestamps)

    #%%-----------------------------------------------------------------------#
    @classmethod
    def split(self, key, periode):

        def get():

            rounded = FiDat.round2(key, periode).index
            for ts in np.unique(rounded):

                yield df.loc[rounded == ts]

        df = getattr(self, key) if isinstance(key, str) else key

        return list(get())

    #%%-----------------------------------------------------------------------#
    def copy(self, **kwargs):

        defaults = dict(symbol=self.symbol, days=self.days,
                        freq=self.freq, end=self.end, json=self.json)
        defaults.update(kwargs)

        return FiDat(**defaults)

    #%%-----------------------------------------------------------------------#
    def plot(self, key=None, ax=None, fmt=None, **kwargs):

        if ax is None:

            _, ax = plt.subplots()

        data = getattr(self, key)

        if isinstance(data, pd.Series):

            default = dict(label=data.name)
            default.update(**kwargs)
            ax.plot(data, label=data.name, **kwargs)

        datetimes = [int(x) for x in ax.get_xticks().tolist()]

        if fmt is None:
            if np.diff(data.index).ptp() >= 86400:

                fmt = '%m/%d/%y'
            else:
                fmt = '%m/%d, %H:%M'

        xticklabels = [datetime.datetime.fromtimestamp(int(x)).strftime(fmt)
                       for x in datetimes]
        plt.xticks(datetimes, xticklabels, rotation=30)

        return ax
    #%%-----------------------------------------------------------------------#
    def dump(self, file):

        with open(file, 'w') as outfile:

            json.dump(self.json, outfile)

        return self
    #%%-----------------------------------------------------------------------#
    def load(self, file):

        with open(file, 'r') as jsonfile:

            js = json.load(jsonfile)

#        self.__init__(json=json)

        return js

#%%---------------------------------------------------------------------------#


if __name__ == '__main__':

    fd = FiDat(js='IFX.DE.txt')

#    dat = fd.split('close', 'day')

    print(fd.json)

#    print(dat)

#    fig, ax = plt.subplots()
#
#    ax.plot(fd.index2date(dat))
