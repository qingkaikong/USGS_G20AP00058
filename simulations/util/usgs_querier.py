#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:29:03 2019

Class to query USGS events

@author: Qingkai Kong
"""

import requests
import csv
import pandas as pd

class EQ_from_USGS(object):
    
    """Query USGS to get EQ info."""
    BASE = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&'
    
    def __init__(self, url=None):
        # pass your query url directly
        self.url = url

    @classmethod
    def from_evid(cls, evid):
        url = cls.BASE + 'eventid=' + evid
        return cls(url)
    
    @classmethod
    def from_time_range(cls, t0, t1, min_mag=4, max_mag=9):
        url = cls.BASE + 'starttime=%s&endtime=%s&minmagnitude=%d&maxmagnitude=%d'%\
             (t0, t1, min_mag, max_mag)
        return cls(url)
    
    @classmethod
    def from_time_space_rect(cls, t0, t1, llat, ulat, llon, ulon, \
                              min_mag=4, max_mag=9):
        url = cls.BASE + 'starttime=%s&endtime=%s&minlatitude=%f&maxlatitude=%f&minlongitude=%f&maxlongitude=%f&minmagnitude=%d&maxmagnitude=%d'%\
             (t0, t1, llat, ulat, llon, ulon, min_mag, max_mag)
        return cls(url)
    
    @classmethod
    def from_time_space_circle(cls, t0, t1, lat, lon, radius_km, \
                              min_mag=4, max_mag=9):
        url = cls.BASE + 'starttime=%s&endtime=%s&latitude=%f&longitude=%f&maxradiuskm=%f&minmagnitude=%d&maxmagnitude=%d'%\
             (t0, t1, lat, lon, radius_km, min_mag, max_mag)
        return cls(url)
    

    def _query_usgs(self, url):
        # request event information from usgs by using the event id
        r = requests.get(url)
        text = (line.decode('utf-8') for line in r.iter_lines())
        cr = list(csv.reader(text))
        # the first row will be header
        # data is in the 2nd row
        return cr

    def get_event(self):
        # query USGS
        event_info = self._query_usgs(self.url)
        # header: ['time', 'latitude', 'longitude', 'depth', 'mag', 'magType',
        #          'nst', 'gap', 'dmin', 'rms', 'net', 'id', 'updated', 'place',
        #          'type', 'horizontalError', 'depthError', 'magError', 'magNst',
        #          'status', 'locationSource', 'magSource']
        earthquake = pd.DataFrame(event_info[1:], columns=event_info[0])
        
        return earthquake
    
if __name__ == '__main__':
    usgs_querier = EQ_from_USGS.from_evid('us70005prt')
    print(usgs_querier.get_event())

    llat = 36
    ulat = 38
    llon = -125
    ulon = -121
    
    usgs_querier = EQ_from_USGS.from_time_space_rect('2014-01-01', '2016-02-01', llat, ulat, llon, ulon)
    print(usgs_querier.get_event())
    
    usgs_querier = EQ_from_USGS.from_time_space_circle('2014-01-01', '2016-02-01', llat, llon, 1000)
    print(usgs_querier.get_event())
    print(usgs_querier.get_event().keys())