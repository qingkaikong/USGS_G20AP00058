#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:38:38 2019

@author: qingkaikong
"""

import s2sphere
import pandas as pd
from collections import Counter

class S2Utils(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_cellId_from_lat_lon(lat, lon, level=11):
        """
        from lat, lon to cellId token
        """
        
        lat_lon = s2sphere.LatLng.from_degrees(lat, lon)
        cellId = s2sphere.CellId.from_lat_lng(lat_lon).parent(level)
        return cellId.to_token()
    
    def prepare_df_cell(self, steady_phones, df_trig, starttime_from_o=0, 
                        endtime_from_o=40, marker_duration=20, 
                        s2cell_threshold=0.1, lisening_threshold=10):
        
        cell_stats = []
        previousTriggering = 0
        
        # get the steady phone stats
        tokens = []
        for item in steady_phones:
            lat, lon = item
            token = self.get_cellId_from_lat_lon(lat, lon, level=11)
            tokens.append(token)
            
        cell_steady_count = Counter(tokens)
        
        for frame in range(starttime_from_o, endtime_from_o):
            
            # get the triggers within cells
            df_eq_trig_slice = \
                df_trig[(df_trig['delta_t']<frame) & \
                            (df_trig['delta_t']>frame - marker_duration)]
            
            tokens = []
            # loop through the triggers in cells
            for ix, row in df_eq_trig_slice.iterrows():
                lat = row['latitude']
                lon = row['longitude']
                
                token = self.get_cellId_from_lat_lon(lat, lon, level=11)
                tokens.append(token)

            cell_trig_count = Counter(tokens)
            
            
            for key in cell_trig_count.keys():
                numLisening = cell_steady_count[key]
                numTriggering = cell_trig_count[key]
                ratio = numTriggering/numLisening
                
                cellId = s2sphere.CellId.from_token(key)
                level = cellId.level()
                lat_lon = cellId.to_lat_lng()
                cell_lat = lat_lon.lat().degrees
                cell_lon = lat_lon.lng().degrees
                
                if numLisening > lisening_threshold:
                    if ratio > s2cell_threshold:
                        cell_stats.append([frame, numLisening, numTriggering, ratio, key, 
                                           level, cell_lat, cell_lon])
                    
        df_cell = pd.DataFrame(cell_stats, columns=['delta_t', 'numLisening', 'numTriggering', 
                                          'ratio', 's2cellToken', 
                                          's2cellLevel', 'lat', 'lon'])
                    
        return df_cell
                
                
            
    
#%%

