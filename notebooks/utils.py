#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:40:51 2020

@author: qingkaikong
"""

import json
import datetime
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats

from geo_utils import calculate_dist


def prepare_myshake_data(earthquake,
                         orig_time,
                         df_trig,
                         df_heartbeat,
                         df_epic,
                         t_start,
                         t_end,
                         max_dist_km=100,
                         myshake_legacy=False,
                         ):
  """function to prepare myshake data. 
  
  Args:
    earthquake:
    orig_time:
    df_trig:
    df_heartbeat:
    df_epic:
    t_start:
    t_end:
    max_dist_km:
    myshake_legacy:
    
  Returns:
    results: tuple of:
      df_trig:
      df_heartbeat:
      df_heartbeat_unique:
      df_stalta_trig:
      df_steady:
      df_epic:
    
  """
  ################################ Triggers ################################
  # slice the data between t_start and t_end
  df_trig = df_trig[t_start:t_end]
  
  # get the trigger relative time
  df_trig = df_trig.assign(tt_rel=df_trig['tt']/1000 - orig_time)
  
  # get the trigger distance from epicenter in km
  df_trig = df_trig.assign(dist_km=[
    calculate_dist(earthquake['latitude'],
                   earthquake['longitude'],
                   row['l']['coordinates'][1],
                   row['l']['coordinates'][0])
    for ix, row in df_trig.iterrows()
    ])
  
  # triggers within max_dist km
  df_trig = df_trig[df_trig['dist_km']<max_dist_km]

  ################################ Heartbeat ################################
  # prepare the heartbeat data
  heartbeat_dist = []
  if myshake_legacy:
    
    # get the heartbeat distance km
    heartbeat_dist = [
        calculate_dist(earthquake['latitude'],
                       earthquake['longitude'],
                       row['latitude'],
                       row['longitude'])
        for ix, row in df_heartbeat.iterrows()
        ]
    
    deviceId = 'deviceId'
    ts = 'ts'
  else:
    # remove the heartbeat that is not contain the Numerical propertise
    df_heartbeat = df_heartbeat[df_heartbeat['numericProperties'] != 'null']
    hbSource = []
    for ix, row in df_heartbeat.iterrows():
        dist = calculate_dist(
            earthquake['latitude'],
            earthquake['longitude'],
            row['location_latitude'],
            row['location_longitude'])
        heartbeat_dist.append(dist)
        # get the hbSource
        hbSource.append(
          json.loads(row['numericProperties']).get('heartbeatSource', -12345))
    # add hbSource to the heartbeat
    df_heartbeat = df_heartbeat.assign(hbSource=hbSource)
    deviceId = 'deviceId_value'
    ts = 'deviceTimestamp'
  
  # get the distance
  df_heartbeat = df_heartbeat.assign(dist_km=heartbeat_dist)
  # heartbeat within max_dist km
  df_heartbeat = df_heartbeat[df_heartbeat['dist_km'] < max_dist_km]
  
  # get heartbeat relative time
  df_heartbeat = df_heartbeat.assign(tt_rel=df_heartbeat[ts]/1000 - orig_time)
  
  # get the unique phones, sta_lta triggers and steady phones
  # unique phones
  df_heartbeat_unique = df_heartbeat[:earthquake['time'][:-5]].drop_duplicates(deviceId)
  # sta_lta triggers
  df_stalta_trig = df_heartbeat[df_heartbeat['hbSource'] == 3][t_start:t_end]
  # steady phones
  df_steady = df_heartbeat[:earthquake['time'][:-5]]
  count_fist = False
  if count_fist:
    # this it how we get the steady phones, we got all the steady heartbeats
    # only keep the last one, but this way, we may have phones already triggered
    # but still count as steady
    df_steady = df_steady[df_steady['hbSource'] == 1]
    df_steady = df_steady.drop_duplicates(deviceId, keep='last')
  else:
    # the other way to get the steady phones, we first only keep the last heartbeat 
    # for each unique phone, and if it is steady, then we count them, but phones
    # steady for longer time will be excluded, since they may send another heartbeat
    # I think this is a more correct way to count
    df_steady = df_steady.drop_duplicates(deviceId, keep='last')
    df_steady = df_steady[df_steady['hbSource'] == 1]
    
  ############################### EPIC triggers ##############################
  # only keep epic triggers from t_start to t_end
  df_epic = df_epic[t_start:t_end]
  # get the relative time and distance for the epic triggers
  df_epic = df_epic.assign(tt_rel= df_epic['time'] - orig_time)
  df_epic = df_epic.assign(
      dist_km=[calculate_dist(
          earthquake['latitude'],
          earthquake['longitude'],
          row['lat'], row['lon'])
                for ix, row in df_epic.iterrows()
              ])
  df_epic = df_epic[df_epic['dist_km'] < max_dist_km]
    
  results = (df_trig,
             df_heartbeat,
             df_heartbeat_unique,
             df_stalta_trig,
             df_steady,
             df_epic)
  
  return results


def slice_time_from_eq_origin(earthquake,
                              start_time_s_from_origin,
                              end_time_s_from_origin):
  """function to prepare time to slice the data. 
  
  Args:
    earthquake: earthquake dictionary.
    start_time_s_from_origin: start time in sec respect to EQ. origin. 
    end_time_s_from_origin: end time in sec respect to EQ. origin.
    
  Returns:
    orig_time: unix timestamp for origin time
    t_start: start time string in format %Y-%m-%dT%H:%M:%S
    t_end: end time string in format %Y-%m-%dT%H:%M:%S
  
  """
  
  evtime_dt = datetime.datetime.strptime(
    earthquake['time'].replace('Z',''), '%Y-%m-%dT%H:%M:%S.%f')
  ############################## Setup Time ##############################
  # origin time in timestamp
  orig_time = (
      evtime_dt - datetime.datetime(1970, 1, 1)
              ).total_seconds()
  
  # max_time_s sec after the EQ
  t_end = (
    evtime_dt + datetime.timedelta(
            seconds=end_time_s_from_origin)).strftime('%Y-%m-%dT%H:%M:%S')
  
  # min_time_s before the EQ
  t_start = (
    evtime_dt + datetime.timedelta(
            seconds=start_time_s_from_origin)).strftime('%Y-%m-%dT%H:%M:%S')
  
  return orig_time, t_start, t_end


def get_spatial_statistics(values, bins):
    try:
        stats_sum, edge, _ = stats.binned_statistic(values, [1]*len(values), 
                       statistic='sum', bins=bins)
    except:
        stats_sum = np.zeros(len(bins[1:]))
        
    return stats_sum


def get_p_and_s(max_dist_km, evdp):
    
    ep_dist_km = np.arange(max_dist_km)
    
    hyp = np.sqrt(ep_dist_km**2 + evdp**2)
    
    return ep_dist_km, hyp/6.10, hyp/3.55


def select_data_between_time_and_space(df,
                                       t_start=0,
                                       t_end=60,
                                       max_dist_km=300):
    df = df[
        df['dist_km'] < max_dist_km]
    df = df[
        (df['tt_rel'] <= t_end) &
        (df['tt_rel'] > 0 )]
    return df


def get_stats(data):
    mu = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    
    results = {'mu':mu,
               'median':median,
               'std':std}
    return results


def simulation_data_count(simulation_dict,
                          alert_time,
                          evla,
                          evlo):
    
    p_wave_distance_at_alert_time = alert_time * 6.10
    
    steady_phone_list = []
    trig_list = []
    for key, simulation in simulation_dict.items():
        count = 0
        steady_phone = simulation['phones_steady']
        for stla, stlo in steady_phone:
            dist = calculate_dist(
              evla,
              evlo,
              stla,
              stlo)
            if dist <= p_wave_distance_at_alert_time:
                count += 1
                
        steady_phone_list.append(count)
        
        df = select_data_between_time_and_space(
            simulation['df_trig'],
            t_start=0,
            t_end=alert_time,
            max_dist_km=p_wave_distance_at_alert_time)
        trig_list.append(len(df))
    
    steady_phone_stats = get_stats(steady_phone_list)
    trig_stats = get_stats(trig_list)
    return steady_phone_stats, trig_stats

def convert_string_to_df(out, include_region):
    if include_region:
        df = pd.read_csv(StringIO('\n'.join(out)),
                 delim_whitespace=True, 
                 names=['E', 'id', 'evid', 'mag', 'evla', 'evlo',
                          'date', 'time', 'system', 'alert_time', 'est_mag',
                          'est_evla', 'est_evlo', 'est_date', 'est_time', 'region'])
    else:
        df = pd.read_csv(StringIO('\n'.join(out)),
                 delim_whitespace=True, 
                 names=['E', 'id', 'evid', 'mag', 'evla', 'evlo',
                          'date', 'time', 'system', 'alert_time', 'est_mag',
                          'est_evla', 'est_evlo', 'est_date', 'est_time'])
    return df

def convert_trig_string_to_df(trig_string):
    df_trig = pd.read_csv(StringIO('\n'.join(trig_string)),
              delim_whitespace=True, 
             names=['T', 'id', 'ver', 'trig_num', 'sta', 'chan',
                      'net', 'loc', 'date', 'time', 'lat', 'lon', 'sps',
                      'rsp', 'tt', 'dist_km', 'pd', 'pv', 'pa', 'pdmag'])
    
    df_trig['trig_timestamp_s'] = ((pd.to_datetime(df_trig['date'] + 'T' + df_trig['time']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms'))/1000.
    return df_trig

def read_alert_data(filename, include_region=True):
  rd = open(filename, "r")
  out = []
  trig_dict = {}
  init = 0
  while True:
      # Read next line
      line = rd.readline()
      # If line is blank, then you struck the EOF
      if len(line) > 0:
          if line[:2] == 'E:':
              if init:
                  df_trig = convert_trig_string_to_df(trig_string)
                  trig_dict[evid] = df_trig
              
              detection_string = line.strip()
              out.append(detection_string)
              df_tmp = convert_string_to_df([detection_string], include_region)
              evid = df_tmp.iloc[0]['evid']
              trig_string = []
              init += 1
          elif line[:2] == 'T:':
              trig_string.append(line.strip())

      if not line :
          break
  
  # This is the capture the last trigger sequence
  df_trig = convert_trig_string_to_df(trig_string)
  trig_dict[evid] = df_trig
  df = convert_string_to_df(out, include_region)
  return df, trig_dict