#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 07:43:18 2020

@author: qingkaikong
"""

# %% import
import glob
import os
import pickle

import numpy as np

from network_detector import mgrs_dbscan_detector

# %% setup
rf_estimator_p = pickle.load(
  open('./data/estimate_magnitude_random_forest_M3pt5_to_M9_with_Uncertainty_P.pkl', 'rb'))
rf_estimator_s = pickle.load(
  open(
    './data/estimate_magnitude_random_forest_M3pt5_to_M9_with_Uncertainty_S.pkl', 'rb'))


# %% 

simulation_folder = '../data/simulation_data_pt1p/'
usgs_event_folder = '../data/ground_truth/'
use_old_myshake_detection = True

shakealert_trig_file = '../data/shake_alert_trigger_dict_from_Ivan.pkl'

shakealert_trig_dict = pickle.load(open(shakealert_trig_file, 'rb'))

error_dict = {}
for simulation_file in glob.glob(os.path.join(simulation_folder, '*')):
  detected_events_withEvents = []
  evid = simulation_file.split('/')[-1].split('_')[0]
  
  # get the trigger information from ShakeAlert
  df_shakealert_trig = shakealert_trig_dict.get(evid, None)
    
  if evid not in error_dict.keys():
      error_dict[evid] = []
  
  ground_truth_file = glob.glob(os.path.join(usgs_event_folder, evid + '*'))[0]
  earthquake = pickle.load(open(ground_truth_file, 'rb'))
    
  mgrs_dbscan = mgrs_dbscan_detector(evid, earthquake, rf_estimator_p, rf_estimator_s)
    
  evla = earthquake['latitude']
  evlo = earthquake['longitude']
  
  llat = evla - 1.5
  ulat = evla + 2.4
  llon = evlo - 2.8
  ulon = evlo + 0.4
  
  simulation_dict = pickle.load(open(simulation_file, 'rb'))
  phones_steady = np.array(simulation_dict[0]['phones_steady'])
  df_trig = simulation_dict[0]['df_trig']
  
  if use_old_myshake_detection:
    # since the pga right now is a list, to use the old code, we only need the 
    # pga at the trigger time. Here we need to divide the amplificaiton factor
    # used to generate the triggers, because the magnitude estimation is trained
    # on without using amplification factor.
    df_trig['pga'] = [row['pga'][0][1]/9.81/2.0 for ix, row in df_trig.iterrows()]
  
    # the errors contains [mag_error, dist_error, originT_error, alertT_from_origin]
    detected_events_withEvents, df_trig_eq, df_heartbeat_eq, errors, triggers_used_for_location, error_tracking = mgrs_dbscan.detector_with_updates(
      phones_steady, df_trig, nupdate=0, include_shakealert_trig = True, df_shakealert_trig)
    
    error_dict[evid].append([detected_events_withEvents, errors])

# %%
pickle.dump(error_dict, open('../data/myshake_detection_results.pkl', 'wb'))