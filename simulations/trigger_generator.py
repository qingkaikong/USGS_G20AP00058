#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:07:34 2019

Class to generate trigger from an earthquake

@author: Qingkai Kong
"""

import numpy as np
import random
from datetime import datetime, timedelta
import pickle
import pandas as pd
from scipy.stats import halfnorm
import pytz
from geopy.distance import vincenty

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from util.cua_relationships import envelope_amplitude, envelope_generator
from util.geo_utils import calculate_dist

class trigger_generator(object):
    '''
    Class to generate MyShake triggers based on different methods
    '''

    def __init__(self, event_info, steady_ratio, phones, trig_rate, myshake_network_latency=None):

        '''
        steady_ratio is a dictionary from MyShake data
        '''

        self.evla = event_info['latitude']
        self.evlo = event_info['longitude']
        self.evdp = event_info['depth']
        self.mag = event_info['mag']
        self.evid = event_info['id']
        self.evtime = event_info['time'][:-1]
        self.evtime_ts = datetime.strptime(self.evtime, '%Y-%m-%dT%H:%M:%S.%f')
        self.steady_ratio = steady_ratio
        self.phones = phones
        self.trig_rate = trig_rate
        self.myshake_network_latency = myshake_network_latency

    def __get_background_trig_rate__(self, starttime, timezone):
        if timezone == 'UTC':
            dt_local = starttime
        else:
            utc = pytz.timezone('UTC')
            dt_utc = utc.localize(starttime)

            dt_local = dt_utc.astimezone(pytz.timezone(timezone))
        hour = dt_local.hour
        mu, std = self.trig_rate[hour]

        trig_rate = np.random.normal(mu, std)

        return trig_rate

    def __get_trig_rate__(self, starttime, timezone):
        if timezone == 'UTC':
            dt_local = starttime
        else:
            utc = pytz.timezone('UTC')
            dt_utc = utc.localize(starttime)

            dt_local = dt_utc.astimezone(pytz.timezone(timezone))
        hour = dt_local.hour
        m_trig, s_trig, m_stalta, s_stalta = self.trig_rate[hour]

        ann_rate = np.random.normal(m_trig, s_trig)
        stalta_rate = np.random.normal(m_stalta, s_stalta)

        return ann_rate, stalta_rate


    def __generate_random_triggers__(self, starttime, start_time_sec_from_origin, phones_steady, phone_thresholds, time_win, evla = None, evlo = None, timezone = 'UTC', n_sec=10, ANN_trig=False):

        '''
        Function to generate random triggers based on the MyShake trigger rate

        starttime - startime of the window, should be a datetime object, starttime = datetime.strptime('2018-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
        phones_steady - a list of steady phones
        time_win - how long we generate these triggers, in seconds
        '''

        ann_rate, stalta_rate = self.__get_trig_rate__(starttime, timezone)

        if ANN_trig:
            trig_rate = ann_rate
        else:
            trig_rate = stalta_rate

        # sometimes trig_rate is less than 0, this is due to the sampling using mean and std, therefore, I force it to zero
        if trig_rate < 0:
            trig_rate = 0
            return []
        n_steady_phones = len(phones_steady)
        n_phones = int(n_steady_phones * trig_rate)
        # generate the random time for this seconds
        random_time = np.random.randint(0, 3600, n_phones)

        phones_select = random_time[random_time <= time_win]
        n_select = len(phones_select)

        if n_select > n_steady_phones:
            print("There are more generated triggers than all the phones in the region")
        
        trig_dict = {}
        phone_ids = np.random.choice(range(n_steady_phones), size=n_phones, replace=False)
        for ix, shift_time in zip(phone_ids, phones_select):

            trig_time = starttime + timedelta(seconds = float(shift_time))
            stla, stlo = phones_steady[ix]

            if evla is None:
                dist = 0
            else:
                dist = calculate_dist(stla, stlo, evla, evlo)

            threshold = phone_thresholds[ix]

            pga_list = []
            for i in range(n_sec):
                tt = trig_time.timestamp() + i
                pga_p = (np.random.random()+0.005)*9.8
                pga_list.append([tt, pga_p])
            # p or s value is random
            ps_val = np.random.randint(2)
            #  earthquake trigger or not
            earthquake_trig = 0

            trig_dict[ix] = [stla, stlo, dist, trig_time, shift_time - start_time_sec_from_origin, pga_list, threshold, ps_val, earthquake_trig]

        return trig_dict

    def __get_steady_percentage__(self, timezone, starttime):

        if timezone == 'UTC':
            dt_local = starttime
        else:
            utc = pytz.timezone('UTC')
            dt_utc = utc.localize(starttime)

            dt_local = dt_utc.astimezone(pytz.timezone(timezone))
        hour = dt_local.hour
        m, s = self.steady_ratio[hour]

        # Steady percentage is determined by the mean and standard deviation from the MyShake one year data
        steady_percentage = np.random.normal(m, s)

        # for some cases, the percentage will be larger than 1, then we will just use 1 instead
        if steady_percentage > 1:
            steady_percentage = 1.0
        return steady_percentage

    def __get_the_steady_phones__(self, timezone, starttime):

        steady_percentage = self.__get_steady_percentage__(timezone, starttime)
        phones_steady = random.sample(self.phones.tolist(), int(steady_percentage * len(self.phones)))

        return steady_percentage, phones_steady

    def get_p_or_s_label(self, dist, evdp, shift_time, v_p, v_s, p_or_s_percentage):
        hypo_dist_km = np.sqrt(dist**2 + evdp**2)
        t_p = hypo_dist_km / v_p
        t_s = hypo_dist_km / v_s

        p_t_diff = np.abs(t_p - shift_time)
        s_t_diff = np.abs(t_s - shift_time)
        
        # We check the trigger time is close to P or S.
        if p_t_diff < s_t_diff:
            p_flag = 1
        else:
            p_flag = 0
        # We keep the correct label about p_or_s_percentage.
        if p_or_s_percentage > np.random.random():
            # this is wrong labeling, we flip it.
            p_flag = abs(p_flag - 1)

        return p_flag

    def generate_triggers_envelope(self, config, 
                                   timezone = 'UTC', 
                                   random_trigger = True, 
                                   time_win = 240,
                                   start_time_sec_from_origin=30,
                                   plot=False):
        # get all the configurations
        v_p = config['v_p']
        v_s = config['v_s']
        # this tries to capture the abormal cases that the phone not trigger
        # how to quantify this parameter?
        discount_factor = config['discount_factor']
        amplification_factor = config['amplification_factor']
        time_after_decay = config['time_after_decay']
        sampling_rate = config['sampling_rate']
        p_or_s_percentage = config['p_or_s_percentage']

        # we need to get the monitoring phones in local time
        steady_percentage, phones_steady = self.__get_the_steady_phones__(timezone, self.evtime_ts)
        
        # generate thresholds
        phone_thresholds = np.random.normal(config['phone_threshold_g_mean'], config['phone_threshold_g_std'], size=len(phones_steady))
        
        phone_thresholds[phone_thresholds < config['phone_threshold_g_bottom']] = config['phone_threshold_g_bottom']

        env_gen = envelope_generator()
        trig_list = []
        
        # add random triggering here, but the triggering rate should be 
        # time dependent or even region dependent
        if random_trigger:
            starttime = self.evtime_ts - timedelta(seconds = start_time_sec_from_origin)
            random_trigger_dict = self.__generate_random_triggers__(starttime, start_time_sec_from_origin, phones_steady, phone_thresholds, time_win, evla = self.evla, evlo = self.evlo, timezone = timezone)
        else:
            random_trigger_dict = {}

        for i, loc, threshold in zip(range(len(phones_steady)), phones_steady, phone_thresholds):
            
            # check if it is already triggered randomly
            if random_trigger_dict:
                random_trig = random_trigger_dict.get(i, -12345)
                
                # if there is no random trigger
                if random_trig == -12345:
                    pass
                else:
                    # if there is a random trigger
                    trig_list.append(random_trig)
                    continue
            else:
                pass
            
            stla, stlo = loc
            dist = calculate_dist(stla, stlo, self.evla, self.evlo)
            RS = random.sample(['R', 'S'], 1)[0]

            tp, p = env_gen.calc_envelope(self.mag, dist, self.evdp, v_p, v_s, 'PGA', 'H', 'P', RS, sampling_rate, time_after_decay)
            ts, s = env_gen.calc_envelope(self.mag, dist, self.evdp, v_p, v_s, 'PGA', 'H', 'S', RS, sampling_rate, time_after_decay)

            n_p = len(p)
            n_s = len(s)
            t = tp
            if n_p > n_s:
                N = n_p - n_s
                s = np.pad(s, (0, N), 'constant')
                t = tp
            elif n_p < n_s:
                N = n_s - n_p
                p = np.pad(p, (0, N), 'constant')
                t = ts

            # unit is m/s2
            c = (p + s) * amplification_factor
            
            trig_ix = c/9.81 > threshold
            
            # This is the trigger time since origin time.
            t_select = t[trig_ix]
            
            # if there are detections
            if len(t_select) > 0:
                # This is the first trigger time since origin time.
                shift_time = t_select[0]
                trig_poz = np.arange(len(t))[trig_ix][0]
                
                trig_time = self.evtime_ts + timedelta(seconds = shift_time)
                
                # get the trig PGA list (every sec)
                pga_list = []
                for s in np.arange(trig_poz-sampling_rate, 
                                   len(t) - sampling_rate, 
                                   sampling_rate):
                    s = int(s)

                    # sometimes, the trigger is very early, less than sampling_rate, therefore
                    # I need to add this to make sure in this case it won't crash. 
                    if s < 0:
                        s = 0
                    pga = np.max(c[s:s+sampling_rate])

                    pga_t = self.evtime_ts + timedelta(seconds = t[s+sampling_rate])
                    pga_list.append([pga_t.timestamp(), pga])

                # p or s flag 1->p, 0->s
                p_flag = self.get_p_or_s_label(dist, self.evdp, shift_time, v_p, v_s, p_or_s_percentage)
                p_flag = 1

                # earthquake trigger
                earthquake_trig = 1
                    
                trig_list.append([stla, stlo, dist, trig_time, shift_time, pga_list, threshold, p_flag, earthquake_trig])
                
                if plot:
                    plt.figure(figsize=(12, 8))
                    plt.plot(t, c, label = 'Envelope')
                    a = np.array(pga_list)
                    plt.plot(a[:, 0]-self.evtime_ts.timestamp(), a[:, 1], label='PA')
                    plt.vlines(t[trig_poz],0,max(c),'r',label='Trigger')
                    plt.xlabel('Time since EQ (sec)')
                    plt.ylabel('Amplitude Envelope $(m/s^2)$')
                    plt.title(f'M{mag} event, depth {evdp} km\n'
                               f'Phone at {dist:.2f} km, '
                               f'and threshold is {threshold:.5f} '
                               f'$m/s^2$')
                    plt.legend()
                    plt.show()
        
        trig_list.sort(key=lambda x: x[-2])

        df_trig = pd.DataFrame(trig_list,
            columns = ['latitude', 'longitude', 'dist_km', 
                       'datetime','tt_rel','pga', 
                       'phone_threshold', 'p_label', 
                       'quake_label'])
        df_trig = df_trig.set_index('datetime')
        df_trig['latitude'] = df_trig['latitude'].astype(np.float32)
        df_trig['longitude'] = df_trig['longitude'].astype(np.float32)
        df_trig['dist_km'] = df_trig['dist_km'].astype(np.float32)
        df_trig['tt_rel'] = df_trig['tt_rel'].astype(np.float32)
        df_trig['phone_threshold'] = df_trig['phone_threshold'].astype(np.float32)
        df_trig['p_label'] = df_trig['p_label'].astype(np.int8)
        df_trig['quake_label'] = df_trig['quake_label'].astype(np.int8)
        if self.myshake_network_latency is not None:
            # add a new filed that take into consideration of the trigger time with network latency
            df_trig = df_trig.assign(
            tt_rel_with_latency=df_trig.tt_rel + np.random.choice(
                self.myshake_network_latency, size=len(df_trig)))

        if len(df_trig) >= 1:
            df_trig.index = df_trig.index.tz_localize('UTC')

        meta_data = {'phone_thresholds_g':phone_thresholds, 
                     'v_p':v_p, 'v_s':v_s, 
                     'discount_factor':discount_factor,
                     'steady_percentage':steady_percentage,
                     'evtime':self.evtime, 'evla':self.evla, 
                     'evtime_ts':self.evtime_ts, 'evlo':self.evlo, 
                     'evdp':self.evdp, 'mag':self.mag, 
                     'eventId':self.evid}

        return phones_steady, meta_data, df_trig

if __name__ == '__main__':
    event_info = {}
    event_info['latitude'] = 36
    event_info['longitude'] = -121
    event_info['depth'] = 10 
    event_info['mag'] = 8.0
    event_info['id'] = '001'
    event_info['time'] = '2012-09-29T23:11:12.323Z'
    steady_ratio = pickle.load(open('./data/steady_phone_ratio_hourly.pkl', 'rb'), encoding='latin1')
    phones = np.array([[36.4, -122], [36.39, -121.5], [36.42, -122], [36.3, -121.5]])
    trig_gen = trigger_generator(event_info, steady_ratio, phones, trig_rate=0.000001)
    
    config = {}
    config['variance_threshold'] = 0.01
    config['v_p'] = 6.1
    config['v_s'] = 3.55
    config['phase_check_accuracy'] = 1
    config['discount_factor'] = 0.8
    config['p_time_sigma'] = 0.5
    config['s_time_sigma'] = 0.5
    config['p_or_s_percentage'] = 0.7
    config['amplification_factor'] = 2.0
    config['time_after_decay'] = 30
    config['sampling_rate'] = 25
    config['phone_threshold_g_mean'] = 0.01
    config['phone_threshold_g_std'] = 0.01
    config['phone_threshold_g_bottom'] = 0.005
    
    phones_steady, meta_data, df_trig = trig_gen.generate_triggers_envelope(\
                                        config, timezone = 'UTC', \
                                        random_trigger = False, \
                                        time_win = 24)
    print(df_trig)
    

