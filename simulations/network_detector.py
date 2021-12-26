from mpl_toolkits.basemap import Basemap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import math
import time
from datetime import datetime, timedelta
import itertools
from scipy import spatial
from scipy.optimize import minimize
import pytz
from util import movie_functions
import matplotlib.animation as manimation
import mgrs
from collections import Counter, OrderedDict
from sklearn.cluster import DBSCAN, Birch
from scipy.stats import zscore
from geo_utils import calculate_dist


from sklearn import metrics
import logging
import pickle
import sys

#logging.basicConfig(filename='/Users/qingkaikong/Dropbox/Research/2018_Research/NZ_EEW_simulation/logs/network_detector.log',level=logging.DEBUG)

DEBUG = False

class network_detector(object):

    def __init__(self, evid, event, rf_estimator_P, rf_estimator_S, trigger_uncert=True):
        self.event = event
        self.evid = evid
        self.evla = self.event['latitude']
        self.evlo = self.event['longitude']
        self.evdp = self.event['depth']
        self.mag = self.event['mag']
        self.evtime = self.event['time']
        self.evtime_ts = pd.Timestamp(self.evtime, tz = "UTC")
        self.weights = None
        self.rf_P = rf_estimator_P
        self.rf_S = rf_estimator_S
        self.trigger_uncert = trigger_uncert

    def _generate_times(self, t0, t1, win_lenth = 20., win_step = 1000):
        '''
        Funciton to generate start and end times for the sliding window

        Parameters
        ----------
        t0 - string or datetime-like
             start time
        t1 - string or datetime-like
             end time
        win_len - int or float
             window length
        win_step - int or float
             window step in milliseconds

        Return
        ------
        start_times
        end_times
        '''

        if isinstance(t0, str):
            t0 = pd.Timestamp(t0)
            t1 = pd.Timestamp(t1)

        # frequency is milliseconds
        freq = str(int(win_step)) + 'L'
        start_times = pd.date_range(t0, t1 - pd.Timedelta(seconds = win_lenth), freq = freq)
        end_times = pd.date_range(t0 + pd.Timedelta(seconds = win_lenth), t1, freq = freq)
        return start_times, end_times

    def _estimate_magnitude(self, arr, eq_lat, eq_lon,weighting=0.1):

        '''
        function to estimate magnitude from the PGA values sent by the phones.
        '''

        random_forest = True

        pga = list(arr[:, 4])
        trig_lat = arr[:, 2]
        trig_lon = arr[:, 3]

        #Get flags for P or S triggers. We'll want to use these here!
        trig_ps = arr[:,-1].astype(int)

        #print('PGA vals: %s' %pga)
        #print('LATS: %s' %trig_lat)
        #print('LONS: %s' %trig_lon)
        #print('PS_flags: %s' %trig_ps)

        n = len(pga)
        #print('Length of input array = %i' %n)

        dist = list(map(calculate_dist, trig_lat, trig_lon, [eq_lat]*n, [eq_lon]*n))
        #print('DISTANCE vals: %s' %dist)

        #Make separate distance and pga vectors for the P and S regressors

        dist_p = [dist[i] for i in range(len(trig_ps)) if trig_ps[i] == 1]
        pga_p = [pga[i] for i in range(len(trig_ps)) if trig_ps[i] == 1]

        dist_s = [dist[i] for i in range(len(trig_ps)) if trig_ps[i] == 0]
        pga_s = [pga[i] for i in range(len(trig_ps)) if trig_ps[i] == 0]

        if random_forest == True:

            #if self.trigger_uncert == True:
                #X = np.c_[np.log10(np.array(pga)), np.log10(dist), trig_ps]

                Xp = np.c_[np.log10(np.array(pga_p)), np.log10(dist_p)]
                Xs = np.c_[np.log10(np.array(pga_s)), np.log10(dist_s)]

                try:
                    #rms changed to only using S model. When this is the case
                    #ensure that weighting = 1
                    magp = self.rf_P.predict(Xp)
                except:
                    magp = np.empty(0)

                try:
                    mags = self.rf_S.predict(Xs)
                except:
                    mags = np.empty(0)

                #determine components of weighted mean magnitude
                mag = sum(np.array(list(weighting*magp)+list(mags)))
                nmag = weighting*len(magp) + len(mags)

                #For large events do use the P wave data
                #if (mag/nmag) > 5.5:
                    #use the P mag too if available
                #    try:
                #        magp = self.rf_P.predict(Xp)
                #    except:
                #        magp = np.empty(0)
                #    try:
                #        mags = self.rf_S.predict(Xs)
                #    except:
                #        mags = np.empty(0)
                #    mag = sum(np.array(list(0.1*magp)+list(mags)))
                #    nmag = 0.1*len(magp) + len(mags)


                #print('Len Mag: %i' %len(mag))
                #print('Len dist: %i' %len(dist))
            #else:
            #    X = np.c_[np.log10(np.array(pga)), np.log10(dist)]
            #    mag = self.rf.predict(X)
        else:
            # original magnitude estimation
            mag = np.mean(1.3524*np.log10(np.array(pga)) + 1.6581*np.log10(dist) + 4.8575)


            #mag = np.log10(np.array(pga) * 9.8) + 1.6581*np.log10(dist) + 4.8575

        return mag/nmag

    def _calculate_time_residual(self, lat, lon, t, evdp = 9.0):

        '''This assumes that we're triggering on the S wave'''

        vs = 3.55
        n = len(trig_lat)

        dist = np.array(list(map(calculate_dist, trig_lat, trig_lon, [lat]*n, [lon]*n)))

        dist_hypo = np.sqrt(dist**2 + evdp**2)

        t_s = dist_hypo / vs

        t_obs = (trig_t - t) / 1000.

        #weight the time error by the cell weights. This penalizes cells that may have
        #had lots of false triggers

        if self.weights is None:
            err = np.abs(t_s - t_obs)
        else:
            err = np.abs(t_s - t_obs) * (self.weights)**2


        return [sum(err), lat, lon, t]

    def _grid_search_blind_test(self,dist_time_arr, eq_lat_0, eq_lon_0, evdp = 15 ):
        trig_0 = dist_time_arr[0]
        t0 = float(trig_0[1])

        v_s = 3.55
        v_p = 6.10

        # forming grid
        lat_grid = np.arange(eq_lat_0 - 1, eq_lat_0 + 1, 0.1)
        lon_grid = np.arange(eq_lon_0 - 1, eq_lon_0 + 1, 0.1)
        t_grid = np.arange(t0 - 15000, t0, 1000)
        global trig_lat, trig_lon, trig_t
        trig_lat = dist_time_arr[:, 2].astype(float)
        trig_lon = dist_time_arr[:, 3].astype(float)
        trig_t = dist_time_arr[:, 1].astype(float)
        error_mat = []

        # simplify the grid search
        lats, lons, times = np.meshgrid(lat_grid, lon_grid, t_grid)
        p_s_dict = {}
        count = 0
        for lat, lon, t in zip(lats.ravel(), lons.ravel(), times.ravel()):
            e = 0
            p_s_list = []
            for stla, stlo, st_time in zip(trig_lat, trig_lon, trig_t):
                dist = calculate_dist(lat, lon, stla, stlo)
                hypo = np.sqrt(dist**2+evdp**2)
                estimated_p = hypo / v_p
                estimated_s = hypo / v_s

                t_diff = (st_time - t0) / 1000.
                t_delta_p = np.abs(t_diff - estimated_p)
                t_delta_s = np.abs(t_diff - estimated_s)

                if t_delta_p < t_delta_s:
                    # 0 is p and 1 is s
                    p_or_s = 0
                    e += t_delta_p**2
                else:
                    p_or_s = 1
                    e += t_delta_s**2
                p_s_list.append(p_or_s)


            error_mat.append([lat, lon, t, count, e])
            p_s_dict[count] =  p_s_list
            count+=1
        error_mat = np.array(error_mat)
        ix = np.argmin(error_mat[:, -1])
        eq_lat, eq_lon, eq_t, num, e = error_mat[ix]

        return eq_lat, eq_lon, eq_t, np.array(p_s_dict[num]), e

    def _optimize_event_location(self,dist_time_arr,eq_lat_0,eq_lon_0,eq_t_0,vs=3.55,vp=6.10,evdp=15,real_evtime=None):

        #All trigger times
        trig_t = dist_time_arr[:, 1].astype(float)
        #trigger lats, lons
        trig_lat = dist_time_arr[:, 2].astype(float)
        trig_lon = dist_time_arr[:, 3].astype(float)
        trig_ps = dist_time_arr[:,-1].astype(int)

        #eq_lat_0 = trig_lat[0]
        #eq_lon_0 = trig_lon[0]

        if real_evtime: #In this case, we know the time and just want the location (test)
            t0 = real_evtime
            T = trig_t- t0*1000 #residuals, in ms
            X0 = np.array([eq_lat_0,eq_lon_0])
        else: #In this case, we want the time and the location
            #first guess vector to optimize
            X0 = np.array([eq_lat_0,eq_lon_0,eq_t_0])

        #print((trig_t - np.median(trig_t))/1000)

        ###############################################################################
        #RMS test - tried constrained optimization but didn't work well
        constrained = False
        #for constrained optimization
        #bnds = ((eq_lat_0-0.5,eq_lat_0+0.5),(eq_lon_0-0.5,eq_lon_0+0.5),(None,None))
        #bnds = ((None,None),(None,None),(t0-5000,t0+10))
        ###############################################################################


        if dist_time_arr[:, -2][0] is not None:
            self.weights = dist_time_arr[:, -2].astype(float)

        def __residual_function_t_known(X,T,lats,lons,vp,vs,dep):

            '''
            Simplified residual function where we know the origin time and just want to find the
            epicenter
            '''

            n = len(lats)
            #travel distance
            dist = np.array(list(map(calculate_dist, trig_lat, trig_lon, [X[0]]*n, [X[1]]*n)))
            dist_hypo = np.sqrt(dist**2 + dep**2)
            T = T/1000.0 #convert to seconds

            #squared travel time residuals for both p and s. If the S residual is smaller, we claim to have
            #triggered on S. If the P residual is smaller, we claim to have triggered on P
            residuals_s = (T - (dist_hypo)/vs)**2
            residuals_p = (T - (dist_hypo)/vp)**2

            #Do we want to report if there has been a trigger on P or S?
            residuals = np.array([residuals_p[i] if residuals_p[i] < residuals_s[i] else residuals_s[i] for i in range(len(residuals_p))])
            #residuals = residuals_s

            #We probably want to minimize the L2 norm of the residuals, but if we have weights we can add this term too
            if self.weights is None:
                error = np.sum(residuals)
            else:
                error = np.sum(residuals*(self.weights)**2)

            return error

        def __residual_function(X,lats,lons,times,trig_ps,vp,vs,dep):

            '''
            Residual function where we want to search for T, X and Y
            Note that trig_ps is a list of 1 or 0 depending on whether or not the trigger is a p-wave
            '''

            #times is the vector of phone trigger times. X[2] is the event time

            T = (times - X[2])/1000.0  #Vector of travel times, in seconds

            n = len(lats)
            #travel distance
            dist = np.array(list(map(calculate_dist, trig_lat, trig_lon, [X[0]]*n, [X[1]]*n)))
            dist_hypo = np.sqrt(dist**2 + dep**2)

            #squared travel time residuals for both p and s. If the S residual is smaller, we claim to have
            #triggered on S. If the P residual is smaller, we claim to have triggered on P
            residuals_s = (T - (dist_hypo/vs))**2
            residuals_p = (T - (dist_hypo/vp))**2

            #Do we want to report if there has been a trigger on P or S? This is based on the actual known flags for
            #P or S triggers, rather than which residial is smaller
            residuals = np.array([residuals_p[i] if trig_ps[i] == 1 else residuals_s[i] for i in range(len(trig_ps))])
            
            #residuals = residuals_p

            #residuals = residuals_s #used for Borrego springs event - improves the location

            #1 for P, 0 for S
            #p_or_s = np.array([1 if residuals_p[i] < residuals_s[i] else 0 for i in range(len(residuals_p))])

            #We probably want to minimize the L2 norm of the residuals, but if we have weights we can add this term too
            if self.weights is None:
                error = np.sum(residuals**2)
            else:
                error = np.sum(self.weights*(residuals)**2)

            return error

        #Note this is currently unconstrained, but we could possibly make it more accurate by using bounds

        if real_evtime:
            res = minimize(__residual_function_t_known,X0,args=(T,trig_lat,trig_lon,vp,vs,evdp))
        else:
            if constrained == True:
                #SLSQP performs better than COBYLA
                failstatus = 0
                res = minimize(__residual_function,X0,args=(trig_lat,trig_lon,trig_t,trig_ps,vp,vs,evdp),bounds=bnds,tol=0.001)

                return ([res.x[0],res.x[1],res.x[2],failstatus,trig_ps])

            else:

                #Testing methods
                #"Nelder-Mead - pretty good"
                #'Gradient methods don't seem to perform well
                failstatus = 0
                res = minimize(__residual_function,X0,args=(trig_lat,trig_lon,trig_t,trig_ps,vp,vs,evdp),method="Nelder-Mead",options={'maxiter':5000})
                #This might fail or report an unreasonable location. If so, report it and try a grid search instead
                dist_error = calculate_dist(res.x[0], res.x[1], eq_lat_0, eq_lon_0)

                if((res.status > 0) or (dist_error) > 150):
                    print("Unconstrained optimization failed! Status = %s. Error = %s. Doing grid search" %(res.status,dist_error))
                    failstatus = 1
                    #res = minimize(__residual_function,X0,args=(trig_lat,trig_lon,trig_t,vp,vs,evdp),bounds=bnds)
                    _,lat,lon,time = self._grid_search_loc(dist_time_arr,eq_lat_0,eq_lon_0)


                    #Estimate which of the arrivals are P and S
                    n = len(trig_lat)
                    dist = np.array(list(map(calculate_dist, trig_lat, trig_lon, [lat]*n, [lon]*n)))
                    dist_hypo = np.sqrt(dist**2 + evdp**2)
                    T = (trig_t - time)/1000

                    #squared travel time residuals for both p and s. If the S residual is smaller, we claim to have
                    #triggered on S. If the P residual is smaller, we claim to have triggered on P

                    #We can do this, or we can just report the input P or S flags

                    #residuals_s = (T - (dist_hypo/vs))**2
                    #residuals_p = (T - (dist_hypo/vp))**2

                    #1 for P, 0 for S
                    #p_or_s = np.array([1 if residuals_p[i] < residuals_s[i] else 0 for i in range(len(residuals_p))])

                    return [lat,lon,time,failstatus,trig_ps]

                else:


                    #Estimate which of the arrivals are P and S
                    n = len(trig_lat)
                    dist = np.array(list(map(calculate_dist, trig_lat, trig_lon, [res.x[0]]*n, [res.x[1]]*n)))
                    dist_hypo = np.sqrt(dist**2 + evdp**2)
                    T = (trig_t - res.x[2])/1000

                    #squared travel time residuals for both p and s. If the S residual is smaller, we claim to have
                    #triggered on S. If the P residual is smaller, we claim to have triggered on P
                    residuals_s = (T - (dist_hypo/vs))**2
                    residuals_p = (T - (dist_hypo/vp))**2

                    #1 for P, 0 for S
                    #p_or_s = np.array([1 if residuals_p[i] < residuals_s[i] else 0 for i in range(len(residuals_p))])

                    return [res.x[0],res.x[1],res.x[2],failstatus,trig_ps]




    def _grid_search_loc(self, dist_time_arr, eq_lat_0, eq_lon_0, evdp = 15):

        trig_0 = dist_time_arr[0]
        t0 = float(trig_0[1])

        v_s = 3.55
        v_p = 6.10

        # forming grid
        lat_grid = np.arange(eq_lat_0 - 1, eq_lat_0 + 1, 0.1)
        lon_grid = np.arange(eq_lon_0 - 1, eq_lon_0 + 1, 0.1)
        t_grid = np.arange(t0 - 15000, t0, 1000)

        #confusing that this is made global
        global trig_lat, trig_lon, trig_t
        trig_lat = dist_time_arr[:, 2].astype(float)
        trig_lon = dist_time_arr[:, 3].astype(float)
        trig_t = dist_time_arr[:, 1].astype(float)

        if dist_time_arr[:, -2][0] is not None:
            self.weights = dist_time_arr[:, -2].astype(float)
        error_mat = []

        # simplify the grid search
        count = 0
        lats, lons, times = np.meshgrid(lat_grid, lon_grid, t_grid)

        # use oneliner to replace the following lines, this is basically doing grid search
        # for 3 nested loops
        error_mat = list(map(self._calculate_time_residual, lats.ravel(), lons.ravel(), times.ravel()))

        #for lat, lon, t in zip(lats.ravel(), lons.ravel(), times.ravel()):
        #
        #    if count %500 == 0:
        #        print(count)
        #    err = calculate_time_residual(trig_lat, trig_lon, trig_t, lat, lon, t, evdp = 9.0)
        #    error_mat.append(err)
        #    count+=1

        error_mat = np.array(error_mat)
        return error_mat[np.argmin(error_mat[:, 0])]

    def making_detection_movie_refine(self, detected_events_withEvents, df_trig_eq, df_heartbeat_eq, win_step, filename, llat, ulat, llon, \
    ulon,identifier,cluster_triggers=None,cities=None,timezone='UTC',location_info='Some earthquake',plot_detections=True,plot_from_real_data=False):

        rcParams['font.size'] = 30
        rcParams['font.weight'] = 'bold'
        rcParams['font.family'] = "sans-serif"
        rcParams['font.sans-serif'] = "Open Sans"
        plt.style.use("ggplot")

        if plot_from_real_data == True:
            pmarkersize = 7.5
        else:
            pmarkersize = 3.5

        if plot_detections == False:

            cluster_triggers = None

        warning_MMI = 4 #MMI for which to plot shaking radius

        if cities is not None:

            #There should always be three cities

            # City 1
            city1_name = cities['city3'][0]
            lat_c1 = cities['city3'][1]
            lon_c1 = cities['city3'][2]

            # City 2
            city2_name = cities['city2'][0]
            lat_c2 = cities['city2'][1]
            lon_c2 = cities['city2'][2]

            # City 3
            city3_name = cities['city1'][0]
            lat_c3 = cities['city1'][1]
            lon_c3 = cities['city1'][2]

            city_lons = [lon_c1,lon_c2,lon_c3]
            city_lats = [lat_c1,lat_c2,lat_c3]
            city_names = [city1_name,city2_name,city3_name]

        event_count = 0 #currently not used

        #print(len(detected_events_withEvents))

        item = detected_events_withEvents[-1]

        #print(item)

        detected_events = item[0]
        meta_data = item[1]
        events_triggers = item[2]

        #print('-----------------------------------------')
        #print(detected_events,meta_data,events_triggers)

        t0 = self.evtime
        t0_ts = pd.Timestamp(t0).value/ 1e9

        evtime_ts = pd.Timestamp(self.evtime, tz = "UTC")

        t0 = (evtime_ts - timedelta(seconds = 20)).strftime('%Y-%m-%d %H:%M:%S')
        t1 = (evtime_ts + timedelta(seconds = 100)).strftime('%Y-%m-%d %H:%M:%S')

        #2 hours before the current time
        t_2h = (evtime_ts - timedelta(hours = 2)).strftime('%Y-%m-%d %H:%M:%S')

        # note that, the win_step I changed to milliseconds
        # this should be the same as used in the detection function!
        # win step should be 500 if trigger plotting is to be used!
        #win_step = 500
        start_times, end_times = movie_functions.generate_times(t0, t1, win_lenth = 20., win_step = win_step)

        hdf_trig = df_trig_eq.loc[t0:t1]
        hdf_trig_background = df_trig_eq.loc[t_2h:t0]

        df_trig_eq.index >= t0

        hdf_hb_background = df_heartbeat_eq

        try:
            hdf_trig['latitude'] = [x['coordinates'][1] for x in hdf_trig['l']]
            hdf_trig['longitude'] = [x['coordinates'][0] for x in hdf_trig['l']]

            hdf_trig_background['latitude'] = [x['coordinates'][1] for x in hdf_trig_background['l']]
            hdf_trig_background['longitude'] = [x['coordinates'][0] for x in hdf_trig_background['l']]
        except:
            hdf_trig['latitude'] = [x[1] for x in hdf_trig['l']]
            hdf_trig['longitude'] = [x[0] for x in hdf_trig['l']]

            hdf_trig_background['latitude'] = [x[1] for x in hdf_trig_background['l']]
            hdf_trig_background['longitude'] = [x[0] for x in hdf_trig_background['l']]

        # select only triggers with location
        df_trig = hdf_trig[(hdf_trig['latitude'] != 0) & (hdf_trig['longitude'] != 0)]
        df_trig_bg = hdf_trig_background[(hdf_trig_background['latitude'] != 0) & (hdf_trig_background['longitude'] != 0)]
        df_hb_background = hdf_hb_background[(hdf_hb_background['latitude'] != 0) & (hdf_hb_background['longitude'] != 0)]

        df_trig_reg = df_trig[(df_trig['latitude'] > llat) & (df_trig['latitude'] < ulat) & \
                         (df_trig['longitude'] > llon) & (df_trig['longitude'] < ulon) ]

        df_trig_reg_bg = df_trig_bg[(df_trig_bg['latitude'] > llat) & (df_trig_bg['latitude'] < ulat) & \
                         (df_trig_bg['longitude'] > llon) & (df_trig_bg['longitude'] < ulon) ]

        df_hb_reg_bg = df_hb_background[(df_hb_background['latitude'] > llat) & (df_hb_background['latitude'] < ulat) & \
                         (df_hb_background['longitude'] > llon) & (df_hb_background['longitude'] < ulon) ]

        df_hb_reg_bg = df_hb_reg_bg.drop_duplicates('deviceId')

        plot_background_rate = True
        service = 'World_Shaded_Relief'
        service = 'World_Imagery'
        service = None
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=filename, artist='Matplotlib',
                comment='Movie support!')
        writer = FFMpegWriter(fps=10, metadata=metadata)

        fig = plt.figure(figsize = (17, 10))
        ax = fig.add_subplot(1,1,1)
        #ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.tight_layout()

        #Need to add a plot that is always the same size!

        if service is not None:
            m = Basemap(projection='merc', lon_0=-121.36929, lat_0=37.3215,
                llcrnrlon=llon,llcrnrlat=llat- 0.01,urcrnrlon=ulon,urcrnrlat=ulat + 0.01,resolution='l',
                    epsg = 4269,ax=ax)
            m.arcgisimage(service=service, xpixels = 3000, verbose= True)
        else:
            m = Basemap(projection='merc', lon_0=-121.36929, lat_0=37.3215,
                llcrnrlon=llon,llcrnrlat=llat- 0.01,urcrnrlon=ulon,urcrnrlat=ulat + 0.01,resolution='l',ax=ax,anchor='W')
            dist_threshold = 100


            #Generate basemap
            m.drawcoastlines(linewidth=0.5)
            m.drawcountries(linewidth=0.5)
            m.drawmapboundary(fill_color='#D5DAE3')
            m.fillcontinents(color='#ECF0F6',lake_color='#D5DAE3', alpha = 1)
            m.drawstates()

            #m.drawlsmask(land_color='#ECF0F6',ocean_color='#D5DAE3',lakes=True,resolution='l',grid=1.25)
            lon_range = ulon - llon
            lat_range = ulat - llat

            #Generate distance scale

            #For Borrego
            
            if "Berkeley" in location_info:
                m.drawmapscale(llon + 0.24 * lon_range, llat + 0.07 * lat_range, self.evlo, self.evla, dist_threshold, barstyle='fancy',fontsize=10, zorder = 10)
            else:
                m.drawmapscale(llon + 0.23 * lon_range, llat + 0.07 * lat_range, self.evlo, self.evla, dist_threshold, barstyle='fancy',fontsize=10, zorder = 10)
            #For Haiti
            #m.drawmapscale(llon + 0.25 * lon_range, llat + 0.07 * lat_range, self.evlo, self.evla, dist_threshold, barstyle='fancy',fontsize=10, zorder = 10)

        if plot_background_rate:
            stla_bg = df_hb_reg_bg['latitude'].values
            stlo_bg = df_hb_reg_bg['longitude'].values
            lons = stlo_bg
            lats = stla_bg
            x, y = m(lons,lats)
            # #9ca1a8
            m.plot(x,y,1,marker='o',markersize=pmarkersize,color='#7BCDC9', alpha = 0.5, lw = 0, zorder = 8, label ='MyShake phone')

        #Create placeholder for the timing information
        if service is None:
            #For non NZ region
            tt = plt.text(0.94, 0.92, 'test', verticalalignment='bottom', horizontalalignment='right',\
            transform=ax.transAxes, color='k', fontsize=25,zorder=10)
            #For NZ region
            #tt = plt.text(0.35, 0.92, 'test', verticalalignment='bottom', horizontalalignment='right',\
            #transform=ax.transAxes, color='k', fontsize=25,zorder=10)
        else:
            tt = plt.text(0.96, 0.92, 'test', verticalalignment='bottom', horizontalalignment='right',\
            transform=ax.transAxes, color='r', fontsize=25,zorder=10)

        # plot empty triggers
        lons = []
        lats = []
        x, y = m(lons,lats)
        sc_trig, _ = m.plot(x,y,1,marker='o',markersize=pmarkersize,color='#FD6E34', lw = 0, zorder = 9, alpha = 1, label ='Triggered phone')

        # plot phones associated with a cluster
        lons = []
        lats = []
        x, y = m(lons,lats)
        cluster_trig, _ = m.plot(x,y,1,marker='o',markersize=pmarkersize,color='#FF9367', lw = 0, zorder = 9, alpha = 1, label="Used for location")

        # plot the event red star
        lons = []
        lats = []
        x, y = m(lons,lats)
        event_star = m.plot(x,y,'*', markersize= 20 , color='#4632FF', lw = 0, zorder = 10, label ='True earthquake location')

        # plot estimated event epicenter
        lons = []
        lats = []
        x, y = m(lons,lats)
        est_event = m.plot(x,y,'*', markersize= 20, lw = 0, color='#8577FF', zorder = 10, label ='Estimated earthquake location')

        # plot P and S
        lons = []
        lats = []
        x, y = m(lons,lats)
        circ_p = plt.plot(x,y, color = '#F3FF7B', alpha = 1, zorder = 9, label ='P-wave front')

        lons = []
        lats = []
        x, y = m(lons,lats)
        circ_s = plt.plot(x,y, color = '#D0DD45', alpha = 1, zorder = 9, label ='S-wave front')

        #plot warning radius
        lons = []
        lats = []
        x, y = m(lons,lats)
        circ_warning = plt.plot(x,y, color = '#FF0000', alpha = 1, zorder = 9, label ='Limit of shaking intensity %s' %warning_MMI)


        event_information_text = plt.text(1.03, 0.45, '', transform=ax.transAxes, fontsize = 18)

        event_simulation_text = plt.text(1.03, 0.3, '', transform=ax.transAxes, fontsize = 22, fontweight='bold')

        # somehow I have repeat legend, therefore, get rid of the duplicate ones
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))

        #Title of the simulation
        title = plt.text(0.05,1.05,"MyShake rerun",transform=ax.transAxes,fontsize=30,fontweight='bold')

        #For Borrego animation
        if 'Borrego' in location_info:
            plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.25, 0.15), fontsize = 12, numpoints = 1)
            
        elif 'Northridge' in location_info:
            plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.25, 0.15), fontsize = 12, numpoints = 1)

        #For Berkeley animation
        elif 'Berkeley' in location_info:
            plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.3, 0.15), fontsize = 12, numpoints = 1)

        #For Palu animation
        elif 'Palu' in location_info:
            plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.25, 0.15), fontsize = 12, numpoints = 1)

        #For NZ animation
        #plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.35, 0.15), fontsize = 12, numpoints = 1)

        #For Nepal/Haiti animation
        elif 'Haiti' in location_info:
            plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.25, 0.15), fontsize = 12, numpoints = 1)

        #For New Zealand
        #plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.32, 0.15), fontsize = 12, numpoints = 1)

        #For Peru
        #plt.legend(by_label.values(), by_label.keys(), loc = 'center', bbox_to_anchor=(1.25, 0.15), fontsize = 12, numpoints = 1)

        if cities is not None:

            for element in zip(city_names,city_lons,city_lats):
                
                if element[0] == 'San Francisco': #check for Richard videos, put SF name on the other side
                    x,y = m(element[1],element[2])
                    xt,yt = m(element[1]-0.42,element[2])

                elif element[0].strip() == 'Lalitpur': #check for Richard video
                    x,y = m(element[1],element[2])
                    xt,yt = m(element[1]-0.58,element[2])

                elif element[0].strip() == 'Leogane':
                    x,y = m(element[1],element[2])
                    xt,yt = m(element[1]-0.28,element[2])

                elif element[0].strip() == 'Callao':
                    x,y = m(element[1],element[2])
                    xt,yt = m(element[1]-0.15,element[2])

                else:
                    x,y = m(element[1],element[2])
                    xt,yt = m(element[1]+0.02,element[2])

                #plot the location of the cities as a black dot
                plt.plot(x,y,'ko',markersize='5',zorder=15)
                plt.annotate(element[0],xy=(x,y),xytext=(xt,yt),textcoords='data',fontsize=14,zorder=15)



        count_ = 0
        count = 0

        time_index = df_trig_reg.index

        with writer.saving(fig, filename, 250):

            #Loop over times frames for movie. This is useful when we want to plot the evolution of the epicenter location
            #and cluster of triggers associated with the even
            event_stepper = 0
            
            #trigger file identifier
            #tfid = 0

            for t_start, t_end in zip(start_times, end_times):


                local_t = t_end.replace(tzinfo=pytz.timezone(timezone))
                local_t = local_t - pd.Timedelta(seconds=8*3600)

                #print(t_start,t_end,local_t)
                #print(evtime_ts)

                count += 1
                if count > 10000:
                    break

                #print(event_stepper)

                # if plot background triggers (1 hour before the start time)
                if plot_background_rate:
                    lons = stlo_bg
                    lats = stla_bg
                    x, y = m(lons,lats)
                    # #9ca1a8
                    m.scatter(x,y,15,marker='o',color='#7BCDC9', alpha = 0.3, lw = 0, zorder = 9)

                # plot the phones
                # This is not working with milliseconds
                #df_select = df_trig_CA.loc[t_start.strftime('%Y%m%d %H:%M:%S.3%f'):t_end.strftime('%Y%m%d %H:%M:%S.3%f')]

                t_start_str = t_start.strftime('%Y%m%d %H:%M:%S.3%f')
                t_end_str = t_end.strftime('%Y%m%d %H:%M:%S.3%f')

                # this is finally let me to slice the milliseconds
                ix_slice = (time_index >= t_start_str) & (time_index <= t_end_str)
                df_select = df_trig_reg[ix_slice]
                df_select.index = df_select.index.tz_localize(pytz.UTC)

                #print(df_select.index)
                #print(t_start)

                #Remove tz information
                #df_select.index = df_select.index.tz_localize(None)

                stla = df_select['latitude'].values
                stlo = df_select['longitude'].values

                alpha = movie_functions.make_color(t_start, df_select.index)

                lons = stlo
                lats = stla
                x, y = m(lons,lats)

                rgba_colors = np.zeros((len(stlo),4))

                # get the tf
                tf = df_select['tf']

                # set the phones triggered by EEW blue (if you change the dots rgba_colors[np.where(tf != 2),0], it will be blue)
                rgba_colors[np.where(tf != 3),0] = 1.0
                # set the phones triggered by ANN green
                rgba_colors[np.where(tf == 3),0] = 1.0
                # the fourth column needs to be your alphas
                rgba_colors[:, 3] = alpha
                #m.scatter(x,y,40,marker='o',color=rgba_colors, lw = 0, zorder = 10)
                sc_trig.set_data(x,y)
                sc_trig.set_color = rgba_colors

                events_color = movie_functions.make_color_events(t_start - 20, evtime_ts)
                lons = self.evlo
                lats = self.evla
                x, y = m(lons,lats)

                rgba_colors = np.zeros((1,4))
                rgba_colors[:,0] = 1.0
                # the fourth column needs to be your alphas
                rgba_colors[:, 3] = events_color

                # add text at the bottom
                t_org = evtime_ts
                if t_end >= evtime_ts:

                    lats = self.evla
                    lons = self.evlo
                    x, y = m(lons,lats)
                    event_star[0].set_data(x,y)

                    vp = 6.10
                    vs = 3.55
                    t_from_origin = (t_end - t_org).seconds + (t_end - t_org).microseconds/1e6

                    #hypocenter distances
                    dist_p = t_from_origin * vp
                    dist_s = t_from_origin * vs

                    #Time to reach surface
                    t_p_surface = self.evdp / vp
                    t_s_surface = self.evdp / vs

                    #There was a + here -- should be -?

                    #d is the epicenter distance
                    d = np.sqrt(dist_p**2 - self.evdp**2)
                    if t_from_origin >= t_p_surface:
                        movie_functions.equi_update(circ_p[0], m, self.evlo, self.evla, d)

                    d = np.sqrt(dist_s**2 - self.evdp**2)
                    if t_from_origin >= t_s_surface:
                        movie_functions.equi_update(circ_s[0], m, self.evlo, self.evla, d)
                else:
                    t_from_origin = -((t_org - t_end).seconds + (t_org - t_end).microseconds/1e6)
                    #print t_from_origin, t_end, t_org

                #Initial alert time
                try:
                    alertTime = detected_events[0][1]
                except:
                    alertTime = t_start
                try:
                    alertTime = alertTime.tz_localize(pytz.utc)
                except:
                    pass
                if t_end >= alertTime:

                    try:
                        ev_tend = detected_events[event_stepper][1]
                        if ev_tend is None:
                            ev_tend = t_start
                    except:
                        ev_tend = t_start
                    #Case where time has advanced enough for us to update the location of the event in the movie
                    if t_end > ev_tend:
                        event_stepper += 1
                    #Case where we've reached the end of the list of detections
                    if event_stepper == len(detected_events):
                        event_stepper = len(detected_events)-1


                    #If cluster triggers are provided, we are going to plot the evolution of th clusters assigned to that particular event
                    #over time

                    if cluster_triggers is not None:

                        #trigger lats, lons
                        trig_lats = cluster_triggers[event_stepper][:, 2].astype(float)
                        trig_lons = cluster_triggers[event_stepper][:, 3].astype(float)
                        
                        #if tfid < 20:
                        #    triggers_df = pd.DataFrame({'lon':trig_lons,'lat':trig_lats})
                        #    triggers_df_name = 'active_triggers_%03i.csv' %tfid
                        #    triggers_df.to_csv(triggers_df_name)
                        #tfid += 1

                        trig_x, trig_y = m(trig_lons,trig_lats)
                        #print(trig_lons,trig_lats)
                        cluster_trig.set_data(trig_x,trig_y)

                    if plot_detections == True:

                        evla_e = detected_events[event_stepper][2]
                        evlo_e = detected_events[event_stepper][3]
                        evdp_e = detected_events[event_stepper][-3]

                        x, y = m(evlo_e,evla_e)
                        est_event[0].set_data(x,y)

                        if cities is not None:

                           dist_c1 = np.sqrt(calculate_dist(self.evla, self.evlo, lat_c1, lon_c1)**2+self.evdp**2)
                           dist_c2 = np.sqrt(calculate_dist(self.evla, self.evlo, lat_c2, lon_c2)**2+self.evdp**2)
                           dist_c3 = np.sqrt(calculate_dist(self.evla, self.evlo, lat_c3, lon_c3)**2+self.evdp**2)

                        #Change the boundary color to red during warning

                        for axis in ['top','bottom','left','right']:
                           ax.spines[axis].set_linewidth(4)
                           ax.spines[axis].set_color('r')

                        mag_e = detected_events[event_stepper][7]

                        #Find the radius of warning corresponding to MMI = 4
                        MMI2_dist = DistanceMMI(10,mag_e,warning_MMI)
                        movie_functions.equi_update(circ_warning[0], m, evlo_e, evla_e, MMI2_dist)

                        #Only plot the cities alert times at the initial detection

                        if count_ < 10000:


                           if cities is not None:

                               if count_ < 1:

                                   dist_c1_e = np.sqrt(calculate_dist(evla_e, evlo_e, lat_c1, lon_c1)**2+self.evdp**2)
                                   dist_c2_e = np.sqrt(calculate_dist(evla_e, evlo_e, lat_c2, lon_c2)**2+self.evdp**2)
                                   dist_c3_e = np.sqrt(calculate_dist(evla_e, evlo_e, lat_c3, lon_c3)**2+self.evdp**2)

                                   mmi_e_c1 = MMIDistance(evdp_e, mag_e, dist_c1_e)
                                   mmi_c1 = MMIDistance(self.evdp, self.mag, dist_c1)

                                   mmi_e_c2 = MMIDistance(evdp_e, mag_e, dist_c2_e)
                                   mmi_c2 = MMIDistance(self.evdp, self.mag, dist_c2)

                                   mmi_e_c3 = MMIDistance(evdp_e, mag_e, dist_c3_e)
                                   mmi_c3 = MMIDistance(self.evdp, self.mag, dist_c3)

                                   if mmi_e_c1 < 1:
                                       mmi_e_c1 = 1

                                   if mmi_e_c2 < 1:
                                       mmi_e_c2 = 1

                                   if mmi_e_c3 < 1:
                                       mmi_e_c3 = 1

                                  #calculate the warning time to the three cities

                                   t_p_c1 = dist_c1 / vp - t_from_origin
                                   t_s_c1 = dist_c1 / vs - t_from_origin

                                   t_p_c2 = dist_c2 / vp - t_from_origin
                                   t_s_c2 = dist_c2 / vs - t_from_origin

                                   t_p_c3 = dist_c3 / vp - t_from_origin
                                   t_s_c3 = dist_c3 / vs - t_from_origin

                                   ts_list = [t_s_c1,t_s_c2,t_s_c3]
                                   mmi_e_list = [mmi_e_c1,mmi_e_c2,mmi_e_c3]
                                   mmi_list = [mmi_c1,mmi_c2,mmi_c3]

                                   sorted_t_indices = np.argsort([t_s_c1,t_s_c2,t_s_c3])
                                   i1 = sorted_t_indices[0]
                                   i2 = sorted_t_indices[1]
                                   i3 = sorted_t_indices[2]

                                   #This text goes in the table
                                   tc1 = '%.1f sec' %ts_list[i1]
                                   tc2 = '%.1f sec' %ts_list[i2]
                                   tc3 = '%.1f sec' %ts_list[i3]

                               #Uncomment whats below for different cities

                               plt.text(1.03, 0.66, 'WARNING M%.1f       \n\n\n' %(mag_e), verticalalignment='bottom', horizontalalignment='left',\
                               transform=ax.transAxes, color='#FF0000', fontsize=40, weight='bold',backgroundcolor='#FFC2C2')

                               plt.text(1.03, 0.83, '%-17s%-10s%-13s%-12s' %('City','Warning','Estimated  ',' Observed '), verticalalignment='bottom', horizontalalignment='left',\
                               transform=ax.transAxes, color='black', fontsize=16, fontstyle='italic', weight='bold',backgroundcolor='#FFC2C2')
                               plt.text(1.03, 0.79, '                      Time       Intensity       Intensity ', verticalalignment='bottom', horizontalalignment='left',\
                               transform=ax.transAxes, color='black', fontsize=16, fontstyle='italic', weight='bold',backgroundcolor='#FFC2C2')


                               ######################################################################
                               #Below are parameter sets that work for various movies that we've made
                               ######################################################################

                               #These parameters work for Indonesia region (Palu event)
                               
                               if 'Palu' in location_info:
                                   text_ = '%-11s%-5.1fsec    %-7i            %-15i\n%-15s%-5.1fsec    %-7i            %-15i\n%-15s%-5.1fsec    %-7i            %-15i'\
                                   %(city_names[i1], ts_list[i1], mmi_e_list[i1], mmi_list[i1], city_names[i2], ts_list[i2], mmi_e_list[i2], mmi_list[i2], city_names[i3], ts_list[i3], mmi_e_list[i3], mmi_list[i3])

                               #These parameters work for Nepal region
                               elif 'Nepal' in location_info:
                                   text_ = '%-18s%-12s%-16i%-13i\n%-16s%-12s%-16i%-13i\n%-13s%-12s%-16i%-13i'\
                                   %(city_names[i1].strip(), tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2].strip(), tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3].strip(), tc3, mmi_e_list[i3], mmi_list[i3])

                               #These parameters work for Haiti region
                               elif 'Haiti' in location_info:
                                   text_ = '%-19s%-12s%-16i%-13i\n%-21s%-12s%-16i%-13i\n%-17s%-12s%-16i%-13i'\
                                   %(city_names[i1].strip(), tc1, round(mmi_e_list[i1]), round(mmi_list[i1]), city_names[i2].strip(), tc2, round(mmi_e_list[i2]), round(mmi_list[i2]), city_names[i3].strip(), tc3, round(mmi_e_list[i3]), round(mmi_list[i3]))

                               #These parameters work for NZ region
                               #text_ = '%-17s%-10s%-20i%-15i\n%-16s%-10s%-20i%-15i\n%-19s%-10s%-20i%-15i'\
                               #%(city_names[i1].strip(), tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2].strip(), tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3].strip(), tc3, mmi_e_list[i3], mmi_list[i3])

                               #These parameters work for California region (Loma Prieta and Southern Hayward)
                               #text_ = '%-19s%-5.1fsec    %-7i          %-15i\n%-15s%-5.1fsec   %-7i          %-15i\n%-16s%-5.1fsec    %-7i          %-15i'\
                               #%(city_names[i1], ts_list[i1], mmi_e_list[i1], mmi_list[i1], city_names[i2], ts_list[i2], mmi_e_list[i2], mmi_list[i2], city_names[i3], ts_list[i3], mmi_e_list[i3], mmi_list[i3])

                               elif ('Borrego' in location_info):
                                   text_ = '%-15s%-11s%-18i%-15i\n%-17s%-10s%-18i%-15i\n%-16s%-10s%-18i%-15i'\
                                   %(city_names[i1], tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2], tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3], tc3, mmi_e_list[i3], mmi_list[i3])
                               
                               elif 'Northridge' in location_info:
                                   text_ = '%-15s%-11s%-18i%-15i\n%-14s%-11s%-18i%-15i\n%-16s%-10s%-18i%-15i'\
                                   %(city_names[i1], tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2], tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3], tc3, mmi_e_list[i3], mmi_list[i3])

                               elif 'Berkeley' in location_info:
                                   text_ = '%-15s%-10s%-20i%-15i\n%-18s%-10s%-20i%-15i\n%-15s%-10s%-20i%-15i'\
                                   %(city_names[i1], tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2], tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3], tc3, mmi_e_list[i3], mmi_list[i3])
                               
                               elif 'New_Zealand' in location_info:
                                   text_ = '%-15s%-10s%-20i%-15i\n%-14s%-10s%-20i%-15i\n%-17s%-10s%-20i%-15i'\
                                   %(city_names[i1], tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2], tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3], tc3, mmi_e_list[i3], mmi_list[i3])
                               
                               elif 'Korea' in location_info:
                                   text_ = '%-16s%-10s%-20i%-15i\n%-14s%-10s%-20i%-15i\n%-17s%-10s%-20i%-15i'\
                                   %(city_names[i1], tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2], tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3], tc3, mmi_e_list[i3], mmi_list[i3])    
                               
                               elif 'Portugal' in location_info:
                                   text_ = '%-16s%-10s%-20i%-15i\n%-14s%-10s%-20i%-15i\n%-17s%-10s%-20i%-15i'\
                                   %(city_names[i1], tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2], tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3], tc3, mmi_e_list[i3], mmi_list[i3])   
                               
                                   

                               #These parameters are good for Kashmir
                               #text_ = '%-18s%-10s%-18.1f%-15.1f\n%-18s%-10s%-18.1f%-15.1f\n%-18s%-10s%-18.1f%-15.1f'\
                               #%(city_names[i1], tc1, mmi_e_list[i1], mmi_list[i1], city_names[i2], tc2, mmi_e_list[i2], mmi_list[i2], city_names[i3], tc3, mmi_e_list[i3], mmi_list[i3])

                               plt.text(1.03, 0.66, text_, verticalalignment='bottom', horizontalalignment='left',
                               transform=ax.transAxes, color='black', fontsize=16, backgroundcolor='#FFC2C2')


                        count_ += 1

                #counting from origin time
                text_ = '%.1f sec'%(t_from_origin)
                tt.set_text(text_)

                location_text = "M%.1f %s \nDepth %.1f km" %(self.mag,location_info,self.evdp)
                #location_text = "M%.1f Palu, Indonesia\nDepth %.1f km" %(self.mag,self.evdp)

                event_simulation_text.set_text(location_text)

                #Update the event information (and time)
                event_information_text.set_text('Time(UTC)  %s\nTime(Local) %s\nTriggers %d'%(t_end.strftime('%Y-%m-%d %H:%M:%S'),local_t.strftime('%Y-%m-%d %H:%M:%S'),len(df_select)))
                plt.savefig('./data/figures/' + str(count) + '_' + identifier + '.png', dpi = 100)
                writer.grab_frame()


    def map_triggers_region(self,cluster_triggers,eq_lat,eq_lon,true_eq_lat,true_eq_lon,eq_lon_0,eq_lat_0,ID,plotting_time_offset):

        '''
        Produce a map of the triggers at some time. For debugging use
        '''

        print("Inside mapping function")

        rcParams['font.size'] = 30
        rcParams['font.weight'] = 'bold'
        rcParams['font.family'] = "sans-serif"
        plt.style.use("ggplot")

        minlat = true_eq_lat - 2
        maxlat = true_eq_lat + 2
        minlon = true_eq_lon - 2
        maxlon = true_eq_lon + 2

        #trigger lats, lons
        trig_lat = cluster_triggers[:, 2].astype(float)
        trig_lon = cluster_triggers[:, 3].astype(float)

        plt.figure(figsize = (10,10))

        m = Basemap(projection='merc', lon_0=true_eq_lon, lat_0=true_eq_lat,
            llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,urcrnrlat=maxlat,resolution='i')

        dist_threshold = 50
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.fillcontinents(color='#cc9966',lake_color='#99ffff', alpha = 0.3)
        m.drawstates()

        x, y = m(trig_lon,trig_lat)
        m.scatter(x,y,4.0,marker='o',color='#FD6E34', alpha = 1, lw = 0, zorder = 10)

        eqx,eqy = m(eq_lon,eq_lat)
        m.plot(eqx,eqy,'*', markersize= 15 , lw = 0, zorder = 10,color='r',label='True epicenter')
        teqx,teqy = m(true_eq_lon,true_eq_lat)
        m.plot(teqx,teqy,'*', markersize= 15 , lw = 0, zorder = 10,color='g',label='Predicted epicenter')
        t0eqx,t0eqy = m(eq_lon_0,eq_lat_0)
        m.plot(t0eqx,t0eqy,'*', markersize= 5 , lw = 0, zorder = 10,color='y',label='starting epicenter')

        # plot the location of the P wave at this time
        lons = []
        lats = []
        x, y = m(lons,lats)
        circ_p = plt.plot(x,y, color = 'g', alpha = 0.5, label = 'P-wave front')

        evdp = 10
        vp = 6.55
        dist_p = vp*plotting_time_offset
        d = np.sqrt(dist_p**2 + self.evdp**2)

        movie_functions.equi_update(circ_p[0], m, true_eq_lon, true_eq_lat, d)

        plt.legend()
        plt.title("MyShake simulation")
        print("Plotting map")
        plt.savefig('event_triggers_'+str(ID)+'.png', bbox_inches = 'tight',pad_inches = 0, dpi = 300)

class mgrs_dbscan_detector(network_detector):

    def __convert_to_mgrs(self, lats, lons, MGRSPrecision = 1):
        '''
         MGRSPrecision = 1, resolution is 10 km
         MGRSPrecision = 2, resolution is 1 km
         MGRSPrecision = 3, resolution is 100 m
         MGRSPrecision = 4, resolution is 10 m
         MGRSPrecision = 5, resolution is 1 m
        '''
        # Note, if you use MGRSPrecision = 1, it returns the center of the cell. I tested using the map visualization: https://mappingsupport.com/p/coordinates-mgrs-google-maps.html
        m = mgrs.MGRS()
        inDegrees = True
        MGRSPrecision=MGRSPrecision
        n = len(lats)
        c = list(map(m.toMGRS, lats, lons, [True]*n, [MGRSPrecision]*n))
        
        return c

    def __get_all_cells_above_threshold(self, ratio_mat_raw, n_steady_phone = 6, ratio_threshold = 0.5):

        #Note rest of nsteadphone to 6

        s_keys = None
        # if there are more than 2 cells light up

        # we here filter out the cells that less than X number of steady phones
        ratio_mat = []
        for i in range(len(ratio_mat_raw)):
            ratio, c_steady_phone, key = ratio_mat_raw[i]

            if float(c_steady_phone) >= n_steady_phone:
                ratio_mat.append([ratio, c_steady_phone, key])

        ratio_mat = np.array(ratio_mat)
        if len(ratio_mat) >= 2:
            ratios = ratio_mat[:,0].astype(float)

            ix = ratios >= ratio_threshold
            keys = ratio_mat[:,-1]

            s_keys = keys[ix]
            n_sum = len(s_keys)
        #elif len(ratio_mat) == 1:
        #    ratios = float(ratio_mat[0][0])
            # if one cell have more than 10 phones and more than 80% of the phones triggered
        #    if ratios > 0.80:
        #        s_keys = ratio_mat[0][1]

        return s_keys


    def _associate_additional_trig(self, trig_buf, active_events, t_start, t_end):
        '''
        This function is to associate triggers in the trigger buffer with the event we
        created.

        Input:
        trig_buf - a buffer contains triggers
        active_events - a list contains the information of the events we created. schema is:
        [eq_t, t_end, eq_lat, eq_lon, evid, _, eq_dep, eq_mag, _]

        Return:
        updated trig_buf
        '''
        for key in active_events.keys():
            # Note, you could choose whether you want to associate with the inital estimation (0) or the latest estimation (-1)
            detected_event = active_events[key][-1]
            # get the event information
            event_lat = detected_event[2]
            event_lon = detected_event[3]
            event_time = detected_event[0]
            event_id = detected_event[4]

            # loop through the triggers in the buffer and associate with the event

            #Remove TZ information from index
            #print(t_start)
            #print(t_end)
            #print(trig_buf.head())
            #print(trig_buf.index.dtype)

            ts = t_start.strftime('%Y-%m-%d %H:%M:%S')
            te = t_end.strftime('%Y-%m-%d %H:%M:%S')

            #print(trig_buf[ts:te])
            triggers_group = trig_buf.loc[ts:te]
            triggers_group['zscore_lat'] = zscore(triggers_group['latitude'])
            triggers_group['zscore_lon'] = zscore(triggers_group['longitude'])


            for key, trig in triggers_group.iterrows():
                if trig['association'] != 0:
                    continue
                else:

                    trig_time = trig['tt']
                    trig_lat = trig['latitude']
                    trig_lon = trig['longitude']
                    trig_zscore = abs((trig['zscore_lat']+trig['zscore_lon'])/2)
                    #print(trig_zscore)

                    # this needs convert to seconds
                    travel_time = (trig_time - event_time)/1000.

                    trigger_distance = calculate_dist(trig_lat, trig_lon, event_lat, event_lon)

                    # do not associate a trigger if it is farther that 500km from the source location
                    # must also be within 2 standard deviations of the mean lon/lat
                    if (travel_time < 500. / 1.5 and trigger_distance < 500 and trig_zscore < 2):

                        # check for a near source trigger within 20km of the event
                        if (trigger_distance < 20 and travel_time >= -2.0 \
                            and travel_time < 20.0):
                            trig_buf.loc[key, 'association'] = event_id

                        else:
                            # tt_min, tt_max got from the association boundary of our historical data
                            #associate all triggers that are within this boundary

                            tt_min = 0.3 * trigger_distance - 18.
                            tt_max = 0.31449 * trigger_distance + 18

                            if (travel_time >= tt_min and travel_time <= tt_max):
                                trig_buf.loc[key, 'association'] = event_id




        return trig_buf

    def __estimate_location_and_magnitude(self, eq_lat_0, eq_lon_0, df_select, trigger_weights, c_ratio_dict, MGRSPrecision, optimize_location, \
    ratio_in_cell_for_quality_trig = 0.5,n_phones_in_cell_threshold=2,from_real_data=False):
        
        #n phones in cell threshold was 2, 1 for MGRS = 2  
        
        dist_time_arr = []
        #number of times we have updated the earthquake location
        number_of_fits = 0

        #number of times the optimization fails during earthquake location
        total_opt_fails = 0

        #eq_lon = None
        #eq_lat = None
        latest_opt_fail = None
        
        #print(df_select)


        for ix, row in df_select.iterrows():
            #Calculate distance from each triggered phone to the event initial location
            #Note that the initial location may be wrong however!
            dist = calculate_dist(eq_lat_0, eq_lon_0, row.latitude, row.longitude)
            #print(dist)

            #This is important - we need to try to reduce the impact of false/random triggers
            #where the density of phones is high

            if dist <= 150: # We don't want to use the further triggers

                #convert phone position to MRGS cell location
                key = self.__convert_to_mgrs([row.latitude], [row.longitude], MGRSPrecision = MGRSPrecision)[0]
                #get trigger weight associated with that cell
                #weight = trig_weights[key]
                weight = c_ratio_dict[key][0]
                n_steady_phone_in_cell = c_ratio_dict[key][1]
                
                #print(key,weight)
                #print(ratio_in_cell_for_quality_trig,n_steady_phone_in_cell,n_phones_in_cell_threshold)

                if trigger_weights:

                    if (weight >= ratio_in_cell_for_quality_trig) & (n_steady_phone_in_cell > n_phones_in_cell_threshold):
                        dist_time_arr.append([ix, row['tt'], row.latitude, row.longitude, row.pga, weight, row['deviceid'],row['p_label']])
                else:
                    #RMS change from 0.3 to 0.8
                    if weight >= ratio_in_cell_for_quality_trig:
                        dist_time_arr.append([ix, row['tt'], row.latitude, row.longitude, row.pga, None, row['deviceid'],row['p_label']])


        #This is the array that contains the triggers that are part of a particular cluster
        dist_time_arr = np.array(dist_time_arr)

        # sort the triggers by trigger time
        #print(dist_time_arr)
        if len(dist_time_arr) > 0:
          dist_time_arr = dist_time_arr[dist_time_arr[:,1].argsort()]
        else:
          return None

        #RMS test - just select the reasonable (low) PGA values from real data
        #This is not ideal because we need the magnitude threshold to scale with the size of the event

        if from_real_data == True:
            dist_time_arr_mag = dist_time_arr[dist_time_arr[:,4]<0.1]
        else:
            dist_time_arr_mag = dist_time_arr


        #Setting first estimate of event time to the first trigger time
        eq_t_0 = dist_time_arr[:, 1].astype(float)[0]

        #This need to be outside the function as a user parameter. We could also optimize for it
        eq_dep = 10

        print("Number of triggers: %i" %len(dist_time_arr))

        #print(dist_time_arr_mag)

        #If the number of triggers is really low, just use the first one to estimate the eq?

        #if len(dist_time_arr) < 10:

        #    eq_mag_0 = np.mean(self._estimate_magnitude(dist_time_arr_mag, self.evla, self.evlo))

        #    return dist_time_arr, eq_lat_0, eq_lon_0, eq_t_0, eq_mag_0, eq_dep, total_opt_fails, np.nan, np.nan


        #optimize_location = True
        #regular_gridsearch = True

        if optimize_location == True:
            results = self._optimize_event_location(dist_time_arr,eq_lat_0,eq_lon_0,eq_t_0,evdp=eq_dep,real_evtime=None)
            eq_lat = results[0]
            eq_lon = results[1]
            eq_t = results[2]
            latest_opt_fail = results[3]

            #This comes from the dist_time_arr info
            ps_trigs = results[4]

            #if the optimization fails, report it
            total_opt_fails += latest_opt_fail

            #Here we need to use the fact that we know if it was a P or S triggs

            #RMS test to understand spread in magntude estimations
            true_location_test = False

            if true_location_test == True:
                eq_mag = self._estimate_magnitude(dist_time_arr_mag, self.evla, self.evlo)
            else:
                eq_mag = self._estimate_magnitude(dist_time_arr_mag, eq_lat, eq_lon)

        elif regular_gridsearch == True:
            ## regular grid search
            results = self._grid_search_loc(dist_time_arr, eq_lat_0, eq_lon_0, evdp = eq_dep)
            _, eq_lat, eq_lon, eq_t = results
            true_location = False
            if true_location:
                eq_mag = self._estimate_magnitude(dist_time_arr_mag, self.evla, self.evlo)
            else:
                eq_mag = self._estimate_magnitude(dist_time_arr_mag, eq_lat, eq_lon)


        #In this case, we attempt to see if the trigger has been on a P or S wave
        else:
            eq_lat, eq_lon, eq_t, p_s_list, e = self._grid_search_blind_test(dist_time_arr, eq_lat_0, eq_lon_0, evdp = eq_dep)

            # get the S wave
            s_trigs = dist_time_arr[p_s_list == 1]

            if len(s_trigs) == 0:
                eq_mag_p = self._estimate_magnitude(dist_time_arr_mag[p_s_list == 0], eq_lat, eq_lon)
                eq_mag = eq_mag_p
            else:
                eq_mag_p = self._estimate_magnitude(dist_time_arr_mag[p_s_list == 0], eq_lat, eq_lon)
                eq_mag_s = self._estimate_magnitude(dist_time_arr_mag[p_s_list == 1], eq_lat, eq_lon)

                if eq_mag_s > eq_mag_p:
                    eq_mag = eq_mag_s
                else:
                    eq_mag = eq_mag_p
        return dist_time_arr, eq_lat, eq_lon, eq_t, eq_mag, eq_dep, total_opt_fails, ps_trigs

    def __get_the_trig_weights_and_c_ratio(self, df_select, counter_steady_phone, MGRSPrecision):

        #if there are no triggers, move on to the next timestep
        stla = df_select['latitude'].values
        stlo = df_select['longitude'].values

        c_ratio_mat = []
        trig_weights = {}
        c_ratio_dict = {}

        #Convert to mgrs grid
        mgrs_trig = Counter(self.__convert_to_mgrs(stla, stlo, MGRSPrecision = MGRSPrecision))

        #Generate trigger weights
        for key in mgrs_trig.keys():

            #number of triggers in this cell
            c_trig = float(mgrs_trig[key])
            #number of steady phones in this cell
            c_steady_phone = float(counter_steady_phone[key])
            if c_steady_phone==0:
                r = 0
            else:
                r = c_trig/c_steady_phone
            trig_weights[key] = r

            c_ratio_mat.append([r, c_steady_phone, key])
            c_ratio_dict[key] = [r, c_steady_phone]
        return trig_weights, c_ratio_mat, c_ratio_dict


    def detector_with_updates(self, phones_steady, df_trig, true_location = False, trigger_weights=True, optimize_location=True, mapping_ID_number=0, \
    config = None, from_real_data=False, nupdate='default', include_shakealert_trig=False, df_shakealert_trig=None):

        '''
        Detector for DBSCAN+associate
        '''


        kms_per_radian = 6371.0088
        if config is None:
            # so our distance threshold is 200 km, and we need at least two cells light up to declare an earthquake
            epsilon = 200 / kms_per_radian
            dbscan_min_sample = 2

            if nupdate == 'default':
                num_of_updates = 20
            else:
                num_of_updates = nupdate

            #number of times that we update the eq detection
            n_cells_to_lightup = 2
            n_steady_phone_in_a_cell = 5 #was 5, 1 for MGRS = 2 For New Zealand we dropped it to 3
            ratio_of_trig_in_a_cell = 0.3 #was 0.5
            ratio_in_cell_for_quality_trig = 0.3 #was 0.5
            n_phones_in_cell_for_quality_trig = 2 #was 2
            sliding_win_length_sec = 20
            sliding_win_step_msec = 500
            sliding_win_start_before_origin_time = 20
            sliding_win_end_after_origin_time = 60
        else:
            epsilon_dist = config['epsilon_dist']
            epsilon = epsilon_dist / kms_per_radian
            dbscan_min_sample = config['dbscan_min_sample']
            num_of_updates = config['num_of_updates']
            n_cells_to_lightup = config['n_cells_to_lightup']
            n_steady_phone_in_a_cell = config['n_steady_phone_in_a_cell']
            ratio_of_trig_in_a_cell = config['ratio_of_trig_in_a_cell']
            ratio_in_cell_for_quality_trig = config['ratio_in_cell_for_quality_trig']
            n_phones_in_cell_for_quality_trig = config['n_phones_in_cell_for_quality_trig']
            sliding_win_length_sec = config['sliding_win_length_sec']
            sliding_win_step_msec = config['sliding_win_step_msec']
            sliding_win_start_before_origin_time = config['sliding_win_start_before_origin_time']
            sliding_win_end_after_origin_time = config['sliding_win_end_after_origin_time']

        dbscan = DBSCAN(eps=epsilon, min_samples=dbscan_min_sample, algorithm='ball_tree', \
                    metric='haversine')



        
        #changed from MGRSPrecision = 1
        MGRSPrecision = 1

        steady_phones_mgrs = self.__convert_to_mgrs(phones_steady[:, 0], phones_steady[:, 1], MGRSPrecision = MGRSPrecision)
        counter_steady_phone = Counter(steady_phones_mgrs)

        counter_trig = Counter(self.__convert_to_mgrs(df_trig.latitude, df_trig.longitude, MGRSPrecision = MGRSPrecision))

        m = mgrs.MGRS()

        evtime = self.evtime_ts
        evla = self.evla
        evlo = self.evlo
        evdp = self.evdp
        mag = self.mag

        evtime_ts = pd.Timestamp(self.evtime, tz = "UTC")

        t0 = (evtime_ts - timedelta(seconds = sliding_win_start_before_origin_time)).strftime('%Y%m%d %H:%M:%S.3%f')
        t1 = (evtime_ts + timedelta(seconds = sliding_win_end_after_origin_time)).strftime('%Y%m%d %H:%M:%S.3%f')
        t_2h = (evtime_ts - timedelta(hours = 2)).strftime('%Y%m%d %H:%M:%S.3%f')

        # now the window becomes 0.1 s apart to improve the detection time

        #Note that there may be an interesting tradeoff between location accuracy and warning time
        #if we wait longer before issuing a warning, we should have a more accurate location
        start_times, end_times = movie_functions.generate_times(t0, t1, win_lenth = sliding_win_length_sec, win_step = sliding_win_step_msec)


        df_heartbeat_eq = pd.DataFrame(phones_steady, columns=['latitude', 'longitude'])


        df_heartbeat_eq['deviceId'] = range(len(df_heartbeat_eq))
        df_heartbeat_eq['datetime'] = evtime
        df_heartbeat_eq = df_heartbeat_eq.set_index('datetime')
        df_heartbeat_eq['ts'] = df_heartbeat_eq.index.astype(np.int64)/1000000
        df_heartbeat_eq['hbSource'] = 1

        #Remove the time zone information from the index (we know its UTC)
        #df_heartbeat_eq.index = df_heartbeat_eq.index.tz_localize(None)
        #df_trig.index = df_trig.index.tz_localize(None)

        try:
            df_trig['l']
            df_trig_eq = df_trig
        except:
            df_trig_eq = df_trig
            df_trig_eq['deviceid'] = [str(x) for x in range(len(df_trig_eq))]
            df_trig_eq['tt'] = df_trig_eq.index.astype(np.int64)/1000000

            df_trig_eq['l'] = df_trig_eq[['longitude', 'latitude']].values.tolist()
            df_trig_eq = df_trig_eq.drop('latitude', 1)
            df_trig_eq = df_trig_eq.drop('longitude', 1)
            df_trig_eq['tf'] = 3

        # Note, in real implementation, this should be at the top level to have the column indicate whether
        # the trigger got associated with a cluster

        df_trig_eq['association'] = 0

        try:
            hdf_trig = df_trig_eq.loc[t0:t1]
            hdf_trig_background = df_trig_eq.loc[t_2h:t0]
        except Exception as e:
            #print('Something might be wrong with time slicing')
            t0 = (evtime - timedelta(seconds = sliding_win_start_before_origin_time)).strftime('%Y%m%d %H:%M:%S')
            t1 = (evtime + timedelta(seconds = sliding_win_end_after_origin_time)).strftime('%Y%m%d %H:%M:%S')
            t_2h = (evtime - timedelta(hours = 2)).strftime('%Y%m%d %H:%M:%S')
            hdf_trig = df_trig_eq.loc[t0:t1]
            hdf_trig_background = df_trig_eq.loc[t_2h:t0]

        hdf_hb_background = df_heartbeat_eq

        try:
            hdf_trig['latitude'] = [x['coordinates'][1] for x in hdf_trig['l']]
            hdf_trig['longitude'] = [x['coordinates'][0] for x in hdf_trig['l']]

            hdf_trig_background['latitude'] = [x['coordinates'][1] for x in hdf_trig_background['l']]
            hdf_trig_background['longitude'] = [x['coordinates'][0] for x in hdf_trig_background['l']]
        except:
            hdf_trig['latitude'] = [x[1] for x in hdf_trig['l']]
            hdf_trig['longitude'] = [x[0] for x in hdf_trig['l']]

            hdf_trig_background['latitude'] = [x[1] for x in hdf_trig_background['l']]
            hdf_trig_background['longitude'] = [x[0] for x in hdf_trig_background['l']]

        # select only triggers with given location
        df_trig = hdf_trig[(hdf_trig['latitude'] != 0) & (hdf_trig['longitude'] != 0)]
        #heartbeat database (triggers)
        df_trig_bg = hdf_trig_background[(hdf_trig_background['latitude'] != 0) & (hdf_trig_background['longitude'] != 0)]
        #heartbeat database (background triggers)
        df_hb_background = hdf_hb_background[(hdf_hb_background['latitude'] != 0) & (hdf_hb_background['longitude'] != 0)]

        df_hb_bg = df_hb_background.drop_duplicates('deviceId')


        t_0 = time.time()
        detected_events = []
        detected_events_time = []
        #list containing the detected events and their characteristics
        detected_events_withEvents = []

        #### DO SOMETHING HERE TO REMOVE THE DETECTED EVENTS OLDER THAN 60 SEC, MEANS WE DON'T ASSOCIATE WITH EVENTS OLDER THAN 60 SEC
        #list the contains the triggers used to locate the event at each timestep of the simulation
        triggers_used_for_location = []
        errors = []

        evid = 0
        cid = 0

        detected_events_with_update = {}
        error_tracking = []
        plot_ID = 0
        first_centroid_error = None

        mapping_information = open('mapping_info_%s.csv' %mapping_ID_number,'w')
        mapping_information.write("Elat,Elon,Tlat,Tlon,Slat,Slon,IDno,t_from_origin\n")
        
        mgrs_cells = open('mgrs_all_cells.csv','w')


        for t_start, t_end in zip(start_times, end_times):

            ###-------------------------------------------------------------------------------
            ### Case where we already have one or more events detected and we want to update
            ###-------------------------------------------------------------------------------

            # if there are detected events in the last X seconds (need implement), then we will associate the new triggers
            if len(detected_events_with_update) > 0:

                df_trig = self._associate_additional_trig(df_trig, detected_events_with_update, t_start, t_end)
                # get the triggers updates the magnitude and location
                df_trig_associated = df_trig[df_trig['association'] != 0]

                #print("Number of detected events: %g" %len(detected_events))

                for detected_event in detected_events:

                    event_lat = detected_event[2]
                    event_lon = detected_event[3]
                    event_time = detected_event[0]
                    event_id = detected_event[4]

                    # let's just do num_of_updates updates for now, but could be changed
                    n_updates = len(detected_events_with_update[event_id])
                    if n_updates > num_of_updates :
                        continue

                    # Note there are two options for the initial locations for the search of the location, using the very first
                    # median location, or updated one, here I only use the very first median location
                    #eq_lat_0, eq_lon_0 = detected_event[5]

                    #Or we can update the next initial location with the old version
                    #eq_lat_0 = event_lat
                    #eq_lon_0 = event_lon

                    # get the associated triggers
                    df_select = df_trig[df_trig['association'] == event_id]

                    #calculate mean of trigger locations
                    #print(df_select.head())
                    eq_lat_0 = np.mean(df_select['latitude'])
                    eq_lon_0 = np.mean(df_select['longitude'])

                    trig_weights, c_ratio_mat, c_ratio_dict = self.__get_the_trig_weights_and_c_ratio(df_select, counter_steady_phone, MGRSPrecision)
                    tmp = self.__estimate_location_and_magnitude(eq_lat_0, eq_lon_0, df_select, trigger_weights, c_ratio_dict, MGRSPrecision, \
                    optimize_location, ratio_in_cell_for_quality_trig,from_real_data=from_real_data)
                    if tmp is not None:
                        dist_time_arr, eq_lat, eq_lon, eq_t, eq_mag, eq_dep, total_opt_fails, ps_trigs = tmp
                    else:
                        continue
                    print('##################%d updated location and magnitude at %.1f sec after the first alert######################'%(n_updates, t_end.timestamp() - detected_event[1].timestamp()))
                    detected_events_with_update[event_id].append([eq_t, t_end, eq_lat, eq_lon, evid, [eq_lat_0, eq_lon_0], eq_dep, eq_mag, _])

                    #Reinsert new locations ready for next update
                    detected_event[2] = eq_lat
                    detected_event[3] = eq_lon

                    #number of p and s triggers (estimated)
                    npwave = np.sum(ps_trigs)
                    nswave = len(ps_trigs)-npwave

                    triggers_used_for_location.append(dist_time_arr)

                    # origin time difference
                    origin_time_error = eq_t/ 1e3 - self.evtime_ts.timestamp()

                    alertTime_from_origin = (t_end.timestamp() - self.evtime_ts.timestamp())
                    dist_error = calculate_dist(eq_lat, eq_lon, self.evla, self.evlo)
                    mag_error = eq_mag - self.mag
                    errors = [mag_error, dist_error, origin_time_error, alertTime_from_origin,total_opt_fails,npwave,nswave,first_centroid_error]
                    print('M%.1f earthquake, estimated M%.1f, alert sent out %.1fs after origin of the Eq.\nMag. difference is %.1f, origin time difference is %.1fs (Est. - Ori.), with %.1fkm error in location'%(self.mag, eq_mag, alertTime_from_origin, mag_error, origin_time_error, dist_error))
                    error_tracking.append([self.mag,eq_mag,origin_time_error,alertTime_from_origin,dist_error,n_updates])
                    print("Writing data for update ID %s ...." %plot_ID)
                    #self.map_triggers_region(df_select,eq_lat,eq_lon,self.evla,self.evlon,eq_lat_0,eq_lon_0,ID,timeoffset)


                    ###
                    ### Added for debugging - save the array containing trigger information to a file
                    dist_time_arr_file = 'dist_time_arr_%03i_%s.npy' %(plot_ID,mapping_ID_number)
                    np.save(dist_time_arr_file,dist_time_arr)
                    mapping_information.write('%g,%g,%g,%g,%g,%g,%i,%g\n' %(eq_lat,eq_lon,self.evla,self.evlo,eq_lat_0,eq_lon_0,plot_ID,alertTime_from_origin))
                    #self.map_triggers_region(dist_time_arr,eq_lat,eq_lon,self.evla,self.evlo,eq_lon_0,eq_lat_0,plot_ID,alertTime_from_origin)
                    plot_ID+=1


            ###-------------------------------------------------------------------------------
            ### Case for the first update
            ###-------------------------------------------------------------------------------
            ### NOTE: The else statement here means that we are restricting ourselves to locating
            ### just one earthquake per simulation


            else:

                # only get the triggers not associated with events for new detections
                df_trig_no_association = df_trig[df_trig['association'] == 0]

                #This looks there was an issue with the time format
                try:
                    df_select = df_trig_no_association.loc[t_start.strftime('%Y-%m-%d %H:%M:%S.3%f'):t_end.strftime('%Y-%m-%d %H:%M:%S.3%f')]
                except:
                    df_select = df_trig_no_association.loc[t_start.strftime('%Y-%m-%d %H:%M:%S'):t_end.strftime('%Y-%m-%d %H:%M:%S')]

                if len(df_select) == 0:
                    continue
                
                #tfilename = 'active_triggers_%03i.csv' %cid
                #df_select.to_csv(tfilename)
                cid += 1

                trig_weights, c_ratio_mat, c_ratio_dict = self.__get_the_trig_weights_and_c_ratio(df_select, counter_steady_phone, MGRSPrecision)

                c_ratio_mat = np.array(c_ratio_mat)

                if DEBUG:
                    logging.debug(c_ratio_mat)
                    print(c_ratio_mat)

                #Get only cells containing more than 5 steady phones where half of them triggered
                
                #These are the MGRS grid cells that contain triggers
                
                cells = []
                for i in range(len(c_ratio_mat)):
                    ratio, c_steady_phone, key = c_ratio_mat[i]
                    cells.append(key)
                    
                    mgrs_cells.write('%s ' %key.decode())
                    
                #print(cells)
                mgrs_cells.write("\n")

                s_keys = self.__get_all_cells_above_threshold(c_ratio_mat, n_steady_phone = n_steady_phone_in_a_cell, ratio_threshold = ratio_of_trig_in_a_cell)
                
                #print(n_cells_to_lightup)

                if s_keys is None:
                    mgrs_cells.write('None\n')
                    continue
                elif len(s_keys) < n_cells_to_lightup: #need at least 2 cells activated to proceed
                    #print("No cells above threshold!")
                    mgrs_cells.write('None\n')
                    continue
                
                    #print(SW.encode().toLatLon(),SE.encode().toLatLon(),NE.encode().toLatLon(),NW.encode().toLatLon())
                for cell in s_keys:
                    mgrs_cells.write('%s ' %cell.decode())
                mgrs_cells.write('\n')
                    
                #sys.exit(1)
                coords = np.array(list(map(m.toLatLon, s_keys)))

                #This is the coordinate array that we're fitting with dbscan
                #print(coords)

                # Run the DBSCAN from sklearn
                dbscan.fit(np.radians(coords))

                cluster_labels_dbscan = dbscan.labels_

                cluster_labels = dbscan.labels_
                n_clusters = len(set(cluster_labels))

                if (n_clusters == 1) & (cluster_labels[0] == -1):
                    continue

                # get the cluster
                # cluster_labels = -1 means outliers
                clusters = \
                    pd.Series([coords[cluster_labels == n] for n in range(-1, n_clusters)])

                #If we have clusters, then we must have detected one or more events.
                #We loop though the clusters and determine the event magnitude and location

                if n_clusters > 0:
                    for i in range(n_clusters):
                        cluster_keys = s_keys[cluster_labels == i]
                        
                        for key in cluster_keys:
                            mgrs_cells.write('%s ' %key.decode())
                        mgrs_cells.write('\n-----------------------------\n')
                        
                        eq_cluster = clusters[i+1]

                        coords = np.array(list(map(m.toLatLon, cluster_keys)))

                        logging.debug('Coords:\n##############################')
                        logging.debug(coords)

                        #Starting location of the earthquake search is the average of the cluster coords

                        #Choose the median to reduce the impact of outliers
                        # the initial mean location of the triggers as the initial estimation
                        #try:

                        #Determine median location
                        eq_lat_0 = np.median(coords[:,0])
                        eq_lon_0 = np.median(coords[:,1])
                        
                        #Determine the distance from the first cluster centroid to the real event. For plotting purposes
                        if plot_ID == 0:
                            first_centroid_lat = eq_lat_0
                            first_centroid_lon = eq_lon_0
                            first_centroid_error = calculate_dist(eq_lat_0, eq_lon_0, evla, evlo)
                            
                        #Find location of initial trigger
                        #eq_lat_0 = coords[0,0]
                        #eq_lon_0 = coords[0,1]


                        #This is where the new event location gets calculated
                        
                        dist_time_arr, eq_lat, eq_lon, eq_t, eq_mag, eq_dep, total_opt_fails, ps_trigs = self.__estimate_location_and_magnitude(eq_lat_0, eq_lon_0, df_select, trigger_weights, c_ratio_dict, MGRSPrecision, \
                        optimize_location, ratio_in_cell_for_quality_trig,from_real_data=from_real_data)
                        #print(dist_time_arr[:, 1]/1000. - t_end.timestamp())
                        _ = ''
                        evid += 1

                        phones_to_create_event = dist_time_arr[:, -1]
                        ix_phones_to_create_event = df_trig['deviceid'].isin(phones_to_create_event)
                        # Note here, it seems in Python 3 I have to use this to assign value to the original dataframe
                        # Otherwise, it will assign the number on a copy instead (be careful, this will be a harder bug)
                        df_trig.loc[:, 'association'][ix_phones_to_create_event] = evid

                        # This is for testing purposes
                        #for ixx, roow in df_trig[ix_phones_to_create_event].iterrows():
                        #    print(calculate_dist(roow.latitude, roow.longitude, eq_lat, eq_lon), (roow.tt - eq_t)/1000 )
                        detected_event = [eq_t, t_end, eq_lat, eq_lon, evid, [eq_lat_0, eq_lon_0], eq_dep, eq_mag, _]
                        #This detected events list gets updated with each 'new' event that is found
                        detected_events.append(detected_event)
                        detected_events_with_update[evid] = []
                        # add the original detection
                        detected_events_with_update[evid].append(detected_event)

                        #detected_events_withEvents.append([detected_events, self.event, _])
                        triggers_used_for_location.append(dist_time_arr)

                        # origin time difference
                        origin_time_error = eq_t/ 1e3 - self.evtime_ts.timestamp()

                        #number of p and s triggers (estimated)
                        npwave = np.sum(ps_trigs)
                        nswave = len(ps_trigs)-npwave

                        alertTime_from_origin = (t_end.timestamp() - self.evtime_ts.timestamp())
                        dist_error = calculate_dist(eq_lat, eq_lon, self.evla, self.evlo)
                        mag_error = eq_mag - self.mag
                        errors = [mag_error, dist_error, origin_time_error, alertTime_from_origin,total_opt_fails,npwave,nswave,first_centroid_error]
                        #error_tracking.append([self.mag,eq_mag,origin_time_error,alertTime_from_origin,dist_error])
                        print('M%.1f earthquake, estimated M%.1f, alert sent out %.1fs after origin of the Eq.\nMag. difference is %.1f, origin time difference is %.1fs (Est. - Ori.), with %.1fkm error in location'%(self.mag, eq_mag, alertTime_from_origin, mag_error, origin_time_error, dist_error))
                        #print("Generating triggers map for ID %s ...." %mapping_ID_number)
                        #RMS addition for debugging: Make a map of the region that shows the cluster triggers
                        #self.map_triggers_region(dist_time_arr,eq_lat,eq_lon,self.evla,self.evlo,eq_lon_0,eq_lat_0,mapping_ID_number,alertTime_from_origin
                        #except:
                        #    print("Error in locating event")


        #This list should have just one element: the list of detected events and the real event we wanted to detect
        mapping_information.close()
        mgrs_cells.close()
        detected_events_withEvents.append([detected_events,self.event,''])

        chosen_ev_id = 1
        try:
            detected_events_withEvents.append([detected_events_with_update[chosen_ev_id],self.event,''])
        except:
            detected_events_withEvents.append([[None,None,None,None,None,[None,None],None,None,None],self.event,''])
            

        return detected_events_withEvents, df_trig_eq, df_heartbeat_eq, errors, triggers_used_for_location, error_tracking


class elarms_detector_with_kdtree(network_detector):

    def _to_Cartesian(self, lat, lng):
        '''
        function to convert latitude and longitude to 3D cartesian coordinates
        '''
        R = 6371 # radius of the Earth in kilometers

        x = R * math.cos(lat) * math.cos(lng)
        y = R * math.cos(lat) * math.sin(lng)
        z = R * math.sin(lat)
        return x, y, z

    def _deg2rad(self, degree):
        '''
        function to convert degree to radian
        '''
        rad = degree * 2*np.pi / 360
        return(rad)

    def _rad2deg(self, rad):
        '''
        function to convert radian to degree
        '''
        degree = rad/2/np.pi * 360
        return(degree)

    def _distToKM(self, x):
        '''
        function to convert cartesian distance to real distance in km
        '''
        R = 6371 # earth radius
        gamma = 2*np.arcsin(self._deg2rad(x/(2*R))) # compute the angle of the isosceles triangle
        dist = 2*R*math.sin(gamma/2) # compute the side of the triangle
        return(dist)

    def _kmToDIST(self, x):
        '''
        function to convert real distance in km to cartesian distance
        '''
        R = 6371 # earth radius
        gamma = 2*np.arcsin(x/2./R)

        dist = 2*R*self._rad2deg(math.sin(gamma / 2.))
        return(dist)

    def _update_location(self, trig_list):
        '''
        This function will update the location of the earthquake with newly triggers.

        Input:
        trig_list - list of triggers with location, 3 columns are: time, lat, lon.

        Return:
        event_time - time of the estimated event.
        evla - latitude of the estimated event.
        evlo - longitude of the estimated event.
        '''
        trig_list = np.array(trig_list)

        # assume the first station trigger time is earthquake time for simplicity
        event_time = trig_list[:, 0].min()

        # the centroid of the stations is the location of the earthquake
        evla = trig_list[:, 1].mean()
        evlo = trig_list[:, 2].mean()
        return event_time, evla, evlo

    def _generate_ANN_triggers_pool(self, df):
        '''
        Function to generate the ANN_triggers_pool
        '''

        #Note, the data may not sorted
        df.sort_index(inplace=True)

        # create the trigger pools for testing purposes, this id should be deviceID + timestamp which should be
        # unique for each trigger, note: I use ordered dictionary
        ANN_triggers_pool = OrderedDict()
        for ix, trig in df.iterrows():
            trig_id = trig['deviceid'] + '_' + str(trig['tt'])
            try:
                ANN_triggers_pool[trig_id] = {'lat': trig['l']['coordinates'][1], 'lon': trig['l']['coordinates'][0], 'trig_time': trig['tt'] /1000,
                                             'associate_event': 0 , 'associate_trig': 0, 'trig_list': [],
                                             'visited_for_eventDetection':0, 'visited_for_association': [], 'pga':trig['pga']}
            except:
                ANN_triggers_pool[trig_id] = {'lat': trig['l'][1], 'lon': trig['l'][0], 'trig_time': trig['tt'] /1000,
                                             'associate_event': 0 , 'associate_trig': 0, 'trig_list': [],
                                             'visited_for_eventDetection':0, 'visited_for_association': [], 'pga':trig['pga']}


        return ANN_triggers_pool

    def _estimate_magnitude(self, associated_trig, event_lat, event_lon):
        '''
        function to estimate magnitude from the PGA values sent by the phones.
        '''
        pga = associated_trig['pga']
        dist = calculate_dist(associated_trig['lat'], associated_trig['lon'], event_lat, event_lon)
        # these constants are from the past earthquakes waveforms
        mag = 1.3524*np.log10(pga) + 1.6581*np.log10(dist) + 4.8575
        return mag

    def _detect_event(self, trig_buf, event_id, max_association_tt_time, current_time, tree_steady,tree_trig):
        '''
        This function is the main function to create an earthquake. Currently, I use the
        suggestion that Arno gave me, and turned it into complex n. The main data structure
        I use this time is ordered dictionary, which I think is similar or can be easily changed
        to our jason format that we used on most of the server side.

        trig_buf - ordered dictionary, nested. An example is:
        {'id':{'lat':latitude, 'lon':longitude, 'trig_time':timestamp, 'associate_event': event_id ,
            'associate_trig': int, 'trig_list': [trig_id], 'visited_for_eventDetection': 0/1, 'visited_for_association': [event_id]}}
        'id' - unique id of the trigger, can be deviceID + triggerTime
        'lat' - latitude of the trigger, float
        'lon' - longitude of the trigger, float
        'trig_time' - Should be a unix timestamp that can be converted to human readable time, but here I use the ANN trigger time since origin of the earthquake
        'associate_event' - used to flag whether this trigger associated with an earthquake using the earthquake id: event_id
        'associate_trig' - a counter to show how many following triggers that associated with this trigger, used to confirm if this is an earthquake
        'trig_list' - a list of trigger IDs that associate with this trigger
        'visited_for_eventDetection' - 0 or 1, used in the detection algorithm to flag whether this trigger have already been paired before
        'visited_for_association' - a list of event_ids used in the association function to mark if this trigger have already been checked with this event

        The logic of this function is:
        Whenever there're new triggers come in, check with the previous triggers to see if they can associate together, if yes, update the counter of the previous triggers,
        and then check if they satisfy the requirement to create an earthquake.
        '''

        # get the trig id of triggers that not been visited before
        new_trig_ids = []
        for key, trig in trig_buf.items():
            if trig['visited_for_eventDetection'] == 0:
                new_trig_ids.append(key)

        # since in 1 time step, there maybe more than one new triggers added in
        for ix_new_trig in new_trig_ids:

            new_trig = trig_buf[ix_new_trig]

            event_time = new_trig['trig_time']
            event_lat = float(new_trig['lat'])
            event_lon = float(new_trig['lon'])
            associate_event = new_trig['associate_event']

            if associate_event == 0:
                # this means this trigger is not associate with any event
                visited_trig_ids = []

                for key, trig in trig_buf.items():
                    if trig['visited_for_eventDetection'] == 1:
                        visited_trig_ids.append(key)
                #print(len(visited_trig_ids))
                # this is for the first trigger that are no visited triggers before
                if len(visited_trig_ids) == 0:
                    trig_buf[ix_new_trig]['visited_for_eventDetection'] = 1
                    return event_id, None, trig_buf

                # using the new triggers to compare with the old triggers that already been visited
                for ix_old_trig in visited_trig_ids:
                    old_trig = trig_buf[ix_old_trig]

                    trig_time = old_trig['trig_time']
                    trig_lat = float(old_trig['lat'])
                    trig_lon = float(old_trig['lon'])
                    trig_associate_event = old_trig['associate_event']
                    trig_list = old_trig['trig_list']

                    add_this_trigger = False
                    if trig_associate_event == 0:

                        if np.abs(trig_time - event_time) <= max_association_tt_time and calculate_dist(event_lat, event_lon, trig_lat, trig_lon) <= 10:
                            add_this_trigger = True

                        # check if the trigger time difference between trigger_stations is greater than the
                        # travel time of surface wave between stations + 2.0secs
                        # loop through the triggers all associate with an event

                        for ix in trig_list:
                            associated_triggers = trig_buf[ix]
                            event_trig_lat = associated_triggers['lat']
                            event_trig_lon = associated_triggers['lon']

                            tt = calculate_dist(event_trig_lat, event_trig_lon, event_lat, event_lon) / 2.0
                            t_check = tt + 2.
                            if (ix_old_trig == '8_1479081781591') and (ix_new_trig == '38_1479081785202'):
                                print('#########', calculate_dist(event_lat, event_lon, trig_lat, trig_lon), np.abs(trig_time - event_time), np.abs(associated_triggers['trig_time'] - event_time) > t_check)

                            if np.abs(associated_triggers['trig_time'] - event_time) > t_check:
                                add_this_trigger = False
                    else:
                        return event_id, None, trig_buf

                    trig_buf[ix_new_trig]['visited_for_eventDetection'] = 1
                    if add_this_trigger:
                        trig_buf[ix_old_trig]['associate_trig'] += 1
                        trig_buf[ix_old_trig]['trig_list'] += [ix_new_trig]

                        # currently we require at least 4 phones trigger to create an event, may change after we have network data
                        if trig_buf[ix_old_trig]['associate_trig'] >= 4:  # (Here may change to 0.3 * active phones)

                            # we only need get station_list here
                            t1 = current_time.strftime('%Y%m%d %H:%M:%S.3%f')
                            t0 = (current_time - pd.Timedelta(minutes = 30)).strftime('%Y%m%d %H:%M:%S.3%f')

                            old_event_id = event_id
                            event_id, detected_event, trig_buf = self._check_create_event(trig_buf, tree_steady, tree_trig, ix_old_trig, event_id, current_time)

                            if event_id > old_event_id:
                                for ix in trig_list:
                                    trig_buf[ix]['visited_for_association'] += [event_id]
                                    trig_buf[ix]['associate_event'] = event_id
                                    trig_buf[ix_new_trig]['associate_event'] = event_id
                                    trig_buf[ix_old_trig]['associate_event'] = event_id
                                    #print ix_old_trig, ix_new_trig
                                return event_id, detected_event, trig_buf
            else:  # if the event is already associated with an event
                trig_buf[ix_new_trig]['visited_for_eventDetection'] = 1

        return event_id, None, trig_buf


    def _check_create_event(self, trig_buf, tree_steady, tree_trig, ix_old_trig, event_id, current_time):
        '''
        Function to check if the triggers satisfy to create an earthquake if there are more than
        4 triggers are associated.
        '''
        associated_triggers_ix = trig_buf[ix_old_trig]['trig_list']

        associated_triggers = [trig_buf[trig_ix] for trig_ix in associated_triggers_ix]

        event_lat = np.mean([trig['lat'] for trig in associated_triggers])
        event_lon = np.mean([trig['lon'] for trig in associated_triggers])
        event_time = np.min([trig['trig_time'] for trig in associated_triggers])

        trig_list = [ [trig['trig_time'], trig['lat'], trig['lon']] for trig in associated_triggers]

        # check if the percentage of stations associated within 10km is >= %60, may change after we have network data

        # get the triggers within 10km of the newly created event
        trig_within_10km = sum(calculate_dist(trigger[1], trigger[2], event_lat, event_lon) <= 10.  for trigger in trig_list)

        #trig_within_10km = len(associated_triggers_ix)

        # get the total phones within 10km of the newly created event
        x_ref, y_ref, z_ref = self._to_Cartesian(event_lat, event_lon)
        dist_r = self._kmToDIST(10.)
        ix = tree_steady.query_ball_point((x_ref, y_ref, z_ref), dist_r)
        total_stations_within_10km = len(ix)
        #print(trig_within_10km, total_stations_within_10km, event_lat, event_lon, ix_old_trig, current_time)
        # print trig_within_10km, total_stations_within_10km
        # if meet the 60% rule
        # in case there's no heartbeat in the area
        if (int(total_stations_within_10km) < 1):
            print("No stations near event trigger.")
            return event_id, None, trig_buf
        elif float(trig_within_10km) / float(total_stations_within_10km) >= 0.6:

            # print associated_triggers_ix
            use_gridSearch = False
            if use_gridSearch:

                dist_time_arr = []
                #for ix in associated_triggers_ix:
                #    row = trig_buf[ix]
                #    dist_time_arr.append([ix, row['trig_time'], row['lat'], row['lon'], row['pga']])
                for key, trig in trig_buf.items():
                    row = trig
                    dist_time_arr.append([ix, row['trig_time'], row['lat'], row['lon'], row['pga']])

                dist_time_arr = np.array(dist_time_arr)
                results = self._grid_search_loc(dist_time_arr, event_lat, event_lon, evdp = 15)

                _, event_lat, event_lon, event_time = results

            # this is a good trigger, and exit the loop
            print('Earthquake detected!', str(datetime.now()))
            event_id += 1
            # print trig_within_10km, total_stations_within_10km
            est_mag_list = []
            trigIDs = []
            # update the trigger buffer for future use
            for trig_ix in associated_triggers_ix:
                trig_buf[trig_ix]['associate_event'] = event_id

                #calculate the magnitude
                associated_trig = trig_buf[trig_ix]
                est_mag_list.append(self._estimate_magnitude(associated_trig, event_lat, event_lon))

                # add the trigger id to the list
                trigIDs.append(str(trig_ix) + '_' + str(associated_trig['trig_time']))

            # get the event average magnitude
            est_mag = np.mean(est_mag_list)

            # TODO:
            alert_time = current_time
            detected_event = [event_time, alert_time, event_lat, event_lon, event_id, trig_within_10km, total_stations_within_10km, est_mag, trigIDs]

            print('Earthquake estimated parameters are:')
            print(detected_event)

            return event_id, detected_event, trig_buf
        else:
            return event_id, None, trig_buf


    def _associate_additional_trig(self, trig_buf, detected_event):
        '''
        This function is to associate triggers in the trigger buffer with the event we
        created.

        Input:
        trig_buf - a buffer contains triggers, type is ordered dictionary.
        detected_event - a list contains the information of the events we created. schema is:
        [event_time, event_lat, event_lon, event_id, trig_within_10km, total_stations_within_10km]

        Return:
        updated trig_buf
        '''

        # get the event information
        event_lat = detected_event[2]
        event_lon = detected_event[3]
        event_time = detected_event[0]
        event_id = detected_event[4]

        # loop through the triggers in the buffer and associate with the event

        for key, trig in trig_buf.items():

            if event_id in trig['visited_for_association']:

                return trig_buf
            else:
                trig_buf[key]['visited_for_association'] += [event_id]

                if trig['associate_event'] == 0:
                    trig_time = trig['trig_time']
                    trig_lat = trig['lat']
                    trig_lon = trig['lon']

                    travel_time = trig_time - event_time

                    trigger_distance = calculate_dist(trig_lat, trig_lon, event_lat, event_lon)

                    # do not associate a trigger if it is farther that 300km
                    if (travel_time < 300. / 1.5 and trigger_distance < 300):

                        # check for a near source trigger within 20km of the event
                        if (trigger_distance < 20 and travel_time >= -2.0 \
                            and travel_time < 20.0):
                            trig_buf[key]['associate_event'] = event_id

                        else:
                            # tt_min, tt_max got from the association boundary of our historical data

                            tt_min = 0.3 * trigger_distance - 12.
                            tt_max = 0.31449 * trigger_distance + 18

                            if (travel_time >= tt_min and travel_time <= tt_max):
                                trig_buf[key]['associate_event'] = event_id
        return trig_buf

    def detector(self, phones_steady, df_trig):
        # generate tree for steady phones
        lats_1d = np.array(phones_steady)[:, 0]
        lons_1d = np.array(phones_steady)[:, 1]
        x, y, z = zip(*map(self._to_Cartesian, lats_1d, lons_1d))

        # create the KD-tree using the 3D cartesian coordinates
        coordinates = list(zip(x, y, z))
        tree_steady = spatial.cKDTree(coordinates)

        # generate tree for trigers
        lats_1d = df_trig.latitude.values
        lons_1d = df_trig.longitude.values
        x, y, z = zip(*map(self._to_Cartesian, lats_1d, lons_1d))

        # create the KD-tree using the 3D cartesian coordinates
        coordinates = list(zip(x, y, z))
        tree_trig = spatial.cKDTree(coordinates)

        df_heartbeat_eq = pd.DataFrame(phones_steady, columns=['latitude', 'longitude'])
        df_heartbeat_eq['deviceId'] = range(len(df_heartbeat_eq))
        df_heartbeat_eq['datetime'] = self.evtime_ts
        df_heartbeat_eq = df_heartbeat_eq.set_index('datetime')
        df_heartbeat_eq['ts'] = df_heartbeat_eq.index.astype(np.int64)/1000000
        df_heartbeat_eq['hbSource'] = 1

        df_trig_eq = df_trig
        df_trig_eq['deviceid'] = [str(x) for x in range(len(df_trig_eq))]
        df_trig_eq['tt'] = df_trig_eq.index.astype(np.int64)/1000000

        df_trig_eq['l'] = df_trig_eq[['longitude', 'latitude']].values.tolist()
        df_trig_eq = df_trig_eq.drop('latitude', 1)
        df_trig_eq = df_trig_eq.drop('longitude', 1)
        df_trig_eq['tf'] = 3

        detected_events_withEvents = []
        events_triggers = []

        t0 = self.evtime
        t0_ts = pd.Timestamp(t0).value/ 1e9

        count = []
        detected_events = []
        event_id = 0

        t_start = (self.evtime_ts - timedelta(seconds = 30)).strftime('%Y%m%d %H:%M:%S.3%f')
        t_end = (self.evtime_ts + timedelta(seconds = 40)).strftime('%Y%m%d %H:%M:%S.3%f')
        start_times, end_times = self._generate_times(t_start, t_end, win_lenth = 20., win_step = 100)

        trig_buf = OrderedDict()
        cur_t = 0

        trig_associate_pool = []
        key_list = []
        errors = []

        for t_start, t_end in zip(start_times, end_times):
            #print(t_start, str(datetime.now()))
            # need convert to string first
            #df_select = df_trig.ix[t_start.strftime('%Y-%m-%d %H:%M:%S'):t_end.strftime('%Y-%m-%d %H:%M:%S')]
            t0 = t_start.strftime('%Y%m%d %H:%M:%S')
            t1 = t_end.strftime('%Y%m%d %H:%M:%S')

            df_select = df_trig_eq.ix[t0:t1]

            # prepare the trigger pool
            ANN_triggers_pool = self._generate_ANN_triggers_pool(df_select)

            ##################################################################################################
            ####### This part is to update the trigger buffer, add new triggers, and delete the old ones
            # loop through trigger pool and prepare the trig_buf
            for key, trig_info in ANN_triggers_pool.items():
                if key not in trig_buf.keys():
                    trig_buf[key] = trig_info

            ts_start = t_start.value / 1e9
            # remove the ones older than 20 sec

            # fix for python 3 problems: https://stackoverflow.com/questions/5384914/how-to-delete-items-from-a-dictionary-while-iterating-over-it
            # short answer, in python 3, trig_buf.items() returns an iterator instead of a copy that cause problems
            key_list = list( trig_buf.keys() )
            for key in key_list:
                trig_info = trig_buf[key]
                if trig_info['trig_time'] < ts_start:
                    del trig_buf[key]
                else:
                    # since the trig_buf contains sorted time, so whenever it is not satisfy, we can break
                    break
            ##################################################################################################

            num_trig_inBuff = len(trig_buf)

            # check whether triggers are associated with an event, if we don't check this, we may
            # create more events from a single earthquake
            if len(detected_events) != 0:
                if np.abs(detected_events[-1][0] - ts_start) < 600:
                    trig_buf = self._associate_additional_trig(trig_buf, detected_events[-1])

            # check to create new earthquakes.
            if len(trig_buf) != 0:
                _ = []
                # adding the triggers that associate with the event
                for key, value in trig_buf.items():
                    if ((value['associate_event'] ==1)):
                        _.append([value['lat'], value['lon'], value['pga'], value['trig_time']])

                if len(_) >0 :
                    trig_associate_pool.append([t_end, _])
                #if win_end > 0 and win_end < 8:
                #    print win_end
                #    eew_ui.plot_trig_stations(station_list, trig_buf, evla = 37.5, evlo = -121.5, idx = i)
                max_association_tt_time = 20
                event_id, detected_event, trig_buf = self._detect_event(trig_buf, event_id,
                                   max_association_tt_time, t_end, tree_steady,tree_trig)

                if detected_event is not None:
                    detected_events.append(detected_event)
                    print('###################################################')
                    alertTime_from_origin = detected_event[1].value/1e9 - t0_ts
                    print('Alert Time - Origin Time: ' + str(alertTime_from_origin))
                    dist_err = calculate_dist(self.evla, self.evlo,detected_event[2], detected_event[3])
                    print('Distance error: {:.2f} km'.format(dist_err))
                    mag_error = detected_event[7] - self.mag
                    print('Real mag: {:.1f}, estimate mag: {:.1f}'.format(self.mag, detected_event[7]))
                    print('###################################################')
                    detected_events_withEvents.append([detected_events, self.event, trig_associate_pool])
                    origin_time_error = detected_event[0] - self.evtime_ts.timestamp()

                    errors = [mag_error, dist_err, origin_time_error, alertTime_from_origin]

        return detected_events_withEvents, df_trig_eq, df_heartbeat_eq, errors
