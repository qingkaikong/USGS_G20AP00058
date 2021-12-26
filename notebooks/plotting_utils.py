import datetime
import os
import glob
import pickle

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec

import utils
from geo_utils import calculate_dist


def plot_comparison_plot(df_trig,
                         df_heartbeat,
                         df_epic,
                         df_dm,
                         earthquake, 
                         max_dist_km = 100.,
                         min_time_s = -30,
                         max_time_s = 60,
                         dist_bins_km = np.arange(0,105,5),
                         time_bins_s = np.arange(-30, 61, 1),
                         save_folder=None, 
                         myshake_legacy=True,
                         plot_simulation=False,
                         simulation_path=None,
                         include_dm_alert_updates=False):
    """function to plot the distance vs time. 
    
    Args:
        df_trig: myshake trigger dataframe. 
        df_heartbeat: myshake heartbeat dataframe. 
        df_epic: epic trigger dataframe. 
        df_dm: dm event dataframe. 
        earthquake: earthquake dictionary.
        max_dist_km: max distance in km to plot.
        min_time_s: min time in sec to plot.
        max_time_s: max time in sec to plot. 
        dist_bins_km: distance bins for spatial statistics. 
        time_bins_s: time bins for temporal statistics. 
        save_folder: the folder to save images. 
        myshake_legacy: wether the data is from legacy myshake or not. 
        plot_simulation:
        simulation_path:
        include_dm_alert_updates:
    
    Returns:
        None
    """
    
    # get earthquake information to easy varables 
    mag = earthquake['mag']
    evla = earthquake['latitude']
    evlo = earthquake['longitude']
    evdp = earthquake['depth']
    evtime = earthquake['time']
    place = earthquake['place']
    evid = earthquake['id']
    
    ################################### Setup Time ###################################
    orig_time, t_start, t_end = utils.slice_time_from_eq_origin(
        earthquake,
        min_time_s,
        max_time_s)
    
    ################################### P and S wave ###################################
    ep_dist_km, tp_sec, ts_sec = utils.get_p_and_s(max_dist_km + 10, evdp)
    
    ################################### Prepare data ###################################
    # get number of triggers before the first alert
    df_trig, df_heartbeat, df_heartbeat_unique, df_stalta_trig, df_steady, df_epic = \
        utils.prepare_myshake_data(earthquake,
                                   orig_time,
                                   df_trig,
                                   df_heartbeat,
                                   df_epic,
                                   t_start,
                                   t_end,
                                   max_dist_km=max_dist_km,
                                   myshake_legacy=myshake_legacy,
                                     )
    
    #################################### EPIC events #####################################
    # select the e2created events at the time range
    df_dm = df_dm[t_start:t_end]
    df_dm = df_dm[df_dm['system']=='dm']
    df_dm = df_dm.assign(tt_rel= df_dm['alert_time'] - orig_time)
    if not include_dm_alert_updates:
        df_dm = df_dm[df_dm['type']=='new']

    ################################### Distance Bins ###################################
    online_sum = utils.get_spatial_statistics(
        df_heartbeat_unique['dist_km'], bins=dist_bins_km)

    stalta_sum = utils.get_spatial_statistics(
        df_stalta_trig['dist_km'], bins=dist_bins_km)
    
    ann_sum = utils.get_spatial_statistics(
        df_trig['dist_km'], bins=dist_bins_km)
    
    steady_sum = utils.get_spatial_statistics(
        df_steady['dist_km'], bins=dist_bins_km)

    epic_sum = utils.get_spatial_statistics(
        df_epic['dist_km'], bins=dist_bins_km)
    
    ################################### Time Bins ###################################
    epic_t_sum = utils.get_spatial_statistics(
        df_epic['tt_rel'], bins=time_bins_s)
    
    stalta_t_sum = utils.get_spatial_statistics(
        df_stalta_trig['tt_rel'], bins=time_bins_s)
    
    ann_t_sum = utils.get_spatial_statistics(
        df_trig['tt_rel'], bins=time_bins_s)

    ################################### If plot simulation ###################################
    if plot_simulation:
        simulation_file = glob.glob(os.path.join(simulation_path, evid + '*_simulations*'))[0]
        simulation_dict = pickle.load(open(simulation_file, 'rb'))

        i = 0   
        # read in all the phones
        phones = simulation_dict[i]['phones']
        steady_phones = simulation_dict[i]['phones_steady']
        df_trig_simulation = simulation_dict[i]['df_trig']

        # get the total phones distance km
        total_phones_dist = np.array([calculate_dist(
            earthquake['latitude'],
            earthquake['longitude'],
            stla,
            stlo) for stla, stlo in phones
            ])
                
        # get the steady phones distance km
        steady_phones_dist = np.array([calculate_dist(
            earthquake['latitude'],
            earthquake['longitude'],
            stla,
            stlo) for stla, stlo in steady_phones
            ])
        
        # within max_dist_km km
        total_phones_dist_select = total_phones_dist[total_phones_dist < max_dist_km]
        steady_phones_dist_select = steady_phones_dist[steady_phones_dist < max_dist_km]
        df_trig_simulation = df_trig_simulation[t_start:t_end]
        df_trig_simulation = df_trig_simulation[df_trig_simulation['dist_km']<max_dist_km]

        triggers_dist_select = df_trig_simulation['dist_km']

        total_simulation_sum = utils.get_spatial_statistics(
            total_phones_dist_select, bins=dist_bins_km)
        
        steady_simulation_sum = utils.get_spatial_statistics(
            steady_phones_dist_select, bins=dist_bins_km)
        
        trig_simulation_sum = utils.get_spatial_statistics(
            triggers_dist_select, bins=dist_bins_km)

        trig_simulation_t_sum = utils.get_spatial_statistics(
            df_trig_simulation['tt_rel'], bins=time_bins_s)

    ################################## Plot figure ###################################
    fig = plt.figure(figsize=(16, 10))
    plt.style.use('seaborn-poster')
    fig.suptitle(f'M{mag} event at depth {evdp} km, on {evtime[:-5]}, at {place}', 
                fontsize=20)

    gs = gridspec.GridSpec(3, 4)

    ############## Scatter plot ##############
    ax1 = plt.subplot(gs[0:2,0:3])

    ax1.plot(ep_dist_km, tp_sec, 'g', label='P-wave')
    ax1.plot(ep_dist_km, ts_sec, 'r', label='S-wave')

    if plot_simulation:
        ax1.scatter(df_trig_simulation['dist_km'], df_trig_simulation['tt_rel'], c='k', 
                s=30,label='Simulated triggers', zorder=10)
    else:
        ax1.scatter(df_stalta_trig['dist_km'], df_stalta_trig['tt_rel'], c='k', 
                s=30,label='STA/LTA', zorder=10)

        ax1.scatter(df_trig['dist_km'], df_trig['tt_rel'],s=40, c='m', 
                    label='ANN triggers', marker='^', zorder=10,
                    alpha=0.8)

    ax1.scatter(df_epic['dist_km'], df_epic['tt_rel'],s=40, c='g', 
                label='EPIC triggers', marker='s', zorder=9)

    ix_new = df_dm['type'] == 'new'
    if include_dm_alert_updates:
        ix_update = df_dm['type'] == 'update'
        ax1.hlines(df_dm['tt_rel'][ix_update], 0,
        max_dist_km, linestyles=':',
        linewidth=2, colors='grey')
    ax1.hlines(df_dm['tt_rel'][ix_new], 0,
        max_dist_km, linestyles=':',
        linewidth=3, colors='c')
    
    ax1.set_xticklabels([])

    ax1.set_ylabel('Time since origin of the earthquake (sec)')
    ax1.set_ylim(min_time_s, max_time_s)
    ax1.set_xlim(0, max_dist_km)
    plt.legend(loc=4)

    ############## Location Map ##############
    ax2 = plt.subplot(gs[2,3:], projection=ccrs.PlateCarree())
    ax2.set_extent([-125, -114, 32, 42], crs=ccrs.PlateCarree())

    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.OCEAN)
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    ax2.add_feature(cfeature.LAKES, alpha=0.5)
    ax2.add_feature(cfeature.RIVERS)
    ax2.add_feature(cfeature.STATES)
    ax2.scatter(evlo, evla, marker='*', s=80, c='r')
    
    ############## Distance Histogram ##############
    ax3 = plt.subplot(gs[2,0:3])
    if plot_simulation:
        ax3.bar(dist_bins_km[1:], total_simulation_sum, width = 5, color='b', label='PHONE')
        #ax3.bar(dist_bins_km[1:], steady_simulation_sum, width = 4, color='c', label='STEADY' )
        ax3.bar(dist_bins_km[1:], trig_simulation_sum, width = 4, color='k', label='TRIGGER')
        ax3.bar(dist_bins_km[1:], epic_sum, width = 3, color='g', label='EPIC')
    else:
        ax3.bar(dist_bins_km[1:], online_sum, width = 5, color='b', label='PHONE')
        #ax3.bar(dist_bins_km[1:], steady_sum, width = 4, color='c', label='STEADY')
        ax3.bar(dist_bins_km[1:], stalta_sum, width = 4, color='k', label='STA' )
        ax3.bar(dist_bins_km[1:], ann_sum, width = 3, color='m', label='ANN',
                    alpha=0.8)
    ax3.set_xlim(0, max_dist_km)
    plt.legend()
    plt.xlabel('Epicentral distance (km)')
    
    ############## Time Histogram ##############
    ax4 = plt.subplot(gs[:2,3:])
    if plot_simulation:
        ax4.barh(time_bins_s[1:], trig_simulation_t_sum, color='k', height=0.8, label='Phone Trigger')
    else:
        ax4.barh(time_bins_s[1:], stalta_t_sum, color='k', height=0.8, label='STA' )
        ax4.barh(time_bins_s[1:], ann_t_sum, color='m', height = 0.7, label='ANN',
                    alpha=0.8)
    
    ax4.barh(time_bins_s[1:], epic_t_sum, color='g', height=1, label='EPIC')
    if plot_simulation:
        x_max = np.max([epic_t_sum, trig_simulation_t_sum])
    else:
        x_max = np.max([epic_t_sum, stalta_t_sum, ann_t_sum])
    if include_dm_alert_updates:
        ix_update = df_dm['type'] == 'update'
        ax4.hlines(df_dm['tt_rel'][ix_update], 0,
        max_dist_km, linestyles=':',
        linewidth=2, colors='grey')
    ax4.hlines(df_dm['tt_rel'][ix_new], 0,
        x_max+5, linestyles=':',
        linewidth=3, colors='c')
    ax4.set_ylim(min_time_s, max_time_s)
    ax4.set_yticklabels([])
    plt.legend(loc=4)
    
    plt.tight_layout()
    if save_folder:
        if plot_simulation:
            output_name = evid + '_simulation.png'
        else:
            output_name = evid + '.png'
        plt.savefig(os.path.join(save_folder, output_name), transparent = False, 
                    bbox_inches = 'tight', pad_inches = 0.1)
        plt.close()
    else:
        plt.show()


def plot_scatter_on_map(x,
                        y,
                        c,
                        s,
                        vmin = 0,
                        vmax = 100,
                        clabel=None,
                        filename=None,
                        resolution='50m',
                        title=None,
                        llat=32,
                        ulat=42.2,
                        llon=-125,
                        ulon=-114):
    
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([llon, ulon, llat, ulat], crs=ccrs.PlateCarree())
        
    ax.add_feature(cfeature.LAND.with_scale(resolution))
    ax.add_feature(cfeature.OCEAN.with_scale(resolution))
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution))
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), linestyle=':')
    ax.add_feature(cfeature.LAKES.with_scale(resolution), alpha=0.5)
    ax.add_feature(cfeature.RIVERS.with_scale(resolution))
    ax.add_feature(cfeature.STATES.with_scale(resolution))
    
    cs = ax.scatter(x, y, marker='o',
                    s=s, c=c, vmin=vmin, 
                    vmax=vmax, zorder=10)
    cbar = plt.colorbar(cs)
    cbar.set_label(clabel)
    
    plt.title(title)
    
    if filename:
        plt.savefig(filename, bbox_inches = 'tight',
        pad_inches = 0.1)
    else:
        plt.show()