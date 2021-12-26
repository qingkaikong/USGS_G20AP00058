#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:07:34 2019

Class to sample phones from world population

@author: Qingkai Kong
"""
#%% define cell
import pandas as pd
import numpy as np
from collections import Counter
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class PhoneSampler(object):
  STEP = 0.00833333333333387 / 2.0
  
  def __init__(self, fname):
    self.fname = fname
    self.df_pop= self.read_pop_file(self.fname)

  @staticmethod
  def read_pop_file(fname):
    """
    Helper function to read the population data into DataFrame
    """
    df_pop = pd.read_hdf(fname, 'count')
    return df_pop

  def get_random_point_in_polygon(self, minx, miny, maxx, maxy):
    """
    Function to get a random point within a polygon
    """

    lon = np.random.uniform(minx, maxx)
    lat = np.random.uniform(miny, maxy)

    return lat, lon
  
  def _plot_phones(self, phones, llat, ulat, llng, ulng, filename):
    """
    Helper function to plot the sampled phones
    """
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    ax.set_extent([llng, ulng, llat, ulat])
    
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    #ax.add_feature(cfeature.COASTLINE, linestyle='-')
    
    # Note that, using this 10m resolution will take a long time to
    # generate the figure. Use 50m instead
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', \
                                           edgecolor='k', 
                                           facecolor=cfeature.COLORS['land'])
    
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', \
                                           edgecolor='k', 
                                           facecolor=cfeature.COLORS['water'])
    ax.add_feature(land)
    ax.add_feature(ocean)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    ax.set_xticks(np.arange(llng, ulng+0.1, 0.5))
    ax.set_yticks(np.arange(llat, ulat+0.1, 0.5))
    
    ax.scatter(phones[:, 1], phones[:, 0], c='b', s=.5, alpha=0.6, \
               zorder=10)
    
    if filename:
        plt.savefig(filename)
    plt.show()

  def sample_phones(self, percentage, min_users, llat=36.5, ulat=38.5, \
                    llng=-123, ulng=-121, plot=False, filename=None):
    """
    function to sample phones based on a percentage of the total population
    
    """
    
    # get the population data to selected region
    df_pop = self.df_pop[(self.df_pop.Lat >= llat) & (self.df_pop.Lat <= ulat) & \
                    (self.df_pop.Lon>=llng) & (self.df_pop.Lon <= ulng)]
    
    cell_pops = df_pop['Values'].values.copy()
    cell_pops[cell_pops <= min_users] = 0
    cellIDs = df_pop.index
    #total population
    total_pop = np.sum(cell_pops)
    #estimated total number of users in the population
    total_myshake_pop = int(total_pop*percentage)

    prob_sample = cell_pops/np.sum(cell_pops)
    sample_cells = np.random.choice(cellIDs,size=total_myshake_pop,replace=True,p=prob_sample)

    cell_counts = Counter(sample_cells)
    user_counts = [cell_counts[key] for key in cellIDs]
    phones = []
    for lat_c, lng_c, nphones in zip(df_pop['Lat'],df_pop['Lon'], user_counts):
        minx = lng_c - self.STEP
        miny = lat_c - self.STEP
        maxx = lng_c + self.STEP
        maxy = lat_c + self.STEP

        for i in range(int(nphones)):
            lat, lng = self.get_random_point_in_polygon(minx, miny, maxx, maxy)
            phones.append([lat,lng])

    phones = np.array(phones)
    
    if plot:
        self._plot_phones(phones, llat, ulat, llng, ulng, filename)
    
    return phones
    
  def sampling_from_android_distribution(self):
    """
    This function will get phones from our real distribution
    Args:

    Returns:
      None.

    """
    ##TODO Qingkai
    pass
    
#%% run cell
if __name__ == '__main__':
  # test the whole class
  phoneSampler = PhoneSampler('./data/database_1km_2020.h5')
  phones = phoneSampler.sample_phones(0.0001, min_users=5, plot=False, \
                                      filename='phones_sampled.png')
  print(len(phones))
