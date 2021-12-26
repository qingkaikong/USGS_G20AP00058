import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_times(t0, t1, win_lenth = 20., win_step = 1000.):
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

    #print(t0)
    freq = str(int(win_step)) + 'L'
    start_times = pd.date_range(t0, t1 - pd.Timedelta(seconds = win_lenth), freq = freq, tz= "UTC")
    end_times = pd.date_range(t0 + pd.Timedelta(seconds = win_lenth), t1, freq = freq, tz= "UTC")
    return start_times, end_times


def make_figures(stla, stlo, alpha, t_end):

    #import matplotlib
    #matplotlib.use("Agg")


    #FFMpegWriter = manimation.writers['ffmpeg']
    #metadata = dict(title=filename, artist='Matplotlib',
    #        comment='Movie support!')
    #writer = FFMpegWriter(fps=fps, metadata=metadata)

    #win_length = datetime.timedelta(seconds=win_length)
    #start_time0 = event_time - win_length
    #time_step = datetime.timedelta(seconds=1)

    #end_time0 = start_time0 + win_length
    #event_id = 1
    #est_loc = {}
    '''
    if evla and evlo is not None:
        llat = evla - 0.27
        ulat = evla + 0.27
        llon = evlo - 0.35
        ulon = evlo + 0.35

        if which_eq == 'La Habra':
            dt = est_t
            est_loc = dict(((5,5.2), (6, 5.2), (7, 5.1)))
        elif which_eq == 'Parkfield':
            #This is where I make the first estimation, not sure why it is need add 1
            dt = est_t + 1
            est_loc = dict(((6,5.5), (7, 5.6), (8, 5.5)))
    '''

    fig = plt.figure(figsize = (10, 12))
    map_ax = fig.add_axes([0.03, 0.13, 0.94, 0.82])
    m = Basemap(projection='merc', lon_0=-121.36929, lat_0=37.3215,
            llcrnrlon=llon,llcrnrlat=llat- 0.01,urcrnrlon=ulon,urcrnrlat=ulat + 0.01,resolution='l')

    m.drawcoastlines()
    m.fillcontinents(color='#cc9966',lake_color='#99ffff', alpha = 0.3)

    lons = stlo
    lats = stla
    x, y = m(lons,lats)

    rgba_colors = np.zeros((len(stlo),4))
    rgba_colors[:,0] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alpha
    m.scatter(x,y,30,marker='o',color=rgba_colors, lw = 0, zorder = 10)
    plt.title(t_end.strftime('%Y-%m-%d %H:%M:%S'))
    plt.show()
    '''
    est_mag = ''
    with writer.saving(fig, filename, 400):
        for iFrame in xrange(Frames):

            trig_buffer = df_trig.ix[start_time0:end_time0]

            start_time0 += time_step
            end_time0 += time_step
            if wait_sec is not None:
                if wait_sec >= iFrame:
                    continue

            trig_lon = np.array(trig_buffer['lon'].tolist())
            trig_lat = np.array(trig_buffer['lat'].tolist())
            fig.clear()
            if draw_details:
                m.drawcoastlines()
                m.fillcontinents(color='#cc9966',lake_color='#99ffff', alpha = 0.3)

            if evla and evlo is not None:
                if iFrame - evtime >= 0:
                    x_0, y_0 = m(evlo, evla)
                    m.plot(x_0, y_0, 'r*', markersize=25, label = 'Event', lw = 0)
                    equi(m, evlo, evla, 10., color = 'k', alpha = 0.5, label = '10 km' if iFrame == 5 else "")
                    equi(m, evlo, evla, 20., color = 'k', alpha = 0.5, label = '20 km' if iFrame == 5 else "")
                    equi(m, evlo, evla, 30., color = 'k', alpha = 0.5, label = '30 km' if iFrame == 5 else "")

                    if est_t is not None:
                        print est_t, iFrame + evtime
                        if dt <=  iFrame + evtime :
                            if iFrame + evtime - est_t < 4:
                                #only plot 3 updates of the location of the updates
                                est_lon = trig_lon.mean()
                                est_lat = trig_lat.mean()

                            x_0, y_0 = m(est_lon, est_lat)
                            print est_lon, est_lat
                            m.plot(x_0, y_0, 'b*', markersize=25, label = 'Estimated Event', zorder = 10, lw =0)
                else:
                    continue
                plt.title(str(iFrame - evtime) + ' s ' + title, fontsize=32)


                if which_eq == 'La Habra':
                    #this if for la habra
                    m.drawmapscale(-118.1, 33.71, -118, 33.9, 30, barstyle='fancy',fontsize=20, zorder = 10)
                    # draw lat/lon grid lines
                    m.drawparallels(np.arange(33.2, 35.2, 0.5), labels=[1,0,0,0], linewidth=0,fontsize=24)
                    m.drawmeridians(np.arange(-118.7, -116.2, 0.5), labels=[0,0,0,1], linewidth=0,fontsize=24)
                elif which_eq == 'Napa':
                    #this if for Napa
                    #map.drawmapscale(-123.0, 37.6, -122, 38, 30, barstyle='fancy')
                    # draw lat/lon grid lines
                    m.drawparallels(np.arange(37.5, 38.8, 0.3), labels=[1,0,0,0], linewidth=0)
                    m.drawmeridians(np.arange(-123.2, -121.5, 0.3), labels=[0,0,0,1], linewidth=0)

                elif which_eq == 'Parkfield':
                    #this if for Napa
                    #map.drawmapscale(-120.65, 35.63, -120.5, 35.7, 30, barstyle='fancy')
                    m.drawmapscale(-120.55, 35.6, -120.5, 35.7, 30, barstyle='fancy',fontsize=20, zorder = 10)
                    # draw lat/lon grid lines
                    m.drawparallels(np.arange(35.1, 36.4, 0.3), labels=[1,0,0,0], linewidth=0,fontsize=24)
                    m.drawmeridians(np.arange(-121.1, -119.7, 0.3), labels=[0,0,0,1], linewidth=0,fontsize=24)
                    #llcrnrlon=-120.8,llcrnrlat=35.6, urcrnrlon=-120,urcrnrlat=36)
            else:
                equi(m, -122, 38, 10., color = 'k', alpha = 0.5, label = '10 km' if iFrame == 5 else "")
                plt.title(str(iFrame -20) + ' s ' + title, fontsize = 24)

            lons = np.array(station_list)[:, 3].astype(np.float)
            lats = np.array(station_list)[:, 2].astype(np.float)
            x, y = m(lons,lats)
            m.scatter(x,y,50,marker='o',color='#A0A0A0', lw = 0, zorder = 10)

            x, y = m(trig_lon,trig_lat)
            m.scatter(x,y,50,marker='o',color='#FF33FF', zorder = 10, lw = 0)

            #plot the estimates of the magnitudes from script 06_test_classifier.py
            try:
                est_mag = 'M=' + str(est_loc[iFrame])

                plt.annotate(est_mag, xy=(0.68, 0.84), xycoords = 'figure fraction', fontsize = 32, fontweight='bold')
            except:
                plt.annotate(est_mag, xy=(0.68, 0.84), xycoords = 'figure fraction', fontsize = 32, fontweight='bold')

            plt.savefig(str(iFrame) + '.pdf', bbox_inches = 'tight',pad_inches = 0)
            plt.close()
    '''

def make_color(t_start, t):

    #t_start = t_start.tz_localize(None)
    delta_t = (t - t_start)

    # get the time difference of the trigger time and the start
    # of the window
    #sec = delta_t.seconds + delta_t.microseconds / 1e6
    sec = delta_t.seconds
    return 1. - (20 - sec) / 20.

def make_color_events(t_start, t):

    #t_start = t_start.tz_localize(None)
    delta_t = (t - t_start)

    # get the time difference of the trigger time and the start
    # of the window
    sec = delta_t.seconds + delta_t.microseconds / 1e6

    return 1 - (40 - sec) / 40

def shoot(lon, lat, azimuth, maxdist=None):
    """Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq
    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.

    EPS= 0.00000000005
    #if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
    #    alert("Only N-S courses are meaningful, starting at a pole!")

    a=6378.13/1.852
    f=1/298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf==0):
        b=0.
    else:
        b=2. * np.arctan2 (tu, cf)

    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while (np.abs (y - c) > EPS):

        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
              d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180./np.pi
    glat2 *= 180./np.pi
    baz *= 180./np.pi

    return (glon2, glat2, baz)

def equi(m, centerlon, centerlat, radius, *args, **kwargs):
    '''
    plot circles on basemap
    '''
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    #m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X,Y = m(X,Y)
    plt.plot(X,Y,**kwargs)


def equi_update(handle, m, centerlon, centerlat, radius, *args, **kwargs):
    '''
    plot circles on basemap
    '''
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    #m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X,Y = m(X,Y)

    handle.set_data(X,Y)
