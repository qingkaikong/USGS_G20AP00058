# %% import packges
import glob
import os
import pickle 

from joblib import Parallel, delayed
import pandas as pd

from phone_sampler import PhoneSampler
import trigger_generator

# %% setup 
data_path = '../data/ground_truth/'
output_folder = '../data/simulation_data_10p/'

num_simulations = 1
# note, this timezone is very important, if it is not setting correctly
# it will have the wrong number of steady phones
timezone = 'US/Pacific'

# population sampling parameters
phone_user_percentage = 0.1 
min_users_to_sample_phone = 5

# region limits from epicenter
lat_dist_from_epicenter = 2.
lng_dist_from_epicenter = 2.

# configuration
# let's set the configuration 

config = {}
# p wave velocity in km/s
config['v_p'] = 6.1
# s wave velocity in km/s
config['v_s'] = 3.55
# due to unknown reason, we give the triggers can be used a discount, 
# for example, 0.8 means 80% of the generated triggers will be used, 
# the other 20% due to various reasons, unavailable, i.e. bad data, 
# missing packet, etc. 
config['discount_factor'] = 0.8
# uncertainty for P wave trigger time (it is a half-normal distribution)
# with mean at the estimate P wave time, sigma is the standard deviation
config['p_time_sigma'] = 0.5
# uncertainty for S wave trigger time (it is a normal distribution)
# with mean at the estimate S wave time, sigma is the standard deviation
config['s_time_sigma'] = 0.5
# since the phone's amplitude is usually larger than the seismic station amplitude
# here we use an amplification_factor to control it. How many times you want the
# amplitude larger than the Cua et al. relationship amplitudes
config['amplification_factor'] = 2.0
# when generate amplitude envelope, how long in seconds you want the waveform after decay
config['time_after_decay'] = 30
# sampling rate for the generated waveforms
config['sampling_rate'] = 25
# this will be used to generate the phone threshold using a normal distribution
# if the generated threshold is smaller than the phone_threshold_bottom, it will 
# be replaced by this phone_threshold_bottom value. Basically, if the amplitude 
# is larger than the phone threshold, it will trigger the phone. 
config['phone_threshold_g_mean'] = 0.01
config['phone_threshold_g_std'] = 0.01
config['phone_threshold_g_bottom'] = 0.005
# this controls the percentage of the triggers we know it is P or S, if it is
# 0.7, it means about 70% of the triggers we know it is correct P or S trigger.
config['p_or_s_percentage'] = 0.7

# %% load data
pop_folder = '/Users/qingkaikong/Google Drive/research/research_2019/Google_Work/03_simulation_platform/data'
phoneSampler = PhoneSampler(os.path.join(pop_folder, 'database_1km_2020.h5'))

# read myshake relationships
steady_ratio = pickle.load(open('./data/steady_phone_ratio_hourly.pkl', 'rb'), encoding='latin1')
trig_rate = pickle.load(open('./data/trig_stalta_rate_dict.pkl', 'rb'), encoding='latin1')

# read in myshake network latencies
myshake_network_latency = pd.read_pickle('../data/myshake_network_latency_20191017_20200301.pkl')

# %% simulation run
def generate_simulated_triggers(filename):
  fname = os.path.basename(filename)
  
  simulation_output = os.path.join(
    output_folder,
    fname.replace('_ground_truth',
                     '_%d_simulations'%(num_simulations))
    ) 
  
  evid = filename.split('/')[-1].split('_')[0]    
  
  # if simulation file exists, ignore
  if os.path.exists(simulation_output):
    return
  
  simulation_runs = {}
  earthquake = pickle.load(open(filename, 'rb'))
  
  # define the region 
  llat = earthquake['latitude'] - lat_dist_from_epicenter
  ulat = earthquake['latitude'] + lat_dist_from_epicenter
  llng = earthquake['longitude'] - lng_dist_from_epicenter
  ulng = earthquake['longitude'] + lng_dist_from_epicenter
  
  for i in range(num_simulations):
    simulation_results = {}
  
    phones = phoneSampler.sample_phones(phone_user_percentage,
                                        min_users=min_users_to_sample_phone,
                                        plot=False,
                                        llat=llat,
                                        ulat=ulat,
                                        llng=llng,
                                        ulng=ulng
                                        )
    
    # generate triggers
    trig_gen = trigger_generator.trigger_generator(earthquake,
                                                   steady_ratio,
                                                   phones,
                                                   trig_rate=trig_rate,
                                                   myshake_network_latency=myshake_network_latency)
  
    phones_steady, meta_data, df_trig = \
      trig_gen.generate_triggers_envelope(config,
                                          timezone = timezone, 
                                          random_trigger = True, 
                                          time_win = 240
                                          )
    
    # 
    simulation_results = {'config': config,
                          'phones': phones,
                          'phones_steady': phones_steady,
                          'df_trig': df_trig,
                          'meta_data': meta_data
                          }
    
    simulation_runs[i] = simulation_results
  
  pickle.dump(simulation_runs,
              open(simulation_output, 'wb')
              )

Parallel(n_jobs=10)([delayed(generate_simulated_triggers)(item) for item in glob.glob(os.path.join(data_path, '*_ground_truth.pkl'))])
# %%
