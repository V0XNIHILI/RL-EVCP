import pickle as pickle
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np

from src.utils.timedata import split_dates_train_and_test_monthly
from src.samplers.time_series_sampler import TimeSeriesSampler
from src.samplers.ev_session_sampler import EVSessionSampler


def load_samplers(config):
    config_hash = hash(repr(sorted(config.items())))

    # Read config
    t0_hr = config['t0_hr']
    dt_min = config['dt_min']
    ev_dt_min = config['ev_dt_min']
    ev_sampling_dt_min = config['ev_sampling_dt_min']
    path_to_data = config['path_to_data']
    days_per_month_train = config['days_per_month_train']
    ev_session_months_train = config['ev_session_months_train']
    ev_session_months_test = config['ev_session_months_test']
    ev_utility_coef_mean = config['ev_utility_coef_mean']
    ev_utility_coef_scale = config['ev_utility_coef_scale']
    apply_gaussian_noise = config['apply_gaussian_noise']

    # Try to load cache files
    if exists(path_to_data + '/cache_config_hash.pickle'):
        with open(path_to_data + '/cache_config_hash.pickle', 'rb') as f:
            cache_hash = pickle.load(f)

        # Assume there is cache if there is a cache config hash and they match
        if cache_hash == config_hash:
            print("loading strait from cache")
            with open(path_to_data + '/cache.pickle', 'rb') as cache_file:
                samplers = pickle.load(cache_file)
                return samplers

    print("Loading ps meta data")

    # Pecan Street data
    with open(path_to_data + '/pecanstreet/metadata_dict.pickle', 'rb') as f:
        ps_metadata = pickle.load(f, )

    ps_all_dates = None
    for hid in ps_metadata:
        hid_dates = set(ps_metadata[hid]['dates'])
        ps_all_dates = hid_dates if ps_all_dates is None else hid_dates & ps_all_dates

    ps_dates_train, ps_dates_test = split_dates_train_and_test_monthly(ps_all_dates, days_per_month_train)
    ps_device_type_to_column = {'pv': 'solar', 'load': 'usage'}

    print("Loading ps solar data")

    # Canopy data
    path_to_canopy_data = path_to_data + '/pvdata.nist.gov/'
    with open(path_to_data + '/pvdata.nist.gov/metadata_dict.pickle', 'rb') as f:
        canopy_metadata = pickle.load(f, )

    canopy_all_dates = canopy_metadata['dates']
    canopy_dates_train, canopy_dates_test = split_dates_train_and_test_monthly(canopy_all_dates, days_per_month_train)
    canopy_device_type_to_column = {'pv': 'InvPDC_kW_Avg', }
    canopy_solar_rated_power = canopy_metadata['solar_rated_power']

    print("Loading arrivals data")

    # ElaadNL data
    path_to_elaadnl_data = path_to_data + '/elaadnl/'
    with open(path_to_data + '/elaadnl/metadata_dict.pickle', 'rb') as f:

        elaadnl_metadata = pickle.load(f, )

        # Plot the data
        plt.plot(np.linspace(0, 24, num=len(elaadnl_metadata['arrival_counts'].keys())), elaadnl_metadata['arrival_counts'].values())
        plt.xlim(0, 24)
        plt.xlabel('Time of day')
        plt.ylabel('Car arrivals per minute')
        plt.show()

    ev_arrivals_per_minute = elaadnl_metadata['arrival_counts']

    print("Loading price data")

    # New York price data
    path_to_price_data = path_to_data + '//newyork_price/'
    with open(path_to_price_data + '/metadata_dict.pickle', 'rb') as f:
        price_metadata = pickle.load(f, )

    print("Loading done")

    price_all_dates = price_metadata['dates']
    price_dates_train, price_dates_test = split_dates_train_and_test_monthly(price_all_dates, days_per_month_train)
    price_solar_rated_power = None
    price_device_type_to_column = {'feeder': 'price', }

    print("Creating samplers")

    # Creating samplers

    # For each house in the street, load date, time, grid power used , solar power generated, car power usage, and total power usage
    pv_samplers_dict = {hid: TimeSeriesSampler(t0_hr, dt_min, path_to_data + 'pecanstreet/houses/' + hid + '.csv',
                                               ps_metadata[hid]['solar_rated_power'],
                                               ps_device_type_to_column, apply_gaussian_noise=apply_gaussian_noise)
                        for hid in ps_metadata}

    # DC and AC generated powers per minute for multiple days
    canopy_sampler = TimeSeriesSampler(t0_hr, dt_min, path_to_data + '/pvdata.nist.gov/processed_data.csv',
                                       canopy_solar_rated_power, canopy_device_type_to_column,
                                       apply_gaussian_noise=apply_gaussian_noise)

    # Energy prices per minute from 1st of Jan 2019 to 31st of March 2019
    price_sampler = TimeSeriesSampler(t0_hr, dt_min, path_to_data + '/newyork_price/price.csv',
                                      price_solar_rated_power, price_device_type_to_column,
                                      apply_gaussian_noise=apply_gaussian_noise)

    # Over multiple days, list the arrival, departure and duration time of each charge and how much was charged
    ev_sampler = EVSessionSampler(t0_hr, dt_min, ev_dt_min, ev_sampling_dt_min,
                                  path_to_data + '/elaadnl/charging_sessions.csv',
                                  ev_utility_coef_mean, ev_utility_coef_scale,
                                  apply_gaussian_noise=apply_gaussian_noise)

    print("Saving cache")

    # Save cache for later
    with open(path_to_data + '/cache.pickle', 'wb') as handle:
        pickle.dump((pv_samplers_dict, ps_metadata, canopy_sampler, canopy_metadata,
            price_sampler, price_metadata, ev_sampler, elaadnl_metadata), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_to_data + '/cache_config_hash.pickle', 'wb') as handle:
        pickle.dump(config_hash, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return (pv_samplers_dict, ps_metadata, canopy_sampler, canopy_metadata,
            price_sampler, price_metadata, ev_sampler, elaadnl_metadata)
