import os, sys

import gym, ray
from ray.rllib.agents import ppo
from ray.tune import register_env

from src.environments.create_env import create_env
from src.environments.gym_power_voltage_env import GymPowerVoltageEnv
from src.samplers.load_samplers import load_samplers


# path_to_this_notebook = os.path.abspath('.')
# path_to_project = path_to_this_notebook[:path_to_this_notebook.find('src')]
# sys.path.append(path_to_project)
config = {'path_to_data':   './data/',
          't0_hr': 6.,  # When the episode start (default value 6AM)
          'dt_min': 30,  # Timestep size
          'ev_dt_min': 60,  # Timestep size for EV arrivals
          'ev_sampling_dt_min': 60,  # How EV sessions are sampled from the data
          'apply_gaussian_noise': False,  # Make data noisy
          'ev_utility_coef_mean': 1,  # Mean value of the utility coefficient for the EVs
          'ev_utility_coef_scale': 0.13,  # STD of the utility coefficient for the EVs
          'days_per_month_train': 20,  # Days per month for training
          'ev_session_months_train': ['01', '02', '03', '04', '06', '07', '08', '09', '10', '11', ],
          # Months to sample EV sessions for training
          'grid_to_use': 'ieee16',  # What grid topology to use. Now supports only IEEE16.
          'ev_session_months_test': ['05', '12'],  # Months to sample EV sessions for test
          'n_ps_pvs': 4,  # Amount of solar panels that use PecanStreet data
          'n_canopy_pvs': 0,  # Amount of solar panels that use canopy data
          'canopy_pv_rated_power': 250,  # Rated power of these panels
          'n_loads': 0,  # Amount of inflexible loads
          'n_feeders': 1,  # Amount of feeders
          'n_ev_chargers': 4,  # Amount of EV chargers

          'ps_pvs_rated_power': 4,  # Rated power of these panels
          'avg_evs_per_day': 3.5,  # Scaling of the EV arrival rate
          'feeder_p_min': -5,  # Capacity of the feeders
          'g': 4,  # Conductance of each line
          'i_max': 25,  # Capacity of each line
          }


def env_creator(env_config):
    # Preload samplers, it is necessary to avoid re-loading data each time env is created
    (ps_samplers_dict, ps_metadata, canopy_sampler, canopy_metadata,
     price_sampler, price_metadata, ev_sampler, elaadnl_metadata) = load_samplers(env_config)

    return create_env(
        env_config,
        ps_samplers_dict,
        ps_metadata,
        canopy_sampler,
        canopy_metadata,
        price_sampler,
        price_metadata,
        ev_sampler,
        elaadnl_metadata
    )  # return an env instance


# Read this on how to run our own environments
# https://docs.ray.io/en/latest/rllib/rllib-env.html

# ray.init()
register_env("my_env", env_creator)


trainer = ppo.PPOTrainer(env="my_env", config={
    "env_config": config,  # config to pass to env class
    "framework": "torch",
})

while True:
    print(trainer.train())
