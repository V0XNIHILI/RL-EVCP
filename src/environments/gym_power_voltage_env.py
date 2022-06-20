from typing import List

import numpy as np
import igraph

import gym
from gym import spaces

from src.optimization.heuristic_greedy import compute_greedy_heuristic
from src.utils.timedata import t_hr_to_t_str, create_timesteps_hr, round_t_hr, split_dates_train_and_test_monthly
from src.environments.visualization import _make_graph
from src.devices.device import Device
from src.optimization.constraint_projection import project_constraints, project_constraints_ev
from collections import defaultdict

import torch as th

class GymPowerVoltageEnv(gym.Env):
    def __init__(self,
                 devices: List[Device],
                 conductance_matrix: np.ndarray,
                 i_max_matrix: np.ndarray,
                 config: dict):

        self._setup_config(config)
        self._setup_devices(devices)
        self.t_ind = None
        self.episode_index = -1
        self.conductance_matrix = conductance_matrix.astype(dtype='float32')
        self.i_max_matrix = i_max_matrix.astype(dtype='float32')

        assert np.shape(conductance_matrix) == (self.n_devices, self.n_devices),\
            'Wrong shape of conductance_matrix %s' % (np.shape(conductance_matrix))
        assert np.shape(i_max_matrix) == (self.n_devices, self.n_devices), \
            'Wrong shape of i_max_matrix %s' % (np.shape(i_max_matrix))

        self.current_episode_statistics = {}
        self.allowed_uncertainties = ['deterministic', 'monthly scenarios', 'monthly average']

        self.observation_space = spaces.Box(
            low = np.concatenate((
                np.full(self.n_devices, 0),  # p_min -5
                np.full(self.n_devices, 0),   # p_max 0
                np.full(self.n_devices, 0), # v_min 300
                np.full(self.n_devices, 0), # v_max 400
                np.full(self.n_devices, 0),   # u     0
            ), dtype=np.float32),
            high = np.concatenate((
                np.full(self.n_devices, 1),   # p_min 0
                np.full(self.n_devices, 1),  # p_max 10
                np.full(self.n_devices, 1), # v_min 300
                np.full(self.n_devices, 1), # v_max 400
                np.full(self.n_devices, 1), # u     1.5
            ), dtype=np.float32),
            dtype=np.float32
        )

        self.p_min_min = np.full(self.n_devices, -5)
        self.p_min_max = np.full(self.n_devices, 0)
        self.p_max_min = np.full(self.n_devices, 0)
        self.p_max_max = np.full(self.n_devices, 10)
        self.v_min_min = np.full(self.n_devices, 300)
        self.v_min_max = np.full(self.n_devices, 300)
        self.v_max_min = np.full(self.n_devices, 400)
        self.v_max_max = np.full(self.n_devices, 400)
        self.u_min = np.full(self.n_devices, -1)
        self.u_max = np.full(self.n_devices, 1.5)

        if self.config["EV_only"]:
            self.action_space = spaces.Box(
                low=np.full(self.n_ev_chargers, -1, dtype=np.float32),
                high=np.full(self.n_ev_chargers, 1, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low = np.concatenate((
                    np.full(self.n_devices, -1),
                    np.full(self.n_devices, -1)
                ), dtype=np.float32),
                high = np.concatenate((
                    np.full(self.n_devices, 1),
                    np.full(self.n_devices, 1)
                ), dtype=np.float32),
                dtype=np.float32
            )

        self.p_min = np.full(self.n_devices, -5)
        self.p_max = np.full(self.n_devices, 10)
        self.v_min = np.full(self.n_devices, 300)
        self.v_max = np.full(self.n_devices, 400)
        self.u = np.full(self.n_devices, 0)

    def _setup_config(self, config):
        self.config = config
        self.use_constraint_projection = config['use_constraint_projection']
        self.normalize_outputs = config['normalize_environment_outputs']
        self.use_rescaled_actions = config['use_rescaled_actions']
        self.t0_hr = config['t0_hr']
        self.t0_str = t_hr_to_t_str(config['t0_hr'])
        self.dt_min = config['dt_min']
        self.ev_dt_min = config['ev_dt_min']
        self.timesteps_hr = create_timesteps_hr(self.t0_hr, self.dt_min)
        self.ev_timesteps_hr = create_timesteps_hr(self.t0_hr, self.ev_dt_min)
        self.timesteps_str = [t_hr_to_t_str(t_hr) for t_hr in self.timesteps_hr]
        self.ev_timesteps_str = [t_hr_to_t_str(t_hr) for t_hr in self.ev_timesteps_hr]

    def _setup_devices(self, devices):
        self.devices = devices
        self.n_devices = len(self.devices)

        self.feeders = [d for d in devices if d.type == 'feeder']
        self.n_feeders = len(self.feeders)
        self.feeder_inds = [i for i, d in enumerate(devices) if d.type == 'feeder']

        self.pvs = [d for d in devices if d.type == 'pv']
        self.n_pvs = len(self.pvs)
        self.pv_inds = [i for i, d in enumerate(devices) if d.type == 'pv']

        self.loads = [d for d in devices if d.type == 'load']
        self.n_loads = len(self.loads)
        self.load_inds = [i for i, d in enumerate(devices) if d.type == 'load']

        self.ev_chargers = [d for d in devices if d.type == 'ev_charger']
        self.n_ev_chargers = len(self.ev_chargers)
        self.ev_charger_inds = [i for i, d in enumerate(devices) if d.type == 'ev_charger']

        self.device_to_dates = {}
        for device in self.devices:
            if device.type != 'ev_charger':
                sampler = device.sampler
                train_dates, test_dates = split_dates_train_and_test_monthly(sampler.dates,
                                                                             self.config['days_per_month_train'])
                self.device_to_dates[device] = {'train': list(train_dates), 'test': list(test_dates)}

    @property
    def done(self):
        if self.t_ind is None or self.t_ind < len(self.timesteps_hr) - 1:
            return False
        else:
            return True

    @property
    def ev_t_ind(self):
        t_hr_round = round_t_hr(self.t_hr, self.ev_dt_min)
        return self.ev_timesteps_hr.index(t_hr_round)

    @property
    def t_hr(self):
        return self.timesteps_hr[self.t_ind] if self.t_ind is not None else None

    @property
    def t_str(self):
        return self.timesteps_str[self.t_ind] if self.t_ind is not None else None

    @property
    def is_ev_ts(self):
        return self.t_hr in self.ev_timesteps_hr

    def reset(self, train=True, episode_index=None):
        self.t_ind = 0

        # if episode_index:
        #     np.random.seed(episode_index)

        if episode_index is None:
            if self.config["random_epoch_order"]:
                self.episode_index = np.random.randint(len(list(self.device_to_dates.values())[0]['train']))
            else:
                self.episode_index += 1
        else:
            self.episode_index = episode_index

        for device in self.devices:
            if device.type == 'ev_charger':
                device.reset()
            else:
                train_or_test = 'train' if train else 'test'
                list_of_dates = self.device_to_dates[device][train_or_test]
                date = list_of_dates[self.episode_index % len(list_of_dates)]
                device.reset(date)

        self.current_episode_statistics = defaultdict(list)

        return self.compute_current_state(normalized=self.normalize_outputs)

    def compute_current_state(self, normalized=False):
        """ Computes nodal power and voltage lower and upper bounds and nodal utility coefficients
            for the current timestep. """
        p_lbs_t, p_ubs_t, v_lbs_t, v_ubs_t, u_t = [], [], [], [], []
        for device in self.devices:
            p_min_d, p_max_d = device.p_min, device.p_max
            v_min_d, v_max_d = device.v_min, device.v_max
            u_d = device.utility_coef
            p_lbs_t.append(p_min_d)
            p_ubs_t.append(p_max_d)
            v_lbs_t.append(v_min_d)
            v_ubs_t.append(v_max_d)
            u_t.append(u_d)

        # update power bounds
        self.p_min, self.p_max = np.array(p_lbs_t), np.array(p_ubs_t)

        # update voltage bounds (should be constant?)
        self.v_min, self.v_max = np.array(v_lbs_t), np.array(v_ubs_t)

        self.u = np.array(u_t)

        # normalize to [0, 1]
        if normalized:
            p_lbs_t, p_ubs_t, v_lbs_t, v_ubs_t, u_t = self.normalize_observation(p_lbs_t, p_ubs_t, v_lbs_t, v_ubs_t, u_t)

        # concatenating all arrays instead of returning a tuple of arrays
        return np.concatenate((p_lbs_t, p_ubs_t, v_lbs_t, v_ubs_t, u_t), axis=0, dtype='float32')

    def compute_full_state(self, uncertainty='deterministic', n_scenarios=10, target_dt_min=None):
        """ Computes nodal power and voltage lower and upper bounds and nodal utility coefficients
            for ALL timesteps in the episode. Uses different methods of estimating future ('uncertainty' parameter).
            This method IS NOT NEEDED for RL. """

        p_lbs, p_ubs, v_lbs, v_ubs, u = [], [], [], [], []
        target_dt_min = target_dt_min if target_dt_min is not None else self.dt_min
        target_timesteps_hr = create_timesteps_hr(self.t0_hr, target_dt_min)
        for t_ind in range(self.t_ind, len(self.timesteps_hr)):
            t_hr = self.timesteps_hr[t_ind]
            if t_hr not in target_timesteps_hr:
                continue
            t_str = self.timesteps_str[t_ind]
            p_lbs_t, p_ubs_t, v_lbs_t, v_ubs_t, u_t = [], [], [], [], []
            for device in self.devices:
                p_min_d, p_max_d = device.get_p_bounds(t_str, uncertainty=uncertainty, target_dt_min=target_dt_min)
                v_min_d, v_max_d = device.get_v_bounds(t_str, uncertainty=uncertainty, target_dt_min=target_dt_min)
                u_d = device.get_utility_coef(t_str, uncertainty=uncertainty, target_dt_min=target_dt_min)
                if uncertainty == 'monthly scenarios':

                    if len(u_d) <= n_scenarios:
                        factor = n_scenarios // len(u_d) + 1
                        u_d = np.concatenate([u_d] * factor)[:n_scenarios]
                    else:
                        u_d = u_d[:n_scenarios]

                    if len(p_min_d) < n_scenarios:
                        factor = n_scenarios // len(p_min_d) + 1
                        p_min_d = np.concatenate([p_min_d] * factor)[:n_scenarios]
                    else:
                        p_min_d = p_min_d[:n_scenarios]

                    if len(p_max_d) < n_scenarios:
                        factor = n_scenarios // len(p_max_d) + 1
                        p_max_d = np.concatenate([p_max_d] * factor)[:n_scenarios]
                    else:
                        p_max_d = p_max_d[:n_scenarios]

                    if len(v_min_d) < n_scenarios:
                        factor = n_scenarios // len(v_min_d) + 1
                        v_min_d = np.concatenate([v_min_d] * factor)[:n_scenarios]
                    else:
                        v_min_d = v_min_d[:n_scenarios]

                    if len(v_max_d) < n_scenarios:
                        factor = n_scenarios // len(v_max_d) + 1
                        v_max_d = np.concatenate([v_max_d] * factor)[:n_scenarios]
                    else:
                        v_max_d = v_max_d[:n_scenarios]

                p_lbs_t.append(p_min_d)
                p_ubs_t.append(p_max_d)
                v_lbs_t.append(v_min_d)
                v_ubs_t.append(v_max_d)
                u_t.append(u_d)

            p_lbs.append(p_lbs_t)
            p_ubs.append(p_ubs_t)
            v_lbs.append(v_lbs_t)
            v_ubs.append(v_ubs_t)
            u.append(u_t)

        p_lbs = np.transpose(p_lbs, (2, 0, 1))
        p_ubs = np.transpose(p_ubs, (2, 0, 1))
        v_lbs = np.transpose(v_lbs, (2, 0, 1))
        v_ubs = np.transpose(v_ubs, (2, 0, 1))
        u = np.transpose(u, (2, 0, 1))

        evs_dict = {}
        abs_ev_ind = 0

        for d_ind in self.ev_charger_inds:
            evc = self.devices[d_ind]

            for ev_ind, ev in evc.info['current_episode_evs_dict'].items():
                t_arr_target_ind = target_timesteps_hr.index(ev.t_arr_hr)
                t_dep_target_ind = target_timesteps_hr.index(ev.t_dep_hr)
                evs_dict[abs_ev_ind] = (d_ind, t_arr_target_ind, t_dep_target_ind, ev.soc_goal, ev.utility_coef)
                abs_ev_ind += 1

        return p_lbs, p_ubs, v_lbs, v_ubs, u, evs_dict

    def compute_constraint_violation(self, p, v):
        i_constraints_violation = 0
        total_i = 0
        total_max_i = 0
        total_p = 0
        total_target_p = 0
        max_i = 0

        for d_from_ind in range(self.n_devices):
            for d_to_ind in range(d_from_ind, self.n_devices):
                g = self.conductance_matrix[d_from_ind, d_to_ind]
                i = (v[d_to_ind] - v[d_from_ind]) * g
                i_max = self.i_max_matrix[d_from_ind, d_to_ind]
                i_constraints_violation += max(0, abs(i) - abs(i_max))
                total_i += abs(i)
                total_max_i += abs(i_max)

                if max_i < max(0, abs(i) - abs(i_max)):
                    max_i = max(0, abs(i) - abs(i_max))


        power_flow_constraints_violation = 0

        for i in range(self.n_devices):
            p_i_target = -v[i] * sum([self.conductance_matrix[i, j] * (v[i] - v[j])
                                      for j in range(self.n_devices) if i != j]) / 1000
            power_flow_constraints_violation += max(0, abs(p_i_target - p[i]))
            total_target_p += abs(p_i_target)
            total_p += abs(p[i])

        return i_constraints_violation, power_flow_constraints_violation, total_i, total_max_i, max_i, total_p, total_target_p

    def step(self, action, full_state=False):
        """ Received actions are in [-1, 1] """

        if self.config["EV_only"] and not full_state:
            self.compute_current_state()

            # if self.use_rescaled_actions:
            #     return to real power and voltage based on current bounds
            p_ev = self.rescale_action_p(action)
            # else:
            #     p_ev = action

            if self.config["predicting_bounds"]:
                # Set the new top power limit for evs to the model predictions
                new_p_max = self.p_max.copy()
                for new_max_power_ev, d_ind in zip(p_ev, self.ev_charger_inds):
                    new_p_max[d_ind] = new_max_power_ev

                # hack to catch failing runs
                no_result = True
                second_try = False
                while no_result:
                    try:
                        p_m = self.p_max if second_try else new_p_max
                        p, v, model = compute_greedy_heuristic(self.u, self.p_min, p_m, self.v_min, self.v_max,
                                                               self.conductance_matrix, self.i_max_matrix,
                                                               lossless=self.config["lossless_solver"], tee=False,
                                                               iterations=self.config["solver_iterations"])
                        if(np.isnan(p[0])):
                            raise ValueError("Got nan")

                        no_result = False
                    except Exception as e:
                        print(e)
                        print("actions before scaling:")
                        print(action)
                        print(f'Rescale actions is {self.use_rescaled_actions}')
                        print("p_max before ev bounds:")
                        print(self.p_max)
                        print("actions after scaling:")
                        print(p_ev)
                        print("ev pmax:")
                        print(self.p_max[self.ev_charger_inds])
                        print("ev pmin:")
                        print(self.p_min[self.ev_charger_inds])
                        print("p_max after ev bounds:")
                        print(self.p_max)

                        if second_try:
                            raise ValueError("Failed twice")

                    second_try = True

            else:
                p, v, model = project_constraints_ev(p_ev, self.ev_charger_inds, self.u, self.p_min,
                                                  self.p_max, self.v_min, self.v_max, self.conductance_matrix,
                                                  self.i_max_matrix, lossless=self.config["lossless_solver"],
                                                  iterations=self.config["solver_iterations"])
        else:
            # give 1 array with p,v instead of 2
            p_in = action[:self.n_devices]
            v_in = action[self.n_devices:]

            if self.use_rescaled_actions and not full_state:
                # return to real power and voltage based on current bounds
                p, v = self.rescale_action(p_in, v_in)
            else:
                p = p_in
                v = v_in

            if self.use_constraint_projection and not full_state:
                self.compute_current_state()
                # TODO: why do we still use the previous steps() self.u here?
                p, v, model = project_constraints(p, v, self.n_devices, self.u, self.p_min,
                                           self.p_max, self.v_min, self.v_max, self.conductance_matrix, self.i_max_matrix, iterations=self.config["solver_iterations"])

        reward = 0
        feeders_power_price = 0
        pvs_power_price = 0
        loads_social_welfare = 0
        evs_social_welfare = 0
        total_requested_min_p = 0
        total_requested_max_p = 0

        for d_ind, d in enumerate(self.devices):
            r = d.update_power_and_voltage(p[d_ind], v[d_ind])

            total_requested_min_p += abs(d.p_min)
            total_requested_max_p += abs(d.p_max)

            if d.type == 'feeder':
                self.current_episode_statistics['feeders_price'].append(r)
                feeders_power_price += r
            elif d.type == 'pv':
                self.current_episode_statistics['pvs_price'].append(r)
                pvs_power_price += r
            elif d.type == 'load':
                loads_social_welfare += r
            elif d.type == 'ev_charger':
                evs_social_welfare += r
                self.current_episode_statistics['evs_welfare'].append(r)

            reward += r

        self.current_episode_statistics['social_welfare'].append(reward)
        self.t_ind += 1

        for d in self.devices:
            d.update_timestep(self.t_str)

        i_constraints_violation, power_flow_constraints_violation, total_i, total_max_i, max_i, total_p, total_target_p = \
            self.compute_constraint_violation(p, v)

        result = {'reward': reward,
                  'feeders_power_price': feeders_power_price,
                  'pvs_power_price': pvs_power_price,
                  'loads_social_welfare': loads_social_welfare,
                  'evs_social_welfare': evs_social_welfare,
                  'i_constraints_violation': i_constraints_violation,
                  'power_flow_constraints_violation': power_flow_constraints_violation,
                  'total_i': total_i,
                  'total_max_i': total_max_i,
                  'max_i': max_i,
                  'total_p': total_p,
                  'total_target_p': total_target_p,
                  'total_requested_min_p': total_requested_min_p,
                  'total_requested_max_p': total_requested_max_p,
                  'p': p,
                  'v': v }

        training_reward = 0
        if self.config["violations_in_reward"]:
            # multiply by 1e-2 to get it in the range of the reward
            training_reward -= self.config["current_reward_factor"] * i_constraints_violation
            training_reward -= self.config["power_reward_factor"] * power_flow_constraints_violation
            training_reward -= self.config["max_current_reward_factor"] * max_i

            if not self.config["one_reward_target"] or training_reward >= -1e-2:
                training_reward += self.config["utility_reward_factor"] * reward
        else:
            training_reward += self.config["utility_reward_factor"] * reward

        # if training_reward > 0:
        #     print(f'training_reward: {training_reward}')
        #     print(f'current component: {-self.config["current_reward_factor"] * i_constraints_violation}')
        #     print(f'power component: {-self.config["power_reward_factor"] * power_flow_constraints_violation}')
        #     print(f'utility component: {self.config["utility_reward_factor"] * reward}')
        #     print('voltages')
        #     print(v)
        #     print('power')
        #     print(p)

        # return in gym format, result is now the info part of result
        return self.compute_current_state(normalized=self.normalize_outputs), training_reward, self.done, result

    def rescale_action(self, p, v):
        """ [-1, 1] to real lower and upper bound """
        new_p = p.copy()
        new_v = v.copy()
        new_p = (new_p + 1) / 2
        new_v = (new_v + 1) / 2

        p_diff = self.p_max - self.p_min
        v_diff = self.v_max - self.v_min

        new_p = (new_p * p_diff) + self.p_min
        new_v = (new_v * v_diff) + self.v_min

        return new_p, new_v

    def rescale_action_p(self, p):
        """ [-1, 1] to real lower and upper bound for ev charger power"""
        new_p = p.copy()
        new_p = (new_p + 1) / 2

        p_diff = self.p_max[self.ev_charger_inds] - self.p_min[self.ev_charger_inds]
        new_p = (new_p * p_diff) + self.p_min[self.ev_charger_inds]

        return new_p

    def normalize_observation(self, p_min, p_max, v_min, v_max, u):
        """ Real values to [0, 1] """
        p_min_diff = self.p_min_max - self.p_min_min
        p_max_diff = self.p_max_max - self.p_max_min

        # This is zero
        # v_min_diff = self.v_min_max - self.v_min_min
        # v_max_diff = self.v_max_max - self.v_max_min

        u_diff = self.u_max - self.u_min

        new_p_min = np.abs(self.p_min_min - p_min) / p_min_diff
        new_p_max = np.abs(self.p_max_min - p_max) / p_max_diff

        # v_min, v_max are always 300, 400 so can always go to 0, 1
        new_v_min = np.full(self.n_devices, 0) # np.abs(v_min_min - v_min) #/ v_min_diff
        new_v_max = np.full(self.n_devices, 1) # np.abs(v_max_min - v_max) #/ v_max_diff

        new_u = np.abs(self.u_min - u) / u_diff

        return new_p_min, new_p_max, new_v_min, new_v_max, new_u

    def compute_result(self, do_print=False):
        assert self.done, 'Compute result should only be called when env is done!'

        evs_soc_achieved = []   
        evs_soc_maximum = []
        evs_utility_coefs = []

        for evc in self.node.ev_chargers:
            for ev_ind, ev in evc.info['current_episode_evs_dict'].items():
                evs_soc_achieved.append(ev.current_soc)
                evs_soc_maximum.append(ev.soc_max)
                evs_utility_coefs.append(ev.utility_coef)

        evs_soc_achieved = np.array(evs_soc_achieved)
        evs_soc_maximum = np.array(evs_soc_maximum)
        evs_soc_ratio = evs_soc_achieved / evs_soc_maximum
        evs_welfare = np.dot(evs_utility_coefs, evs_soc_achieved)

        feeders_power = 0
        pvs_power = 0
        feeders_price = sum(self.current_episode_statistics['feeders_price'])
        pvs_price = sum(self.current_episode_statistics['pvs_price'])

        for feeder in self.node.feeders:
            feeders_power += sum(feeder.info['current_episode_power']) * self.dt_min / 60
        for pv in self.node.pvs:
            pvs_power += sum(pv.info['current_episode_power']) * self.dt_min / 60
        if do_print:
            print('Episode %d is finished. Results are:' % self.episode_index)
            print('EVs average SOC = %.2f, total fulfilled demand = %.2fkW' % (evs_soc_ratio.mean(),
                                                                                evs_soc_achieved.sum()))
            print('Power generation: %.2fkW from PVs and %.2fkW from feeders' % (pvs_power, feeders_power))
            print('Feeders power cost = %.2f, PV power cost = %.2f, EVs welfare = % .2f' %
                  (feeders_price, pvs_price, evs_welfare.sum()))
            print('Total social welfare = %.2f' % (sum(self.current_episode_statistics['social_welfare'])))

        results_dict = {'evs_soc': evs_soc_ratio.mean(), 'evs_welfare': evs_welfare.sum(),
                        'power_price': feeders_price + pvs_price,
                        'social_welfare': sum(self.current_episode_statistics['social_welfare'])}

        return results_dict

    def plot_grid(self, bbox=(0, 0, 500, 500), margin=30, save=False, path_to_figures=None, title=None):
        graph = _make_graph(self)
        if save:
            assert path_to_figures is not None, 'Specify path_to_figures to save the plot!'
            assert title is not None, 'Specify title to save the plot!'
            return igraph.plot(graph, path_to_figures + title + '.png', bbox=bbox, margin=margin)
        else:
            return igraph.plot(graph, bbox=bbox, margin=margin)
