import time

import numpy as np
import torch

from .rescaler import rescale

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_sample_dict(sample_dict):
    # state = sample_dict['state']
    # state_next = sample_dict['state_next']
    observations = sample_dict['observations']
    observations_next = sample_dict['observations_next']
    actions = sample_dict['actions']
    dones = sample_dict['done']
    reset_mask = sample_dict['reset_mask']
    rewards = sample_dict['reward']
    observations_extended = np.concatenate([observations, observations_next[:, -1:]], axis=1)
    reset_mask_extended = np.concatenate([reset_mask, dones[:, -1:]], axis=1)
    # state_extended = np.concatenate([state, state_next[:, -1:]], axis=1)
    # state_extended = torch.tensor(state_extended, dtype=torch.float32).to(device)
    observations_extended = torch.tensor(observations_extended, dtype=torch.float32).to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
    reset_mask_extended = torch.tensor(reset_mask_extended, dtype=torch.float32).to(DEVICE)

    return observations_extended, actions, rewards, dones, reset_mask_extended


class Runner:

    def __init__(self, env, memory, agent, default_episode_index):
        self.env = env
        self.memory = memory
        self.agent = agent
        self.default_episode_index = default_episode_index

    def run(self, train=True, save_to_memory=True, train_bath_size=128, final=False):
        # Set the seed to make sure that the behavior of the EVs is the same every time when a fixed episode is ran
        if self.default_episode_index:
            np.random.seed(self.default_episode_index)
            
        obs = self.env.reset(train=train, episode_index=self.default_episode_index)
        hidden_state = self.agent.actor.get_initial_state(1)
        done = False
        reset_mask = True
        episode_results = {'reward': 0, 'length': 0, 'env_time': 0, 'sampling_time': 0, 'training_time': 0,
                           'bare_reward': 0,                    # reward without optional constraint violations added
                           'total_feeders_power_price': 0,      # total price of the feeders of the episode
                           'total_pvs_power_price': 0,          # total pv power price during the episode
                           'total_loads_social_welfare': 0,
                           'total_evs_social_welfare': 0,
                           'total_i_constraints_violation': 0,
                           'total_power_flow_constraints_violation': 0,
                           'total_i': 0,
                           'total_max_i': 0,
                           'total_power': 0,
                           'total_target_power': 0,
                           }

        final_results_list = []
        if save_to_memory:
            self.memory.start_episode()
        while not done:
            action, hidden_state = self.agent.select_action(torch.tensor(obs).reshape((1, 1, -1)).to(DEVICE),
                                                            hidden_state, noisy=train, use_target=False)
            action = action.cpu().detach().numpy().reshape(-1)
            t = time.time()
            obs_next, reward, done, result = self.env.step(action)
            episode_results['env_time'] += time.time() - t
            episode_results['reward'] += float(reward)
            episode_results['length'] += 1
            episode_results['bare_reward'] += float(result['reward'])
            episode_results['total_feeders_power_price'] += result['feeders_power_price']
            episode_results['total_pvs_power_price'] += result['pvs_power_price']
            episode_results['total_loads_social_welfare'] += result['loads_social_welfare']
            episode_results['total_evs_social_welfare'] += result['evs_social_welfare']
            episode_results['total_i_constraints_violation'] += result['i_constraints_violation']
            episode_results['total_power_flow_constraints_violation'] += result['power_flow_constraints_violation']
            episode_results['total_i'] += result['total_i']
            episode_results['total_max_i'] += result['total_max_i']
            episode_results['total_power'] += result['total_p']
            episode_results['total_target_power'] += result['total_target_p']

            transition_dict = {'observations': obs.reshape(-1),
                               'observations_next': obs_next.reshape(-1),
                               'actions': action,
                               'done': np.reshape(done, -1),
                               'reward': np.reshape(reward, -1),
                               'reset_mask': np.reshape(reset_mask, -1)}
            if save_to_memory:
                self.memory.observe_transition(transition_dict)
                if done:
                    self.memory.finish_episode()
            if train and self.memory.can_sample:
                t = time.time()
                sample_dict = self.memory.sample_batch(train_bath_size)
                (observations_extended, actions, rewards,
                 dones, reset_mask_extended) = parse_sample_dict(sample_dict)
                episode_results['sampling_time'] += time.time() - t
                t = time.time()
                self.agent.train(observations_extended, actions, rewards, dones, reset_mask_extended)
                # print('Running training')
                episode_results['training_time'] += time.time() - t
            if final:
                final_results_list.append(result)
            obs = obs_next
            reset_mask = bool(done)

        if final:
            return episode_results, final_results_list
        else:
            return episode_results
