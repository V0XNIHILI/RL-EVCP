from src.td3_code.memory.episode_trajectory import EpisodeTrajectory
from src.td3_code.memory.replay_buffer import ReplayBuffer
import numpy as np


class TrajectoryMemoryBuffer(ReplayBuffer):

    def __init__(self, scheme, max_size=1000, min_size_to_sample=100,
                 use_transitions=False, sample_during_episode=False):
        super().__init__(scheme, max_size, min_size_to_sample, sample_during_episode)
        self.use_transitions = use_transitions
        self.current_trajectory = None

    def start_episode(self):
        self.episode_started = True
        self.current_trajectory = EpisodeTrajectory(self.scheme)

    def observe_transition(self, transition_dict):
        self.current_trajectory.add_transition(transition_dict)

    def finish_episode(self, ):
        self.episode_started = False
        self.data[self.index] = self.current_trajectory
        if not self.is_full:
            self.size += 1
        self.index += 1
        self.index %= self.max_size

    def sample_batch(self, batch_size,):
        assert self.can_sample, 'Trying to sample from not complete memory buffer'
        shuffled_indexes = np.random.permutation(self.size)
        sample_dict = {name: [None for _ in range(batch_size)] for name in self.scheme}
        longest_episode_length = 0
        for i in range(batch_size):
            episode_ind = np.random.randint(self.size)
            if self.use_transitions:
                ep_sample = self.data[episode_ind].read_single_timestep()
                longest_episode_length = 1
            else:
                ep_sample = self.data[episode_ind].read()
                longest_episode_length = max(longest_episode_length, self.data[episode_ind].length)
            for name, val in ep_sample.items():
                sample_dict[name][i] = val
        mask = np.ones((batch_size, longest_episode_length, 1), dtype='float32')
        for name, val in sample_dict.items():
            for bs in range(len(val)):
                mask[bs, len(val[bs]):] = 0
                zeros_to_pad = longest_episode_length - len(val[bs])
                val[bs] = np.concatenate([val[bs], np.zeros((zeros_to_pad, *np.shape(val[bs])[1:]))])
            sample_dict[name] = np.stack(val)

        return sample_dict
