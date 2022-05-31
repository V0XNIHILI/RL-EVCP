from src.td3_code.memory.replay_buffer import ReplayBuffer
import numpy as np


class TransitionMemoryBuffer(ReplayBuffer):

    def __init__(self, scheme, max_size=1000, min_size_to_sample=100, sample_during_episode=False):
        super().__init__(scheme, max_size, min_size_to_sample, sample_during_episode)

    def start_episode(self):
        self.episode_started = True

    def observe_transition(self, transition_dict):
        self.data[self.index] = transition_dict
        if not self.is_full:
            self.size += 1
        self.index += 1
        self.index %= self.max_size

    def finish_episode(self):
        self.episode_started = False

    def sample_batch(self, batch_size, ):
        assert self.can_sample, 'Trying to sample from not complete memory buffer'
        shuffled_indexes = np.random.permutation(self.size)
        sample_dict = {name: [None for _ in range(batch_size)] for name in self.scheme}
        for i in range(batch_size):
            transition_dict = self.data[shuffled_indexes[i % len(shuffled_indexes)]]
            for name, val in transition_dict.items():
                sample_dict[name][i] = [val]
        for name, val in sample_dict.items():
            sample_dict[name] = np.stack(val)
        return sample_dict
