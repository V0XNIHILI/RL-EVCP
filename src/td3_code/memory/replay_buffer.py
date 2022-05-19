import numpy as np


class ReplayBuffer:

    def __init__(self, scheme, max_size=1000, min_size_to_sample=100, sample_during_episode=True):
        self.scheme = scheme
        self.max_size = max_size
        self.min_size_to_sample = min_size_to_sample
        self.data = [None for _ in range(self.max_size)]

        self.sample_during_episode = sample_during_episode
        self.episode_started = False
        self.index = 0
        self.size = 0

    @property
    def can_sample(self):
        return self.size >= self.min_size_to_sample and (self.sample_during_episode or not self.episode_started)

    @property
    def is_full(self):
        return self.size == self.max_size

    def start_episode(self):
        raise NotImplementedError

    def observe_transition(self, transition_dict):
        raise NotImplementedError

    def finish_episode(self,):
        raise NotImplementedError

    def sample_batch(self, batch_size,):
        assert self.can_sample, 'Trying to sample from not complete memory buffer'
        raise NotImplementedError
