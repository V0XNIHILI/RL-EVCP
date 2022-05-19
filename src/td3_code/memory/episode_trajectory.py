import numpy as np


class EpisodeTrajectory:

    def __init__(self, scheme):
        self.scheme = scheme
        self.length = 0
        self.finished = False
        self.data = {name: [] for name in scheme.keys()}

    def add_transition(self, transition_dict):
        assert not self.finished, 'Trying to write to a finished episode trajectory'
        for name, value in transition_dict.items():
            self.data[name].append(value)
        self.length += 1
        self.finished = transition_dict['done'][0]

    def read(self, max_length=None):
        assert self.finished, 'Trying to read unfinished episode trajectory'
        length_to_sample = self.length if max_length is None else min(self.length, max_length)
        indices = np.arange(0, length_to_sample, dtype='int')
        sample_dict = {name: [self.data[name][ind] for ind in indices] for name in self.data}
        return sample_dict

    def read_single_timestep(self, t_ind=None):
        assert self.finished, 'Trying to read unfinished episode trajectory'
        index = np.random.randint(0, self.length, ) if t_ind is None else t_ind
        sample_dict = {name: [self.data[name][index]] for name in self.data}
        return sample_dict
