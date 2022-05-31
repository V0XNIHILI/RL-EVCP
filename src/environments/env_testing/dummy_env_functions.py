import numpy as np

class DummyEnv:

    def __init__(self):

        # Devices
        self.n_devices = 3

        # Action space
        self.p_min = np.full(self.n_devices, -5)
        self.p_max = np.full(self.n_devices, 10)
        self.v_min = np.full(self.n_devices, 300)
        self.v_max = np.full(self.n_devices, 400)
        self.u = np.full(self.n_devices, 0)

        # Observation space
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


    def rescale_action(self):
        pass

    def normalize_observation(self):
        pass
