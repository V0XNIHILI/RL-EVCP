import numpy as np

class DummyEnv:

    def __init__(self, n, p_min, p_max, v_min, v_max, u=0):

        # Devices
        self.n_devices = n

        # Action space
        self.p_min = np.full(self.n_devices, p_min)
        self.p_max = np.full(self.n_devices, p_max)
        self.v_min = np.full(self.n_devices, v_min)
        self.v_max = np.full(self.n_devices, v_max)

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
        new_v_min = np.full(self.n_devices, 0)  # np.abs(v_min_min - v_min) #/ v_min_diff
        new_v_max = np.full(self.n_devices, 1)  # np.abs(v_max_min - v_max) #/ v_max_diff

        new_u = np.abs(self.u_min - u) / u_diff

        return new_p_min, new_p_max, new_v_min, new_v_max, new_u
