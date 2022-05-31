import unittest
import numpy as np
import numpy.testing
from dummy_env_functions import DummyEnv

class TestDummyEnv(unittest.TestCase):

    def test_rescale_action(self):
        n = 3
        dummyEnv = DummyEnv(n, -5, 10, 300, 400)

        p, v = np.full(n, -0.6), np.full(n, 0.5)

        p_scaled, v_scaled = dummyEnv.rescale_action(p, v)

        p_correct, v_correct = np.full(n, -2), np.full(n, 375)


        numpy.testing.assert_almost_equal(p_scaled, p_correct)
        numpy.testing.assert_almost_equal(v_scaled, v_correct)

    def test_normalize_observation(self):
        pass

if __name__ == '__main__':
    unittest.main()
