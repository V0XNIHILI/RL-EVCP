"""
TODO:
Rescale actions from current [-1, 1] to the min and max action values specified by the obesvation

takes in:
    - Tensor of actions
        - length is num_actions (2 * n_devices)
        - each value is in range [-1, 1]
    - Tuple of tensors (p_min, p_max, v_min, v_max)
outputs:
    - Tensor of actions rescaled between p_min, p_max (first half) and v_min, v_max (second half)
"""
def rescale(p, v, p_min, p_max, v_min, v_max):
    new_p = p.copy()
    new_v = v.copy()
    new_p = (new_p + 1) / 2
    new_v = (new_v + 1) / 2

    p_diff = p_max - p_min
    v_diff = v_max - v_min

    new_p = (new_p * p_diff) + p_min
    new_v = (new_v * v_diff) + v_min

    return new_p, new_v