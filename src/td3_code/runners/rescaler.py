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
def rescale(action, p_min, p_max, v_min, v_max):
    return action