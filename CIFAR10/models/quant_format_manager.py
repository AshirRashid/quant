from .biscaled import biscaled_quantized_weights
from .flint import flint_quantized_weights


def get_current_q_function(m):
    # if not hasattr(m, "q_method"): raise ValueError("No q_method specified for layer", m)
    if not hasattr(m, "q_method"): return lambda bit_width, params: (params, None)

    if m.q_method == "biscaled":
        return biscaled_quantized_weights
    elif m.q_method == "flint":
        return flint_quantized_weights
    else:
        raise ValueError("Unknown quantization method", m.q_method)
