import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F


def get_exponent_bin(b: int, i: int) -> str:
    if i == 0:
        return "0" * b
    elif i == 2*b - 1:
        return "1" + "0" * (b-1)
    elif i < b:
        zeros = "0" * (b-i)
        return zeros + "1"
    else:
        zeros = "0" * (i - b)
        return "1" + zeros + "1"

def get_flint_quant_possible_values(b: int) -> list:
    # return list(range(0, 128))
    en = 2*b
    max_i = math.floor(math.log(2**(en-2), 2)) + 1
    possible_quantized_values = []
    for i in range(max_i + 1):
        exp = get_exponent_bin(b, i)  # First-one exponent.
        mb = b - len(exp)  # Mantissa bit.
        m_arr = []
        if i == 0:
            m_arr.append(0)
        else:
            m_arr.append(1)
            for idx in range(1, mb + 1):
                m_arr.extend([x + 2**(-idx) for x in m_arr])
        bias = 1
        possible_quantized_values.extend(
            sorted([int(m * 2**max(i - bias, 0)) for m in m_arr]))
    return possible_quantized_values

def flint_quant_tensor_vec(e: torch.Tensor, steps: list):
    stepified_tensor = torch.zeros_like(e)
    for val in steps:
        mask = e >= val
        stepified_tensor = torch.where(mask, val, stepified_tensor)
    return stepified_tensor

def flint_quant_signed(e: torch.Tensor, b: int) -> torch.Tensor:
    b = 8
    b -= 1
    en = 2*b
    scaling_factor = (2.0**(en - 2)) / (torch.abs(e).max().item())
    # scaling_factor = 127/(torch.abs(e).max().item())
    e = torch.round(e * scaling_factor)
    neg_tensor_vals = torch.where(e < 0, e, 0).abs()
    positive_tensor_vals = torch.where(e >= 0, e, 0)
    possible_quantized_values = get_flint_quant_possible_values(b)
    # print("NUMBER OF QUANTIZATION LEVELS:", len(possible_quantized_values))
    flint_quantized = (
        -flint_quant_tensor_vec(neg_tensor_vals, possible_quantized_values)
        + flint_quant_tensor_vec(positive_tensor_vals, possible_quantized_values)
        ) / scaling_factor
    
    return flint_quantized

def get_flint_lookup_table(bit_width):
    positive_lookup_table = torch.tensor(get_flint_quant_possible_values(bit_width), dtype=torch.float)
    full_lookup_table = torch.cat([positive_lookup_table * -1, positive_lookup_table])
    full_lookup_table = torch.unique(full_lookup_table)
    full_lookup_table, _ = torch.sort(full_lookup_table)
    return full_lookup_table.cuda()

def flint_lookup(input_tensor, reference_tensor):
    """Maps each value in an input tensor to the nearest value in a reference 1D tensor.

    Args:
        input_tensor: The input tensor.
        reference_tensor: The 1D reference tensor.

    Returns:
        A tensor with the same shape as input_tensor, where each value has been mapped 
        to the nearest value in reference_tensor.
    """
    # Reshape input_tensor for broadcasting
    expanded_input = input_tensor.unsqueeze(-1)  
    
    # Calculate absolute differences
    distances = torch.abs(expanded_input - reference_tensor) 
    
    # Find indices of nearest values
    nearest_indices = torch.argmin(distances, dim=-1)  
    
    # Gather nearest values from reference_tensor
    mapped_tensor = reference_tensor[nearest_indices]  
    
    return mapped_tensor

def flint_quantized_weights(w_bits, weights):
    flint_lookup_table = get_flint_lookup_table(w_bits)
    lookup_result = flint_lookup(weights, flint_lookup_table)
    weight_max = weights.abs().max().item()
    processed_weight = (weights/weight_max) * flint_lookup_table.max().item()
    raw_lookup_result = flint_lookup(processed_weight, flint_lookup_table)
    lookup_result = (raw_lookup_result*weight_max) / flint_lookup_table.max().item()
    return lookup_result, flint_lookup_table
