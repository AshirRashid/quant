import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F

BINS_NUM = 15
DROP_FACTOR = 10.0
BISCALED_PC_SCALE_WIDE = 10
IS_USING_DROP_FACTOR = True

def biscaled_get_value_dist(t: torch.Tensor, nbins=15):
    nbins = BINS_NUM
    t = t.abs()
    tensor_np = t.cpu().numpy()
    # print("NBINS:", nbins)
    return np.histogram(tensor_np, bins=nbins) # = counts, bin_edges


def biscaled_apply_ratio_heuristic_on_histogram(stats: tuple, drop_factor: float = 10.):
    counts, bin_edges = stats
    drop_factor = DROP_FACTOR

    for i in range(1, len(counts)):
        if counts[i-1] > 0 and counts[i] > 0:
            # print(counts[i-1] / counts[i])
            # print("drop factor:", drop_factor)
            if counts[i-1] / counts[i] >= drop_factor:
                break
    else:
        print("No such drop found.")
    
    return math.ceil(math.log(bin_edges[i], 2)), bin_edges[i]


def biscaled_apply_top_pc_heuristic(stats: tuple):
    counts, bins = stats
    # Compute the cumulative sum of the counts to get the cumulative distribution
    cumulative_counts = np.cumsum(counts)
    
    # Find the total number of elements represented by the histogram
    total_elements = cumulative_counts[-1]
    
    sw_pc = BISCALED_PC_SCALE_WIDE
    threshold_index = np.searchsorted(cumulative_counts, (1 - sw_pc/100.) * total_elements)
    # breakpoint()
    
    # Handle the case where the percentile falls within the last bin
    if threshold_index >= len(bins) - 1:
        threshold_index = len(bins) - 2
    
    # The corresponding threshold value from bins
    threshold = bins[threshold_index]
    
    if threshold == 0.:
        threshold = 0.00001
    # breakpoint()

    return abs(math.ceil(math.log(threshold, 2))), threshold # bits req for the threshold value, threshold value


def biscaled_quant(t: torch.Tensor, target_bit_width: int, stats: tuple): # with training dataset
    # print(f"max: {t.abs().max().item()}")
    t_max = t.abs().max().item()
    ib_sw = abs(math.ceil(math.log(t_max, 2)))
    if IS_USING_DROP_FACTOR:
        ib_sf, bin_edge = biscaled_apply_ratio_heuristic_on_histogram(stats)
    else:
        ib_sf, bin_edge = biscaled_apply_top_pc_heuristic(stats)

    fb_sw = target_bit_width - 1 - ib_sw # -1 for the sign bit
    fb_sf = target_bit_width - 1 - ib_sf # -1 for the sign bit
    
    sf_tensor = torch.where(t.abs() - bin_edge < 0, t, 0)
    sw_tensor = torch.where(t.abs() - bin_edge >= 0, t, 0)
    
    sw_scale_factor = (2 ** (target_bit_width - 1) - 1) / t_max
    sf_scale_factor = (2 ** (target_bit_width - 1) - 1) / abs(bin_edge)

    qsf_tensor = sf_tensor.mul_(sf_scale_factor).round_().div_(sf_scale_factor)
    qsw_tensor = sw_tensor.mul_(sw_scale_factor).round_().div_(sw_scale_factor)

    # return t.mul_(sw_scale_factor).round_().div_(sw_scale_factor)
    # print((qsf_tensor + qsw_tensor).flatten().sum())
    return qsf_tensor + qsw_tensor


def biscaled_calibrate(qlayer, t: torch.Tensor, is_activation: bool = False):
    if is_activation:
        qlayer.biscaled_stats_weight = biscaled_get_value_dist(t.abs())
    else:
        qlayer.biscaled_stats_activation = biscaled_get_value_dist(t.abs())


def get_biscaled_lookup_table(total_bits, param_tensor):
    param_tensor = param_tensor.detach()

    bit_width_sf, threshold_sf = biscaled_apply_top_pc_heuristic(biscaled_get_value_dist(param_tensor.abs()))
    # print(threshold_sf)
    bit_width_sw = max(0, total_bits - bit_width_sf - 1)
    num_q_levels_sf = 2 ** bit_width_sf
    num_q_levels_sw = 2 ** bit_width_sw

    # print("SF Q Levels", num_q_levels_sf)
    # print("SW Q Levels", num_q_levels_sw)

    t_max = param_tensor.abs().max().item()
    positive_lookup_table = torch.zeros(num_q_levels_sf + num_q_levels_sw - 1)
    positive_lookup_table[:num_q_levels_sf - 1] = torch.linspace(0, threshold_sf, num_q_levels_sf)[:-1]
    positive_lookup_table[num_q_levels_sf - 1:] = torch.linspace(threshold_sf, t_max, num_q_levels_sw)

    full_lookup_table = torch.cat([positive_lookup_table * -1, positive_lookup_table])
    full_lookup_table = torch.unique(full_lookup_table)
    full_lookup_table, _ = torch.sort(full_lookup_table)
    return full_lookup_table.cuda()
  
def biscaled_lookup(input_tensor, reference_tensor):
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

def biscaled_quantized_weights(w_bits, weight_tensor):
    biscaled_lookup_table = get_biscaled_lookup_table(w_bits, weight_tensor)
    return biscaled_lookup(weight_tensor, biscaled_lookup_table), biscaled_lookup_table