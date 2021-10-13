"""
Simple binning methods such as fixed-length and frequency binning.
"""
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo, Any, NDArray
from .utils import binning_function
import numpy as np


@binning_function
def fixed_length(vals: ArrayLike, encoding_info: EFuncInfo, num_bins:int = 10) -> NDArray[Any]:
    """
    Bin values into fixed-length bins.

    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into, or to use if this is a subsequent call
        to this function and you wish to encode/decode in the same way as encoded previously
    :param num_bins: the number of bins to use
    :return: a numpy array
    """
    # If num_bins is 1, then just return bins=vals[0], vals[0] + BIN_RIGHT_EDGE_OFFSET
    if num_bins == 1:
        return [[v[0], v[0] + BIN_RIGHT_EDGE_OFFSET] for v in vals]

    # If num_bins <= 0, raise an error
    if num_bins <= 0:
        raise ValueError("num_bins must be integer > 0, instead is: " + str(num_bins))

    # Otherwise if num_bins != 1, but each column only has length 1, raise an error
    if vals.shape[0] == 1:
        raise ValueError("Cannot bin columns of length 1 into multiple bins.")

    ret = []
    for col in vals.T:
        _min, _max = np.min(col), np.max(col)
        ret.append(np.empty([num_bins + 1]))
        ret[-1][:-1] = np.arange(_min, _max, step=(_max - _min) / num_bins)
        ret[-1][-1] = _max + BIN_RIGHT_EDGE_OFFSET

    return ret
