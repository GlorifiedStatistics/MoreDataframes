"""
Simple binning methods such as fixed-length and frequency binning.
"""
from moredataframes.encodings.binning import binning_function
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo
from moredataframes.constants import ENCODING_INFO_BINS_KEY
import numpy as np


def digitize(vals: ArrayLike, encoding_info: EFuncInfo, edge_behavior=''):
    """
    Similar to numpy's digitize() method, bins the values based on specified bins.
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into when inverse=False, or a dictionary that
        contains the encoding information when inverse=True. The encoding_info is a separate, empty dictionary created
        for each call to the encoding function. Modify as you wish within this function call without fears of
        overwriting other data.

        Stores:
            - 'bins': a list of lists of bin indices (one list of bin indices for each column, from left to right)
            - 'edge_behavior':

    :param edge_behavior:
    :return:
    """


@binning_function(num_bins=int)
def fixed_length(vals: ArrayLike, encoding_info: EFuncInfo, num_bins:int = 10):
    """
    Bin values into fixed-length bins.

    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into when inverse=False, or a dictionary that
        contains the encoding information when inverse=True. The encoding_info is a separate, empty dictionary created
        for each call to the encoding function. Modify as you wish within this function call without fears of
        overwriting other data.

        Stores:
            - 'bins': a list of lists of bin indices (one list of bin indices for each column, from left to right)
    
    :param num_bins: the number of bins to use
    :return: a numpy array
    """

    ret = []
    efunc_bins = []
    for col in vals.T:
        _min, _max = np.min(col), np.max(col)
        bins = np.arange(_min, _max, step=(_max-_min) / num_bins)
        ret.append(np.digitize(col, bins, ))
        efunc_bins.append(bins)
    
    encoding_info[ENCODING_INFO_BINS_KEY] = efunc_bins
    
    # Remember to transpose
    return np.array(ret).T
