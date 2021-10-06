"""
Simple binning methods such as fixed-length and frequency binning.
"""
from moredataframes.constants import ENCODING_INFO_BINS_KEY
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo, Any, Callable, NDArray, Optional, Union, Sequence
from moredataframes.errors import UserFunctionCallError
from moredataframes.mdfutils import to_numpy, string_param
import numpy as np


def binning_function(func: Callable[..., NDArray[Any]]) -> Callable[..., NDArray[Any]]:
    """
    Wraps a binning function to help abstract out some common functionality.
    Ensures that:
        - the input vals will be a numpy NDArray
        - if inverse=True, then decode_bins is called instead of the binning function
    :param func: the binning function
    :return: a binning function
    """
    def wrapper(vals: ArrayLike, einfo: EFuncInfo, inverse: Optional[bool] = False, **kwargs: Any) -> NDArray[Any]:
        vals = to_numpy(vals)

        # Check that encoding_info contains what it should, if inverse = True, then return the decoded values
        if inverse:

            # We want two different function calls here so the default decode_method can be kept if I change it ever
            if 'decode_method' in kwargs:
                return decode_bins(vals, einfo[ENCODING_INFO_BINS_KEY], decode_method=kwargs['decode_method'])
            return decode_bins(vals, einfo[ENCODING_INFO_BINS_KEY])

        return func(vals, einfo, **kwargs)

    return wrapper


def decode_bins(vals: ArrayLike, bin_edges: Sequence[Union[Sequence[Union[float, int]], NDArray[Any]]],
                decode_method: Optional[Union[str, Callable[[NDArray[Any], NDArray[Any]], ArrayLike]]] = 'range') \
        -> NDArray[Any]:
    """
    Decodes binned columns with the given bin_edges. Multiple different binning decoding methods are available since
        decoding binned columns is lossy. (If you wish for exact decodings, use continuous_binning methods).
    :param vals: values to decode
    :param bin_edges: the edges of the bins. Should be a sequence of sequences of floats or ints such that a bin index 
        of 'i' corresponds to the value having been in the range [bin_edges[i], bin_edges[i + 1]) for i in the range
        [0, len(bin_edges) - 1). This corresponds to numpy's binning (digitize, searchsorted, etc.) being 'left', or
        having right=False.
    :param decode_method: a string or function describing the decoding method to use. Currently available methods:
        - 'range': return a string for each record describing the range. IE: a record having a bin value of '3' would
            be decoded into the string '[x_0, x_1)' where x_0 = bin_edges[3] and x_1 = bin_edges[4].
        - 'left': return the left side of the bin. IE: assume the values were the smallest possible value.
        - 'right': return the right side of the bin. IE: assume the values were the largest possible value.
        - 'mid': return the midpoint of the bin. IE: assume the values were the average of the possible values.
            NOTE: If either of the values are np.inf or -np.inf, then the value returned will be the non-infinity
                value only instead of the midpoint.

        If a function, then the function should take in two numpy arrays (first being the lower bin edges, second
            being the upper bin edges), and return an ArrayLike the same size as both the input arrays.
    :return: a numpy array
    """
    vals = to_numpy(vals)

    # Make sure bin_edges is correct format
    def _make_numpy(a, error_message):
        try:
            a = to_numpy(a).reshape(-1)
        except (ValueError, TypeError):
            try:
                a = np.array([b for b in a])
            except TypeError:
                raise TypeError(error_message % type(a))
    
    bin_edges = _make_numpy(bin_edges, "bin_edges is not a sequence of sequences, instead is: %s")
    
    # Make sure each value in bin_edges is an array of values
    for i, b in enumerate(bin_edges):
        bin_edges[i] = _make_numpy(b, "bin list at bin_edges[" + str(i) + "] is not a sequence of floats or ints"
            ", instead is: %s")

    # Sort the bin edges
    bin_edges = np.sort(bin_edges)

    # Check decode_method is either a string or implements __call__
    if not (isinstance(decode_method, str) or callable(decode_method)):
        raise TypeError("decode_method should be either a string or callable, instead is: %s" % type(decode_method))

    ret = []

    # Apply over each column in vals
    for col, bins in zip(vals.T, bin_edges):
        if isinstance(decode_method, str):

            if string_param(decode_method, ['range', 'str', 'string']):
                _str = '[%d, %d)' if np.issubdtype(bins.dtype, np.integer) else '[%f, %f)'
                ret.append(np.array([(_str % (low, high)) for low, high in zip(bins[col], bins[col + 1])]))
            elif string_param(decode_method, ['left', 'small', 'smallest', 'min', 'minimum']):
                ret.append(bins[col])
            elif string_param(decode_method, ['right', 'large', 'largest', 'max', 'maximum']):
                ret.append(bins[col + 1])
            elif string_param(decode_method, ['mid', 'middle', 'midpoint', 'average', 'mean']):
                # Need to make sure values aren't infinity
                a = np.where(np.isinf(bins[col]), bins[col + 1], bins[col])
                b = np.where(np.isinf(bins[col + 1]), bins[col], bins[col + 1])
                ret.append((a + b) / 2)
            else:
                raise ValueError("Unknown binning decode_method: %s" % decode_method)

        else:
            # If we are doing a function call, apply our function
            try:
                ret.append(to_numpy(decode_method(bins[col], bins[col + 1])).reshape(-1))
            except Exception as e:
                raise UserFunctionCallError("Attempted to call decode_method function, but got error: " + str(e))

            # In case the decode_method function does not return an array the same size as col
            if len(ret[-1]) != len(col):
                raise ValueError("Callable decode_method should return an array the same size as each input array."
                                 "Instead returned array of size() = %d" % len(ret[-1]))

    # Convert to numpy array and take transpose since it's a list of columns instead of rows
    return np.array(ret).T


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
    pass


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
