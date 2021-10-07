"""
Simple binning methods such as fixed-length and frequency binning.
"""
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo, Any, Callable, NDArray, Optional, Union, Sequence, List
from moredataframes.errors import UserFunctionCallError
from moredataframes.mdfutils import to_numpy, string_param
from moredataframes.encodings import encoding_function
from moredataframes.constants import BIN_RIGHT_EDGE_OFFSET
import numpy as np


def binning_function(func: Callable[..., NDArray[Any]]) -> Callable[..., NDArray[Any]]:
    """
    Wraps a binning function to help abstract out some common functionality.
    Ensures that:
        - the input vals will be a numpy NDArray
        - if inverse=True, then decode_bins is called instead of the binning function
        - if 'bins' is in encoding_info, then digitize is called instead of the binning function
    :param func: the binning function
    :return: a binning function
    """
    def wrapper(vals: ArrayLike, encoding_info: Union[None, EFuncInfo] = None, inverse: Optional[bool] = False,
                **kwargs: Any) -> NDArray[Any]:
        encoding_info = {} if encoding_info is None else encoding_info
        vals = to_numpy(vals)

        # If vals is empty, just return vals
        if vals.size == 0:
            return vals, encoding_info

        # Check that encoding_info contains what it should, if inverse = True, then return the decoded values
        if inverse:
            if 'decode_method' in kwargs:
                return _decode_bins(vals, encoding_info['bins'], decode_method=kwargs['decode_method']), encoding_info
            return _decode_bins(vals, encoding_info['bins']), encoding_info

        # Returns the digitized values after bins have been computed
        def _dig(v):
            if 'edge_behavior' in kwargs:
                return digitize(v, encoding_info, edge_behavior=kwargs['edge_behavior'])
            return digitize(v, encoding_info)

        # Check if 'bins' is in einfo. We dont need encoding_function here since future binning is done based
        #   on past bins, not on new kwargs
        if 'bins' in encoding_info:
            return _dig(vals)

        encoding_info['bins'] = func(vals, encoding_info, **kwargs)
        return _dig(vals)

    return wrapper


def _decode_bins(vals: ArrayLike, bin_edges: Sequence[NDArray[Any]], decode_method: \
        Optional[Union[str, Callable[[NDArray[Any], NDArray[Any]], ArrayLike]]] = 'range') -> NDArray[Any]:
    """
    Decodes binned columns with the given bin_edges. Multiple different binning decoding methods are available since
        decoding binned columns is lossy. (If you wish for exact decodings, use continuous_binning methods).
    :param vals: values to decode
    :param bin_edges: the edges of the bins. Should be a sequence of ndarray's such that a bin index
        of 'i' corresponds to the value having been in the range [bin_edges[i], bin_edges[i + 1]) for i in the range
        [0, len(bin_edges) - 1). This corresponds to numpy's binning (digitize, searchsorted, etc.) being 'left', or
        having right=False.
    :param decode_method: a string or function describing the decoding method to use. Currently available methods:
        - 'range': return a string for each record describing the range. IE: a record having a bin value of '3' would
            be decoded into the string '[x_0, x_1)' where x_0 = bin_edges[3] and x_1 = bin_edges[4].
        - 'left': return the left side of the bin. IE: assume the values were the smallest possible value.
        - 'right': return the right side of the bin. IE: assume the values were the largest possible value.
        - 'mid': return the midpoint of the bin. IE: assume the values were the average of the possible values.

        If a function, then the function should take in two numpy arrays (first being the lower bin edges, second
            being the upper bin edges), and return an ArrayLike the same size as both the input arrays.
    :return: a numpy array
    """
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
                ret.append((bins[col] + bins[col + 1]) / 2)
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


_WARN_VALS = ['warn', 'raise', 'error']
_CLOSEST_VALS = ['closest', 'close']
_EXPAND_VALS = ['expand', 'extend']


@encoding_function  # Not a binning function since it gets called in binning_function and would be infinte recursion
def digitize(vals: ArrayLike, encoding_info: EFuncInfo, edge_behavior: str = 'warn', inverse: bool = False):
    """
    Similar to numpy's digitize() method, bins the values based on specified bins.
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into, or to use if this is a subsequent call
        to this function and you wish to encode/decode in the same way as encoded previously

        Should contain at least the key:
        - 'bins': the edges of the bins. Should be a sequence of sequences of floats or ints such that a bin index
            of 'i' corresponds to the value having been in the range [bin_edges[i], bin_edges[i + 1]) for i in the range
            [0, len(bin_edges) - 1). This corresponds to numpy's binning (digitize, searchsorted, etc.) being 'left', or
            having right=False.
    :param edge_behavior: how to deal with values that fall outside the bin_edges (IE: are smaller than the smallest
        bin_edge, or larger than the largest). This (should) only happen on future calls to binning functions using
        a previous call's encoding_info. Can be:
            - 'warn': raises an error if this occurs
            - 'closest': puts value into the closest bin
            - 'expand': expands the bins to contain the current values at the edges, and returns new bins in the
                encoding_info
    :param inverse: not used. Here because of @encoding_function decorator call
    :return: numpy array
    """

    if 'bins' not in encoding_info or encoding_info['bins'] is None:
        raise TypeError("Must pass bins in encoding_info in order to digitize.")

    if vals.size == 0:
        return vals

    ret, ret_bins = [], []
    for col, b in zip(vals.T, encoding_info['bins']):

        # If edge_behavior is 'closest', then clip the values to be in the right range
        if string_param(edge_behavior, _CLOSEST_VALS):
            col = np.clip(col, b[0], (b[-1] + b[-2]) / 2)

        # Check for values outside the bin edges
        _min, _max = np.min(col), np.max(col)
        if _min < b[0]:
            if string_param(edge_behavior, _WARN_VALS):
                raise ValueError("Value found smaller than the minimum bin edge: " + str(min(col)))
            elif string_param(edge_behavior, _EXPAND_VALS):
                b = np.concatenate(([_min], b))
        if _max >= b[-1]:
            if string_param(edge_behavior, _WARN_VALS):
                raise ValueError("Value found larger than or equal to the maximum bin edge: " + str(max(col)))
            elif string_param(edge_behavior, _EXPAND_VALS):
                b = np.concatenate((b, [_max + BIN_RIGHT_EDGE_OFFSET]))

        # Otherwise, time to digitize
        ret.append(np.digitize(col, b) - 1)
        ret_bins.append(b)

    encoding_info.update({'bins': ret_bins, 'edge_behavior': edge_behavior})
    return np.array(ret).T


@binning_function
def fixed_length(vals: ArrayLike, encoding_info: EFuncInfo, num_bins:int = 10):
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
