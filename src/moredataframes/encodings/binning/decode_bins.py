"""
Decodes any binned data (not continuous-binned).
"""
from moredataframes.mdfutils.typing import ArrayLike, Union, NDArray, Any, Optional, Callable, Sequence
from moredataframes.mdfutils import to_numpy, string_param
from moredataframes.errors import UserFunctionCallError
import numpy as np


def decode_bins(vals: ArrayLike, bin_edges: Sequence[Union[float, int]],
                decode_method: Optional[Union[str, Callable[[NDArray[Any], NDArray[Any]], ArrayLike]]] = 'range') \
        -> NDArray[Any]:
    """
    Decodes binned columns with the given bin_edges. Multiple different binning decoding methods are available since
        decoding binned columns is lossy. (If you wish for exact decodings, use continuous_binning methods).
    :param vals: values to decode
    :param bin_edges: the edges of the bins. Should be a sequence of floats or ints such that a bin index of 'i'
        corresponds to the value having been in the range [bin_edges[i], bin_edges[i + 1]) for i in the range
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
    vals = to_numpy(vals)

    # Make sure bin_edges is correct format
    try:
        bin_edges = to_numpy(bin_edges).reshape(-1)
    except (ValueError, TypeError):
        try:
            _bins = [b for b in bin_edges]
            if not all([(isinstance(b, (float, int)) or
                         issubclass(type(b), (np.integer, np.floating))) for b in _bins]):
                raise TypeError
            bin_edges = np.array(_bins)
        except TypeError:
            raise TypeError("bin_edges is not a sequence of floats, instead is: %s" % type(bin_edges))

    # Check decode_method is either a string or implements __call__
    if not (isinstance(decode_method, str) or callable(decode_method)):
        raise TypeError("decode_method should be either a string or callable, instead is: %s" % type(decode_method))

    ret = []

    # Apply over each column in vals
    for col in vals.T:
        if isinstance(decode_method, str):

            if string_param(decode_method, ['range', 'str', 'string']):
                _str = '[%d, %d)' if np.issubdtype(bin_edges.dtype, np.integer) else '[%f, %f)'
                ret.append(np.array([(_str % (low, high)) for low, high in zip(bin_edges[col], bin_edges[col + 1])]))
            elif string_param(decode_method, ['left', 'small', 'smallest', 'min', 'minimum']):
                ret.append(bin_edges[col])
            elif string_param(decode_method, ['right', 'large', 'largest', 'max', 'maximum']):
                ret.append(bin_edges[col + 1])
            elif string_param(decode_method, ['mid', 'middle', 'midpoint', 'average', 'mean']):
                ret.append((bin_edges[col] + bin_edges[col + 1]) / 2)
            else:
                raise ValueError("Unknown binning decode_method: %s" % decode_method)

        else:
            # If we are doing a function call, apply our function
            try:
                ret.append(to_numpy(decode_method(bin_edges[col], bin_edges[col + 1])).reshape(-1))
            except Exception as e:
                raise UserFunctionCallError("Attempted to call decode_method function, but got error: " + str(e))

            # In case the decode_method function does not return an array the same size as col
            if len(ret[-1]) != len(col):
                raise ValueError("Callable decode_method should return an array the same size as each input array."
                                 "Instead returned array of size() = %d" % len(ret[-1]))

    # Convert to numpy array and take transpose since it's a list of columns instead of rows
    return np.array(ret).T
