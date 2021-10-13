"""
Utils for binning functions
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


def get_boundary_points(vals: NDArray[Any], labels: NDArray[Any], boundary_type='best') -> NDArray[Any]:
    """
    Returns the points that can act as boundary points. Boundary points are indices in the vals array at which we
        could conceivably partition for binning.
    :param vals: the values to find the boundary points in
    :param labels: the classes for the values
    :param boundary_type: the method for choosing boundary points. Can be:
        - 'vals' or 'change_vals': boundary points occur only when vals changes. This is generally the method most
            papers use to find boundary points. Note: this would be equivalent to the logical 'or' between 'vals' and
            'labels' since we cannot have a boundary point unless vals changes: IE we cannot have a bin halfway
            between 1 and 1.
        - 'labels' or 'change_labels': boundary points occur only when labels changes (or if the label changes within
            a single unique value in vals, then after that value). This is generally less effective than the other
            methods, but is decently fast
        - 'and': the logical and of 'vals' and 'labels' - boundary points occur when both vals and labels change at
            the same spot. This is normally far faster than other methods, but easily prone to bad boundary points
        - 'best': the best method implemented in terms of finding what are logically good boundary points, but also
            the slowest. A boundary point occurs only when ALL of the following conditions have been met:
                1. vals changes, and
                2. Either:
                    a. The previous boundary contains a single label, and this new label is different than that label
                    b. The previous boundary contains multiple labels
                    c. This value has multiple possible labels associated with it

            This helps to remove some boundary points that we would not split on. For example, the values and labels:
                vals = [0, 1, 2, 2, 4, 5, 6]
                labels = [0, 0, 0, 1, 1, 1, 1]
            The 'vals' method would give us boundary points of [1, 2, 3, 4, 5], which is too many bins since we can
                group values 4, 5, and 6 together since they have the same label
            The 'labels' method would give us boundary points of [4], which bins the values into [0, 4) and [4, 6].
                This leaves the problematic value of vals=2 (since it has labels of both 0 and 1), which could mess
                with the known "good" values of vals=0,1 which both have a label of 1. It may be better if we could
                separate out that '2' value into its own bin.
            The 'and' method would give us no boundary points since there is never a time when both vals and labels
                change on the same index. But, this is obviously a bad choice since there are two labels 0 and 1
            However, the 'best' method would instead give us boundary points of [2, 4] thus binning into bins with
                edges [0, 2), [2, 4), [4, 6]. This gives us the most number of 'ideal' bins (those with a single
                class label), and separates those bins with multiple class labels into their own, 'problematic' bins.
                This method can be useful for binning methods such as CAIM, in which this would help lower the number
                of excess bins produced.
    :return: a numpy array of boundary points
    """
    if len(vals) != len(labels):
        raise ValueError("Vals and labels should have the same length, instead have lengths: %d, %d" %
                         (len(vals), len(labels)))

    if len(vals) == 0:
        return np.array([])
    
    if string_param(boundary_type, ['vals', 'change_vals', 'cvals']):
        pass
    elif string_param(boundary_type, ['labels', 'change_labels', 'clabels']):
        pass
    elif string_param(boundary_type, ['and', 'logical_and', 'both']):
        pass
    elif string_param(boundary_type, ['best', 'ideal']):
        pass
    else:
        raise ValueError("Unkown boundary_type method: %s" % boundary_type)
