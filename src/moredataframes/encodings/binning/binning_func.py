"""
Wrapper for binning functions
"""
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo, Any, ExpectedType, \
    Callable, NDArray, Optional, EncodingInfoExpectedType
from moredataframes.mdfutils import to_numpy, check_encoding_info, check_kwargs_types
from moredataframes.encodings.binning import decode_bins
from moredataframes.constants import ENCODING_INFO_BINS_KEY
import numpy as np


# The different types that could be used for defining the bins in a binning function
ENCODING_INFO_BINS_TYPES_TUPLE = (list, tuple, np.ndarray)


def binning_function(*encoding_info_keys_and_types: EncodingInfoExpectedType,
                     **kwarg_types: ExpectedType) \
        -> Callable[[Callable[..., NDArray[Any]]], Callable[..., NDArray[Any]]]:
    """
    Wraps a binning function to help abstract out some common functionality.
    Ensures that:
        - the input vals will be a numpy NDArray
        - the encoding_info will be a dictionary that has key, type pairs defined in encoding_info_keys_and_types.
            There is a default value of (ENCODING_INFO_BINS_KEY, ENCODING_INFO_BINS_TYPES_TUPLE) added to the 
                encoding_info_keys_and_types to make sure the bins exists in the encoding info on inverse=True
            * NOTE: This is only checked if inverse=True
            * NOTE: extra keys in encoding_info but not in encoding_info_keys_and_types will be ignored
        - if inverse=True, then decode_bins is called instead of the binning function
    :param func: the binning function
    :param encoding_info_keys_and_types: the sequence of tuples of strings and TypeConversions to check for type and
        key correctness
        NOTE: This is only checked if inverse=True
    :param kwarg_types: checks that the given kwargs have the given types. Each key should be a parameter kwarg that
        could be passed to the function, and value should be either a type or sequence of types where the value could
        be any one of the given types.
    :return: a binning function
    """
    encoding_info_keys_and_types.append((ENCODING_INFO_BINS_KEY, ENCODING_INFO_BINS_TYPES_TUPLE))

    def arg_wrapper(func: Callable[..., NDArray[Any]]) -> Callable[..., NDArray[Any]]:
        def wrapper(vals: ArrayLike, encoding_info: EFuncInfo, inverse: Optional[bool] = False,
                    **kwargs: Any) -> NDArray[Any]:
            vals = to_numpy(vals)

            # Check that encoding_info contains what it should, if inverse = True, then return the decoded values
            if inverse:
                check_encoding_info(encoding_info, encoding_info_keys_and_types)

                # We want two different function calls here so the default decode_method can be kept if I change it ever
                if 'decode_method' in kwargs:
                    return decode_bins(vals, encoding_info[ENCODING_INFO_BINS_KEY],
                                       decode_method=kwargs['decode_method'])
                return decode_bins(vals, encoding_info[ENCODING_INFO_BINS_KEY])

            # Otherwise, check if we are re-encoding something else in the same format, and thus need to call
            #   our own 'digitize'
            elif ENCODING_INFO_BINS_KEY in encoding_info:
                check_encoding_info(encoding_info, encoding_info_keys_and_types)
                check_kwargs_types({}, **kwargs)

            # The first two arg_names can be ignored since they are vals and encoding_info
            check_kwargs_types(kwarg_types, **kwargs)
            return func(vals, encoding_info, **kwargs)

        return wrapper

    return arg_wrapper
