"""
A collection of built-in encoding functions for use in moredataframes.encode_df()
"""
import pandas as pd
import numpy as np
import inspect
from moredataframes.mdfutils.typing import ArrayLike, NDArray, EFuncInfo, Any, Optional, Callable, Union
from moredataframes.mdfutils import to_numpy


def encoding_function(func: Callable[..., NDArray[Any]]) -> Callable[..., NDArray[Any]]:
    """
    Decorator for encoding functions to automatically unpack method kwargs from the given encoding_info and pass them
        to the encoding_function. This way, if any kwargs are stored in encoding_info for a future call, then they will
        be automatically used.
        Also converts the input to numpy array automatically, and handles None encoding_info's

        Note: Ignores the 'inverse' kwarg so encoding_info would not be able to override it
    :param func: the encodingfunction to decorated
    :return: a decorated encoding function
    """
    def wrapper(vals: ArrayLike, encoding_info: Optional[Union[EFuncInfo, None]] = None,
                inverse: Optional[bool] = False, **kwargs):

        encoding_info = {} if encoding_info is None else encoding_info
        variable_names = [n for n in inspect.signature(func).parameters][2:]  # Ignore the first two args

        # Make sure encoding_info does not contain 'inverse'
        if 'inverse' in encoding_info:
            del encoding_info['inverse']

        # Get all kwargs
        call_kwargs = {n: encoding_info[n] for n in variable_names if n in encoding_info}

        # If there are kwargs in **kwargs that are not in the given function's parameters, raise an error
        extra_kwargs = [k for k in kwargs if k not in variable_names]
        if len(extra_kwargs) > 0:
            raise TypeError("Unexpected keyword argument supplied: %s"
                             % (extra_kwargs if len(extra_kwargs > 1) else extra_kwargs[0]))
        call_kwargs.update(kwargs)

        return func(to_numpy(vals), encoding_info, inverse=inverse, **call_kwargs), encoding_info

    return wrapper


@encoding_function
def noop(vals: ArrayLike, encoding_info: EFuncInfo, inverse: Optional[bool] = False) -> NDArray[Any]:
    """
    Perform no operation (identity function)
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into, or to use if this is a subsequent call
        to this function and you wish to encode/decode in the same way as encoded previously
    :param inverse: if False, then encode the values, otherwise decode it
    :return: a numpy array
    """
    return to_numpy(vals)


@encoding_function
def drop(vals: ArrayLike, encoding_info: EFuncInfo, inverse: Optional[bool] = False) -> None:
    """
    Drop the dataframe, and do not keep track of it for decoding.
    Perform no operation (identity function)
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into, or to use if this is a subsequent call
        to this function and you wish to encode/decode in the same way as encoded previously
    :param inverse: if False, then encode the values, otherwise decode it
    :return: a numpy array
    """
    return None


@encoding_function
def factorize(vals: ArrayLike, encoding_info: EFuncInfo, inverse: Optional[bool] = False) -> NDArray[np.int64]:
    """
    Factorizes each column, and stores the label information into encoding_info.
    Perform no operation (identity function)
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into, or to use if this is a subsequent call
        to this function and you wish to encode/decode in the same way as encoded previously
    :param inverse: if False, then encode the values, otherwise decode it
    :return: a numpy array of dtype int64
    """
    ret = []

    for i, col in enumerate(vals.T):
        if inverse:
            col = encoding_info[i][col]
        else:
            col, encoding_info[i] = pd.factorize(col)
        ret.append(col)

    return np.array(ret).T
