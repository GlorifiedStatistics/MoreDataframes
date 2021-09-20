"""
A collection of built-in encoding functions for use in moredataframes.encode_df()
"""
import pandas as pd
import numpy as np
from .mdf_typing import ArrayLike, NDArray, EFuncInfo, Any


def noop(vals: ArrayLike, encoding_info: EFuncInfo, inverse: bool = False) -> NDArray[Any]:
    """
    Perform no operation (identity function)
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into when inverse=False, or a dictionary that
        contains the encoding information when inverse=True. The encoding_info is a separate, empty dictionary created
        for each call to the encoding function. Modify as you wish within this function call without fears of
        overwriting other data.
    :param inverse: if False, then encode the values, otherwise decode it
    :return: a numpy array
    """
    return to_numpy(vals)


def drop(vals: ArrayLike, encoding_info: EFuncInfo, inverse: bool = False) -> None:
    """
    Drop the dataframe, and do not keep track of it for decoding.
    Perform no operation (identity function)
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into when inverse=False, or a dictionary that
        contains the encoding information when inverse=True. The encoding_info is a separate, empty dictionary created
        for each call to the encoding function. Modify as you wish within this function call without fears of
        overwriting other data.
    :param inverse: if False, then encode the values, otherwise decode it
    :return: a numpy array
    """
    return None


def factorize(vals: ArrayLike, encoding_info: EFuncInfo, inverse: bool = False) -> NDArray[np.int64]:
    """
    Factorizes each column, and stores the label information into encoding_info.
    Perform no operation (identity function)
    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into when inverse=False, or a dictionary that
        contains the encoding information when inverse=True. The encoding_info is a separate, empty dictionary created
        for each call to the encoding function. Modify as you wish within this function call without fears of
        overwriting other data.
    :param inverse: if False, then encode the values, otherwise decode it
    :return: a numpy array of dtype int64
    """
    vals = to_numpy(vals)
    ret = np.empty(vals.shape, dtype=np.int64)

    for c_idx in range(vals.shape[1]):
        ret[:, c_idx], encoding_info[c_idx] = pd.factorize(vals[:, c_idx])

    return ret


def to_numpy(vals: ArrayLike) -> NDArray[Any]:
    """
    Converts input values into 2d numpy array. If vals is 1d, then is reshaped into size [-1, 1]. If vals is not already
        a numpy ndarray, converts:
            - pandas DataFrame/Series into numpy using the to_numpy() function
            - anything else into numpy by calling numpy.array(vals)
    :raises: TypeError - if vals could
    :param vals: the values to convert
    :return: a numpy array
    """
    if isinstance(vals, (pd.DataFrame, pd.Series)):
        vals = vals.to_numpy()

    ret = vals if isinstance(vals, np.ndarray) else np.array(vals)

    if ret.ndim == 0:
        raise TypeError(f"Could not convert object of type '{type(vals)}' into an array.")

    if ret.ndim == 1 or (ret.ndim == 2 and ret.size == 0):
        ret = ret.reshape([-1, 1])
    elif ret.ndim > 2:
        raise ValueError(f"Input must be 2d-array, instead has {ret.ndim} dimensions and shape {ret.shape}")

    return ret
