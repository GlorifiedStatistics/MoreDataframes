"""
A collection of built-in encoding functions for use in moredataframes.encode_df()
"""
import pandas as pd
import numpy as np
from moredataframes.mdfutils.typing import ArrayLike, NDArray, EFuncInfo, Any, Optional
from moredataframes.mdfutils import to_numpy


def noop(vals: ArrayLike, encoding_info: EFuncInfo, inverse: Optional[bool] = False) -> NDArray[Any]:
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


def drop(vals: ArrayLike, encoding_info: EFuncInfo, inverse: Optional[bool] = False) -> None:
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


def factorize(vals: ArrayLike, encoding_info: EFuncInfo, inverse: Optional[bool] = False) -> NDArray[np.int64]:
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
