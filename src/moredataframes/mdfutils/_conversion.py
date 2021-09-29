"""
Functions for converting types into other types (not during encode_df).
"""
import pandas as pd
import numpy as np
from moredataframes.mdfutils.typing import ArrayLike, NDArray, Any


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
