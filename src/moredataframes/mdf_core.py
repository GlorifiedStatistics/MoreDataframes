"""
A collection of useful functions for manipulating/encoding pandas dataframes for data science.
"""
# import pandas as pd
# import numpy as np
from .mdf_typing import ArrayLike, EncodingDict


def encode_df(df: ArrayLike, encodings: EncodingDict):
    """
    Encodes the given data based on the given encodings. Returns a tuple of the encoded data, as well as a dictionary
        of the encoding information that can be passed into a future encode_data() call (if you want to encode another
        set of data in the same exact way the initial one was encoded), or that can be passed to a decode_df() call
        to automatically decode data back into its original form.


    :param df: a pandas DataFrame/Series, numpy Array, or some array-like object which can be converted into a numpy array
    :param encodings: the dictionary of encodings. Encodings are described above
    :return: a 2-tuple of the encoded data, and a dictionary of the encoding information
    """
    return None
