"""
Provides the ability to easily apply functions to a pandas dataframe-like object
    column by column with functionality including default encodings, and regex
    matching, along with some built-in encoding functions.
"""

import numpy as np
import pandas as pd
import re
import math
from scipy.stats import norm
import functools

_DEFAULT_STR = "__default__"
_INIT_STR = "__init__"
_FINALLY_STR = "__finally__"
_SPECIAL_ENCODINGS = [_DEFAULT_STR, _INIT_STR, _FINALLY_STR]


def encode_dataframe(df, encodings, re_match=True, print_progress=False):
    """
    Takes the given dataframe, and encodes each column as specified in encodings.

    :param df: the dataframe to encode
    :param encodings: a dictionary with keys as strings for column names and 
        values that correspond to how to encode each column.

        If re_match is True, the keys are treated as regular expressions, for
        example, the encoding:

            encodings = {
                '.*re.*': 'drop'
            }

        would drop the columns 'red', 'green' but not the columns 'blue', 'yellow'.

        Special Keys:
            - '__default__': the encoding to be used for all columns that are in df, 
                but not in encodings. If this key does not exist, then all columns
                that do not match any keys will be copied directly

            - '__init__': This encoding is performed on every column before encoding

            - '__finally__': This encoding is performed on every column after encoding

        Encoding keys are case-sensitive.
        The encodings themselves are functions that take in a pandas dataframe, 
            perform some computation on that dataframe column by column, and return the 
            resultant dataframe.
        If a column in df does not match any key in encodings, then that column will be
            copied over without applying any function to it.

    :param re_match: If True, will match keys in encodings to keys in df as if the 
        encoding keys were regex's
    :param print_progress: If True, will print the progress

    :return: the encoded dataframe
    """

    ret_df = pd.DataFrame()

    for column_name in df.columns:
        if print_progress:
            print("Encoding:", column_name)

        key_match = None
        for current_key in encodings.keys():

            # If we are doing regular expressions, or if not and they are equal
            if (re_match and re.fullmatch(current_key, column_name)) \
                or (current_key == column_name):

                if key_match is not None:
                    raise DuplicateMatchError(column_name, key_match, current_key)

                else:
                    key_match = current_key

        if key_match is None:
            if _DEFAULT_STR in encodings.keys():
                append_df = encodings[_DEFAULT_STR](pd.DataFrame(df[column_name]))
            else:
                append_df = pd.DataFrame(df[column_name])
        else:
            # Need to turn the column into a dataframe before applying function
            append_df = encodings[key_match](pd.DataFrame(df[column_name]))

        for col in append_df.columns:
            ret_df[col] = append_df[col]

    return ret_df


def no_op(df):
    """
    No operation, just returns the dataframe
    """
    return df


def drop(df):
    """
    Drops the column (returns an empty dataframe no matter the input df)
    """
    return pd.DataFrame()


def normalize(df):
    """
    Normalizes the dataframe by dividing by the largest value
    """
    return df / df.max().max()


def to_defined_buckets(df, buckets):
    """
    Splits the data into n + 1 buckets described by the list of integers 
        buckets [x_1...x_n]. All values below x_1 will be placed into bucket
        '0', all values above x_n will be placed into bucket n, and all 
        values inbetween will be sorted into their bucket depending on which 
        x_i, x_(i+1) the value falls inbetween. Bucketing is determined from left to
        right, ie: the value 'i' is returned such that buckets[i] <= v < buckets[i+1]
        for every value v in df
    
    :param df: the dataframe
    :param buckets: a strictly increasing list of real numbers of size > 1
    """
    if len(buckets) <= 1:
        raise ValueError("length of buckets must be int > 1, got: %d" % len(buckets))
    return pd.DataFrame(np.digitize(df, buckets, right=False), columns=df.columns)


def to_buckets(df, n_buckets, suppress_warnings=False):
    """
    Splits the data into n buckets. The buckets are equal size between the 
        minimum and the maximum of the dataset.
    
    :param df: the dataframe
    :param n_buckets: the number of buckets to use, must be > 1
    :param suppress_warnings: this function can produce some warning messages
        (for example, there being only one unique value in df), and if this
        is True, then those messages will be suppressed.
    """
    if len(df) == 0:
        return df
    if n_buckets <= 1:
        raise ValueError("n_buckets must be int > 1, got: %d" % n_buckets)

    if len(np.unique(df)) == 1:
        if not suppress_warnings:
            print("Warning: only one unique value while bucketing")
        bins = [df.min().min(), df.min().min() + 1]
    else:
        _min = df.min().min()
        _max = df.max().max()
        bins = list(np.arange(_min, _max, (_max - _min) / (n_buckets - 1))) + [_max + 1]

    return pd.DataFrame(np.digitize(df, bins, right=False).astype(int), columns=df.columns)


def to_percent_buckets(df, n_buckets, initial=0, cutoff=0.01, discrete=False):
    """
    Buckets the given dataframe based on what percent of the dataset each
    unique value takes up (in the case of a discrete dataset), or based on
    what percent of the total area of the normal distribution with mean and
    std taken from the dataset.
    
    :param df: the dataframe to bucket
    :param n_buckets: the number of buckets to use (excluding any initial)
    :param initial: if > 0, the number of buckets to use for an initial bucketing.
        Takes all of the most common unique values in the dataset until
        (1 - cutoff)% of the dataset is used, then sorts these into a
        discrete percent bucketing. It then sorts the rest of the values
        into whatever percent bucketing is initially passed.
        This is useful if the data has large spikes in specific values
        and does not follow a pseudo-normal distribution, yet still
        has a large number of unique values that should be sorted into
        a smaller number of bins.
    :param cutoff: the cutoff to use if initial > 0
    :param discrete: if True, buckets based on the frequency of each unique
        value in the dataset. If False, buckets using cutoffs based on
        what percent of the total area of the normal distibution with mean
        and standard deviation takes from the dataset. Bucketing should
        be done discretely if the number of unique values is relatively
        small compared to the size of the dataset. Bucketing should not
        be discrete if there are many unique values in the dataset, and
        it follows a pseudo-normal distribution.
    """
    if len(df) == 0:
        return df
    if n_buckets <= 1:
        raise ValueError("n_buckets must be int > 1, got: %d" % n_buckets)

    vals = df.values.reshape([-1])
    ret = np.empty(len(vals))
    
    if initial > 0:
        # Make uniques and counts and sort them by counts
        uniques, counts = np.unique(df, return_counts=True)
        pos = np.argsort(counts)[::-1]
        counts = counts[pos]
        uniques = uniques[pos]

        # Make the total running sum and stopping count (based on cutoff)
        total = np.cumsum(counts)
        stop_val = total[-1] * (1 - cutoff)

        # Get the first few values that we will be working on
        p = np.searchsorted(total, stop_val, side='right')
        total = total[:p + 2 if p != 0 else 1]
        initial_uniques = uniques[:len(total)]

        # Do the initial bucketing
        mask = np.isin(vals, initial_uniques)
        ret[mask] = _pb_discrete(vals[mask], initial)

        ret[~mask] = _pb(vals[~mask], n_buckets) if not discrete else _pb_discrete(vals[~mask], n_buckets)
        ret[~mask] += max(ret[mask]) + 1
    else:
        ret = _pb(vals, n_buckets) if not discrete else _pb_discrete(vals, n_buckets)

    return pd.DataFrame(ret.astype(int).reshape(df.shape), columns=df.columns)


def _pb(vals, n_buckets):
    loc = np.mean(vals)
    scale = np.std(vals)

    if scale == 0:
        m = min(vals)  # Could take either min or max since std is 0
        bins = [m, 1.00000001 * m]  # add in a small increment for max of bin

    else:
        # Split based off standard deviation and inverse c.d.f of normal dist
        bins = [norm.ppf((i + 1) * (1.0 / n_buckets), loc=loc, scale=scale) 
                for i in range(n_buckets - 1)]

    return np.digitize(vals, bins, right=False)


def _pb_discrete(vals, n_buckets):

    # Get all the unique values and their counts, and order them in descending
    #   order by counts
    uniques, counts = np.unique(vals, return_counts=True)
    pos = np.argsort(counts)[::-1]
    counts = counts[pos]
    uniques = uniques[pos]

    # The running total sum of the array
    total = np.cumsum(counts)

    bins = []
    current_sum = 0
    for i in range(n_buckets - 1):
        if len(uniques) == 1:
            break
        
        # The sum to stop at is remaining_buckets % of current counts sum of remaining uniques
        stop_sum = current_sum + (total[-1] - current_sum) / (n_buckets - i)

        # Searchsorted through counts to find the index of the value that exceeds stop_sum (use side=right)
        idx = np.searchsorted(total, stop_sum, side='right')

        # Add in the values to keep in this bin, then remove them from both uniques and total
        #   while changing the current sum
        bins.append(uniques[:idx + 1])
        uniques = uniques[idx + 1:]
        current_sum = total[idx]
        total = total[idx + 1:]
    
    # Add the last few uniques in
    bins.append(uniques)
    ret = np.full(len(vals), -1)

    for i, bin in enumerate(bins[:-1]):
        places = np.argwhere(np.isin(vals, bin))
        ret[places] = i

    # Fill in the remaining
    ret[ret == -1] = len(bins) - 1
    
    return ret


def to_binary(df, neg=None, n_bits=None):
    """
    Converts the dataframe into a binary representation of the values. Values
        must be positive integers. There will be a separate column for each
        bit in each binary number with the same column name as the original
        and the appended "_BIN_i" for all i's for each column used.
    
    :param df: the dataframe
    :param neg: Whether or not to allow negative values. Can be True/False or None
        If True, then negative values will be assumed, and an extra
            column with the appended name "_BIN_NEG" will be added and the binary
            will turn into a one's complement representation. 
        If False, then it will be assumed no negative values exist, and if any
            any are found, then an error will be shown.
        If None, then the value of neg will be assumed from whether or not
            there are any negative values in the dataframe.
    :param n_bits: the number of bits to use. If None, then the number of bits
        used will be the minimum number of bits needed to encompass all integers
        in df. If not None, and values fall outside the range capable of
        representation by this number of bits, then the value will instead become
        all 1's (same as constraining to the value +/- 2^n_bits - 1)
    """
    # Return empty df now because we need to index a non-empty one
    if len(df) == 0:
        return df
    if not isinstance(df.iloc[0, 0], (np.integer, int)):
        raise ValueError("Values must be integers for binary encoding")
    
    if neg is False and df.min().min() < 0:
        raise ValueError("Values cannot be negative when converting to binary " 
            "if 'neg' is False. Found negative value: %d" % df.min().min())
    elif neg is None:
        neg = df.min().min() < 0
    
    if n_bits is None:
        n_bits = math.ceil(math.log2(
            max(abs(df.min().min()), abs(df.max().max()))
        ))
    else:
        # Convert all values outside the range to within
        # Copy df so as to not change anything
        df = df.copy()
        df[df > 2**n_bits - 1] = 2**n_bits - 1
        df[df < -(2**n_bits - 1)] = -(2**n_bits - 1)

    ret_df = pd.DataFrame()
    for col in df.columns:
        if neg:
            ret_df[col + "_BIN_NEG"] = (df[col] < 0).astype(int)

        # Find the binary number by doing bit shifts and AND's
        adfc = df[col].abs()
        for i in range(n_bits):
            ret_df[col + ("_BIN_%d" % i)] = \
                (np.right_shift(adfc, n_bits - i - 1) & 1).astype(int)

    return ret_df


def to_onehot(df, max_n=None):
    """
    Makes each column into a one-hot representation. Values in df must be 
        positive integers. 
    Column names will have a "_HOT_i" appended for each 'i' value needed 
        from 0 to the number of columns.
    
    :param df: the dataframe
    :param max_n: the maximum value to assume in the dataframe. 
        If no values in the dataframe are equal to max_n, then the last 
            column(s) added will end up being all 0's. 
        If any values in the dataframe are greater than max_n, then an error
            will be shown. 
        If max_n is None, then it will be set to the maximum value in the 
            dataframe.
    """
    # Return empty df now because we need to index a non-empty one
    if len(df) == 0:
        return df
    if not isinstance(df.iloc[0, 0], (np.integer, int)):
        raise ValueError("Values must be integers for onehot encoding")

    # Should be a dataframe, so .max().max() is the way to go
    if max_n is None:
        max_n = df.max().max()
    elif max_n < 0:
        raise ValueError("onehot max_n must be non-negative integer, got: %d" % max_n)
    elif df.max().max() > max_n:
        raise ValueError("Found a value in the dataframe larger than max_n."
            "Value: %d, max_n: %d" % (df.max().max(), max_n))
    
    if df.min().min() < 0:
        raise ValueError("Cannot onehot encode negative values, found: "
            + str(df.min().min()))

    ret_df = pd.DataFrame()
    for col in df.columns:
        for i in range(max_n + 1):
            ret_df[col + ("_HOT_%d" % i)] = (df[col] == i).astype(int)

    return ret_df


def string_to_bool(df, true_val='True', ignore_case=True):
    """
    Converts the entire dataframe of strings, column by column, into booleans.
    True if the value is equal to true_val, false otherwise.

    :param df: the dataframe
    :param true_val: the string value for True
    :param ignore_case: if True, then strings are converted to all lowercase 
        before comparison
    """
    ret = pd.DataFrame()

    for col in df.columns:
        if ignore_case:
            ret[col] = np.where(df[col].str.lower() == true_val.lower(), 1, 0)
        else:
            ret[col] = np.where(df[col].str == true_val, 1, 0)

    return ret


# An error for if there are multiple possible non-unique encodings for a given 
#   column name
class DuplicateMatchError(Exception):
    def __init__(self, column_name, key_match, current_key):
        self.message = ("The column: %s, is matched by both keys: \"%s\"  and  "
            "\"%s\"" % (column_name, key_match, current_key))
