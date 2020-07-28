"""
Contains all the utility functions for now
"""

import pandas as pd
import numpy as np
import math

# Checks to make sure numba is installed, implement as a decorator function
try:
    import numba
    def check_for_numba(func, *args, **kwargs):
        if 'normal_jit' in kwargs and kwargs['normal_jit'] == True:
            return numba.jit(func, *args, **kwargs)
        return numba.njit(func, *args, **kwargs)
except ImportError:
    print("Warning: numba is not installed, running in no-numba mode. Install it to drastically improve speed.")
    def check_for_numba(func, *args, **kwargs):
        return func


def partition(arr):
    """
    Returns a list of indicies that designate boundaries for partitions of the
        data. For example:
            partition([0, 0, 0, 1, 2, 2, 2, 3, 4, 5])
        would return the list:
            [0, 3, 4, 7, 8, 9, 10]
        
    Assumes arr is already sorted, otherwise behavior is undefined

    :param arr: the sorted array to partition
    :return: a numpy array of indicies describing partitioning of the data
    """
    arr = arr.values if hasattr(arr, "values") else np.array(arr)
    ret = np.where(arr[:-1] != arr[1:])[0] + 1
    return np.insert(ret, [0, len(ret)], [0, len(arr)])


def partition_multiple(cols):
    """
    Partitions based on multiple columns. IE: if a value changes in any
        column, then that is considered a new index of partitioning. Each
        column must be the same length
    :param cols: a list-like object of 1-D lists with at least one sublist
    :return: a numpy array fo indicies describing the partitioning of the
        data on multiple columns
    """
    cols = cols.values if hasattr(cols, "values") else cols if isinstance(cols, np.ndarray) else np.array(cols).T
    return np.sort(np.unique(np.concatenate([partition(cols[:, i]) for i in range(cols.shape[1])])))


def keep_min_max_examples(data, min_examples=None, max_examples=None, groupby=None,
                          examples_cols=None, return_removed_min_indices=False, already_sorted=False):
    """
    Keeps only the datapoints that contain at least min_examples unique examples, and
        will keep up to max_examples unique examples of each datapoint, grouped by groupby,
        and using the columns examples_cols to determine what is a "unique" example.
    If there are multiple of any unique value, they are treated as one value and all
        kept/removed together.

    :param data: the dataframe, dataframe column, array, or list
        of values to keep min examples from.
    :param min_examples: the minimum number of examples (inclusive)
        necessary to keep each datapoint group
    :param max_examples: the maximum number of examples (inclusive)
        to keep from each datapoint group
    :param groupby: the column or columns to groupby. Only used if data
        is two-dimensional. Can be integers corresponding to the column
        indexes to group by, or column names in the case of a dataframe.
        If left as None with two-dimensional data, then the first column
        will be used to group the data.
    :param examples_cols: if not None, then these columns will be used to 
        determine what a "unique" example is. Can be integers corresponding 
        to the column indexes to group by, or column names in the case of a
        dataframe. Every unique value over these columns will be considered
        a single example, and any values that are not unique will be grouped
        together and considered to be one single example. If None, then this
        will not be done and the number of examples per group will be the
        total number of datapoints.
    :param return_removed_min_indices: if True, will also return the indices of 
        the datapoint removed due to there being too few examples of them
    :param already_sorted: if True, will skip the sorting step to save
        on computation; assumes data is already sorted on the groupby cols
    :return: either a pandas dataframe or a numpy array
    """

    # Just in case the user is stupid
    if min_examples is None and max_examples is None:
        raise TypeError("Both min_examples and max_examples cannot be None.")

    groupby, examples_cols, ispandas, data = _fgep(groupby, examples_cols, data)
    
    # Sort if not already_sorted
    if not already_sorted:
        data = _not_already_sorted(data, groupby + (examples_cols if examples_cols is not None else []), ispandas)

    def _f(arr):
        gparts = partition_multiple(arr[:, groupby])
        eparts = np.arange(0, gparts[-1]) if examples_cols is None else partition_multiple(arr[:, examples_cols])
        return _min_max_idx(gparts, eparts, min_examples, max_examples)

    keeps, removes = _f(data.values if ispandas else np.array(data).reshape([len(data), -1]))
    ret = data.iloc[keeps] if ispandas else data[keeps]
    removes = data.iloc[removes] if ispandas else data[removes]
    return (ret, removes) if return_removed_min_indices else ret


def split_by_group(data, sizes, groupby=None, examples_cols=None, randomize=True, 
                   random_state=None, already_sorted=False):
    """
    Groups the data by groupby, and splits up each group based on the percent
        to use for each size in sizes. (Makes it so you don't have to rely on
        law of large numbers to make sure each split of the dataset contains
        an acceptable ammount of examples). Fills with at least one example
        per size in the order of sizes that are not -1's, then fills in the
        sizes that are -1's.
    
    :param data: the dataframe, dataframe column, array, or list to split.
        If datatype is not a numeric or string, or if the lists passed are
        of different sizes, behavior is undefined.
    :param sizes: a list of floats between [0.0, 1.0] describing the percent
        of each group's examples to use for that set. The value -1 can
        be used to take up the rest of the dataset not used in the positive
        float values. If more than one -1 exists, then all the extra
        examples will be split evenly between all -1 values. The total sum
        of all values that are not -1 must be <= 1.0
    :param groupby: the column or columns to groupby. Can be integers 
        corresponding to the column indexes to group by, or column names
        in the case of a dataframe. If left as None with two-dimensional 
        data, then the first column will be used to group the data.
    :param examples_cols: if not None, then these columns will be used to 
        determine what a "unique" example is. Can be integers corresponding 
        to the column indexes to group by, or column names in the case of a
        dataframe. Every unique value over these columns will be considered
        a single example, and any values that are not unique will be grouped
        together and considered to be one single example. If None, then this
        will not be done and the number of examples per group will be the
        total number of datapoints.
    :param randomize: if True, then the values will be taken randomly from
        each group. If False, then the values will be taken from the start
        of each group down in the order: training, test, validation sets.
    :param random_state: the value to use as the seed for the RNG
    :param already_sorted: if True, will skip the sorting step to save
        on computation; assumes data is already sorted on the groupby cols
    :return: the split data, as a tuple if there are multiple datasets
        to return in the order given in sizes
    """

    # Make sure the percent sizes passed are good
    if len(sizes) == 0:
        raise ValueError("Sizes must not be empty list")
    for n in sizes:
        if not (0 <= n <= 1) and n != -1:
            raise ValueError("Dataset size outside of bounds [0.0, 1.0]: %f" % n)
    s = sum([n for n in sizes if n != -1])
    if s > 1:
        raise ValueError("Total percent size for all sets is > 1: %f" % s)

    # If data is empty
    if len(data) == 0:
        return [[] for s in sizes]

    groupby, examples_cols, ispandas, data = _fgep(groupby, examples_cols, data)

    # If the columns given to data are empty
    if 0 in data.shape:
        return [[] for s in sizes]
    
    # Sort if not already_sorted
    if not already_sorted:
        data = _not_already_sorted(data, groupby + (examples_cols if examples_cols is not None else []), ispandas)

    arr = data.values[:, groupby + (examples_cols if examples_cols is not None else [])] if ispandas else data[:, groupby]
    gparts = partition_multiple(arr[:, groupby])
    examples = partition_multiple(arr[:, examples_cols]) if examples_cols is not None else np.arange(gparts[-1])
    all_parts = np.sort(np.unique(np.concatenate((gparts, examples)))) 
    splits = _split_all_parts(all_parts, gparts)

    # Set up for RNG
    RNG = np.random.default_rng(random_state)

    ret_idxs = [[] for s in sizes]
    for i in range(len(splits) - 1):
        start, end = splits[i], splits[i + 1]
        num_unique = end - start

        # If we are doing random things, randomize the array
        rands = np.arange(num_unique) + start
        if randomize:
            RNG.shuffle(rands)

        for _idx, _size in enumerate(sizes):
            if _size == -1 or _size == 0:
                continue

            add = min(math.ceil(_size * num_unique), len(rands))

            ret_idxs[_idx] += list(_make_keeps(all_parts[rands[:add]], all_parts[rands[:add] + 1]).astype(int))
            rands = rands[add:]

            # If we run out of examples, break
            if len(rands) == 0:
                break
        
        else:
            # If the loop completes normally, fill in the -1's, if there are any
            
            if len([s for s in sizes if s == -1]) == 0:
                return [(data.iloc[ris] if ispandas else data[ris]) for ris in ret_idxs]

            # Fill in the -1's
            for _idx, _size in enumerate(sizes):
                if _size != -1:
                    continue
                
                neg_size = math.ceil(len(rands) / len([s for s in sizes[_idx:] if s == -1]))
                add = min(neg_size, len(rands))

                ret_idxs[_idx] += list(_make_keeps(all_parts[rands[:add]], all_parts[rands[:add] + 1]).astype(int))
                rands = rands[add:]

                # If we run out of examples, break
                if len(rands) == 0:
                    break


    return [(data.iloc[ris] if ispandas else data[ris]) for ris in ret_idxs]
                

@check_for_numba
def _make_keeps(starts, ends):
    total_size = np.sum(ends - starts)
    ret = np.empty(total_size)
    size = 0
    for i in range(len(starts)):
        start, end = starts[i], ends[i]
        s = end - start
        ret[size:size + s] = np.arange(start, end)
        size += s
    return ret
   

def _min_max_idx(gparts, eparts, min_examples, max_examples):
    """
    Returns the indices to keep for min_examples and max_examples,
        along with the indices that were removed due to there being
        too few examples
    """
    # Put all the gparts and eparts together, make sure they are sorted just in case
    all_parts = np.sort(np.unique(np.concatenate((gparts, eparts))))

    # Split up all_parts based on gparts
    splits = _split_all_parts(all_parts, gparts)

    keeps = []
    removes = []

    # Go through each gpart bin keeping each all_part increment that is of size 
    #   min_examples or larger, and keeping up to max_examples of them
    for i in range(len(splits) - 1):
        start, end = splits[i], splits[i + 1]

        if min_examples is not None and end - start < min_examples:
            removes += list(range(all_parts[start], all_parts[end]))
        
        elif max_examples is not None and end - start > max_examples:
            keeps += list(range(all_parts[start], all_parts[start + max_examples]))

        else:
            keeps += list(range(all_parts[start], all_parts[end]))

    return keeps, removes


@check_for_numba
def _split_all_parts(all_parts, gparts):
    ret = [0]

    # We only need to go from gparts[1] -> gparts[-1] cause first and last are obvious
    for i in range(len(gparts) - 2):
        start, val = ret[-1], gparts[i + 1]
        idx = np.searchsorted(all_parts[start:], val) + start
        ret.append(idx)
    
    ret.append(len(all_parts) - 1)
    return ret


def _fgep(groupby, examples_cols, data):
    """
    Makes sure groupby is a list of elements, and [0] if None,
    makes sure examples_cols is a list of elements, and returns None
    if None, and returns a boolean as to whether or not data is pandas
    """
    ispandas = hasattr(data, "values")
    groupby = [0] if groupby is None else list(groupby) if _is_list(groupby) else [groupby]
    if ispandas and not isinstance(groupby[0], int):
        groupby = [list(data.columns).index(g) for g in groupby]
    
    if examples_cols is not None:
        examples_cols = list(examples_cols) if _is_list(examples_cols) else [examples_cols]
        if ispandas and not isinstance(examples_cols[0], int):
            examples_cols = [list(data.columns).index(e) for e in examples_cols]
    
    if not ispandas and not isinstance(data, np.ndarray):
        data = np.array(data)
        if len(data.shape) > 1:
            data = data.T
        else:
            data = data.reshape([len(data), -1])

    return groupby, examples_cols, ispandas, data


def _is_list(l):
    """
    Checks iterability and indexing to see if l is a list
    """
    try:
        for _ in l:
            break
        if len(l) > 0 and l[0] is not None:
            pass
        return not isinstance(l, dict)
    except TypeError:
        return False


def _not_already_sorted(data, groupby, ispandas):
    """
    Sorts the data by the given groupby columns (list of ints)
    """
    if ispandas:
        return data.iloc[np.lexsort([data.values[:, i] for i in reversed(groupby)])]
    return data[np.lexsort([data[:, i] for i in reversed(groupby)])]


