import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import math


def ngram_counts(df, ns, group_cols=None, partition_cols=None, extra_sort_cols=None,
                 ignore_cols=None, padding_val=None, return_totals=True, n_cores=None,
                 return_everything=False):
    """
    Computes counts of ngrams over every column. The dataframe is first sorted by 
        columns group_cols, then partition_cols, then extra_sort_cols
    
    Returns a dictionary of the ngram counts. If group_cols is None or empty, the 
        dictionary will take the shape:
        {
            n_1: {
                col_1: {
                    '[x_1, x_2, x_3, ..., x_(n_1)]': c_1
                    ...
                }
                ...
            }
            ...
        }

        where 'n_1' is each ngram, 'col_1' is the name of each column that is not
        an ignore_cols, a partition_cols, or a group_cols column, the list of x's
        is the list of unique values that creates each ngram (a string, equivalent
        to calling str(l) where l would be that list of values), and c_1 is the 
        count of the number of occurances of that ngram.

        If group_cols is not None, then an extra layer describing either the 
        singular grouping column, or a list of such columns, like so:
        {
            n_1: {
                col_1: {
                    g_1: {
                        '[x_1, x_2, x_3, ..., x_(n_1)]': c_1
                        ...
                    }
                }
                ...
            }
            ...
        }

        where 'g_1' describes the group column or columns. If there is only a 
        singular grouping column, then g_1 would just be the values in that 
        column eg: 'group_1' (Will always be a string). If it is a list of names,
        then it will be a string of the list of names like so: 
        "['group_col_name_1', 'group_col_name_2', ..., 'group_col_name_n']"

    :param df: the dataframe
    :param ns: an int or list of ints - the ngrams to run
    :param group_cols: the column or columns to group the dataset on. 
    :param partition_cols: the column or columns to partition the ngrams by. If 
        None, then there will be no partitioning on the columns and each column 
        will be treated as one long list of integers to compute the ngram counts. 
        If a column name or list of column names, then the dataset will be 
        partitioned (grouped) on those columns, and the ngram counts will be taken
        from each of these columns. This is different from group_cols because the 
        counts will be summed together in the end for each group.
    :param extra_sort_cols: if not None, the extra column or columns to sort on 
        before computing the ngrams. The dataframe will be sorted on group_cols 
        and partition_cols before sorting on extra_sort_cols. Any columns that 
        already exist in group_cols or partition_cols will be ignored.
    :param ignore_cols: columns to not compute ngram counts on. Ngram counts will
        already not be counted on group_cols nor partition_cols, so these would 
        be in addition to those columns.
    :param padding_val: the value to use to pad the ngrams. For example, given the
        list [1, 2, 3] to compute 3-grams, and the padding val of -1, the ngrams 
        that would be counted are:
            [-1, -1, 1], [-1, 1, 2], [1, 2, 3], [2, 3, -1], [3, -1, -1]
        If padding_val is None, then no values will be used to pad the ngrams, and
        any partitions with size < n will be ignored.
    :param return_counts: If true, will add in an extra key "__total__" into each
        column dicitonary which contains the total count of all ngrams for that
        column
    :param pool: if not None, a multiprocessing Pool initialized by the user that
        can be used to speed up computation with multiprocessing (computations are
        split on columns). Whenever a multiprocessing Pool is initialized, the
        entire memory is directly copied for each process. As such, it is often
        best to initialize the Pool at the beginning of the program, after the 
        imports, and before loading in any large datasets, so to avoid using
        too much needless memory.
    """
    group_cols = _ensure_list(group_cols)
    partition_cols = _ensure_list(partition_cols)
    extra_sort_cols = _ensure_list(extra_sort_cols)
    ignore_cols = _ensure_list(ignore_cols)
    ns = _ensure_list(ns)

    # Sort and partition if necessary
    all_cols = group_cols + [p for p in partition_cols if p not in group_cols] \
        + [e for e in extra_sort_cols if e not in partition_cols + group_cols]
    if len(all_cols) != 0:
        df = df.sort_values(by=all_cols)

    ignore_cols = [c for c in ignore_cols if c not in group_cols + partition_cols]
    if len(ignore_cols) > 0:
        df = df.drop(ignore_cols, axis=1)
    
    df = df.reset_index(drop=True)

    # Calculate all the partitions, merge them together, and sort them
    group_parts = [0, len(df)]
    for g in group_cols:
        group_parts += partition(df[g])
    group_parts = np.sort(np.unique(group_parts))

            
    partition_parts = []  # Don't need to sort this one
    for p in partition_cols:
        partition_parts += partition(df[p])
    partition_parts = np.array(partition_parts)

    all_parts = np.sort(np.unique(np.concatenate((group_parts, partition_parts))))

    # Make a temporary padding_val if needed
    if padding_val is None:
        int_cols = [df[c] for c in df.columns if df[c].dtype == np.int64]
        if len(int_cols) == 0:
            padding_val = -1
        else:
            padding_val = max([max(c) for c in int_cols]) + 1
        
    # Fill all of the val_cols using either the padding_val or the temporary one
    grammed_cols = [c for c in df.columns if c not in group_cols + partition_cols + ignore_cols]
    if pool is not None:
        args = [[df[c].values, c, all_parts, padding_val, max(ns)] for c in grammed_cols]
        f = pool.map(_make_vals_col_mp, args)
        val_cols = {c:arr for c, arr in f}
    else:
        val_cols = {c:_make_vals_col(df[c].values, c, all_parts, padding_val, max(ns))[1] for c in grammed_cols}
    
    # Make the group names, and change the group_partitions now that we have added in all 
    #   of the padding_val's
    if len(group_cols) == 1:
        group_names = df[group_cols[0]].iloc[group_parts[:-1]].values.astype('str')
    else:
        group_names = df[group_cols].iloc[group_parts[:-1]].values
        group_names = np.apply_along_axis(lambda r: str(list(r)), 0, group_names)
    new_group_parts = np.empty(len(group_parts))
    new_group_parts[1:] = group_parts[1:] + ((np.arange(0, len(group_parts) - 1) + 2) * (max(ns) - 1))
    new_group_parts[0] = 0
    group_parts = new_group_parts.astype('int')

    # Compute all of the actual ngrams
    ret = {}
    if pool is not None:
        # Split for mp based on columns
        for n in ns:
            args = [[v, c, group_parts, group_names, n] for c, v in val_cols.items()]
            ret[n] = {c:counts for c, counts in pool.map(_count_ngram_over_col, args)}
    else:
        for n in ns:
            ret[n] = {c:_count_ngram_over_col(v, c, group_parts, group_names, n)[1] for c, v in val_cols.items()}

    # Remove all useless padding_val's, and add in total counts
    for n, cols in ret.items():
        bad_gram = str([padding_val for i in range(n)])
        for c, groups in cols.items():
            for _, grams in groups.items():
                if bad_gram in grams:
                    del grams[bad_gram]
                
                if return_totals:
                    total = 0
                    for _, val in grams.items():
                        total += val
                    grams['__total__'] = total
    
    if return_everything:
        return val_cols, all_parts, df, ret
    return ret


def _count_ngram_over_col(col, col_name=None, group_parts=None, group_names=None, n=None):
    """
    Returns a dictionary of the counts per group in this column
    """
    # For the multiprocessing
    if col_name is None:
        return _count_ngram_over_col(*col)
    
    # A normal one
    ret = {}
    for i in range(len(group_parts) - 1):
        start, end, name = group_parts[i], group_parts[i+1], group_names[i]
        ret[name] = _make_dict(*_count_grams(col[start:end + n - 1], n))
    return col_name, ret


def _make_dict(uniques, counts):
    ret = {}
    for gram, val in zip(uniques, counts):
        ret[str(list(gram))] = val
    return ret


def _count_grams(arr, n):
    return np.unique(as_strided(arr, (max(arr.size + 1 - n, 0), n), (arr.itemsize, arr.itemsize)), 
                     return_counts=True, axis=0)

def _make_vals_col_mp(t):
    return _make_vals_col(*t)

@numba.njit(nogil=True)
def _make_vals_col(col, c, partitions, padding_val, n, start_pad=True, end_pad=True):
    """
    Adds in (n-1) padding_val's at the start and end of each partition
    Partitions should include the 0 at the beginning
    """
    
    ret = np.empty(len(col) + (len(partitions) - (0 if start_pad else 1) - (0 if end_pad else 1)) * (n - 1), dtype=type(padding_val))
    
    start = 0
    for i in range(len(partitions) - 1):
        length = partitions[i + 1] - partitions[i]

        # If we are starting with padding
        if (i == 0 and start_pad) or (i != 0):
            ret[start:start + n - 1] = padding_val
            ret[start + n - 1:start + length + n - 1] = col[partitions[i]:partitions[i + 1]]
            start += length + n - 1
        else:
            ret[start:start + length] = col[start:start + length]
            start += length

    if end_pad:
        ret[start:] = padding_val
    
    return (c, ret)


def list_from_string(s_l):
    """
    Converts the string into a list (assuming that list was turned into a string by the
        ngram_counts function). If the string was not initially a list (for example,
        there was only one column in group_cols), then a list with only that string
        as an element will be returned
    """
    if s_l[0] != '[':
        return [s_l]
    return [str(s) if "'" in s else float(s) if '.' in s else int(s) for s in s_l]


def _is_list(l):
    """
    Checks iterability and indexing to see if l is a 'list'
    """
    try:
        for _ in l:
            pass
        if len(l) > 0 and l[0] is not None:
            pass
        return not isinstance(l, dict) and not isinstance(l, str)
    except TypeError:
        return False


def _ensure_list(*args):
    ret = [[] if a is None else [a] if not _is_list(a) else list(a) for a in args]
    return ret if len(ret) > 1 else ret[0] if len(ret) == 1 else None
