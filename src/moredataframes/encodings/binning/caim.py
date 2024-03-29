"""
CAIM binning technique described in paper here:
http://biomine.cs.vcu.edu/papers/IC-AI2001.pdf
"""
from moredataframes.encodings.binning import binning_function
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo, Optional, Union, NDArray, Any
from moredataframes.mdfutils import to_numpy
import numpy as np
import pandas as pd


@binning_function
def caim(vals: ArrayLike, encoding_info: EFuncInfo, labels: ArrayLike = None,
         max_bins: Optional[Union[int, None]] = None) -> NDArray[Any]:
    """
    Implements the CAIM binning algorithm seen here:
    http://biomine.cs.vcu.edu/papers/IC-AI2001.pdf

    Algorithm overview:
        1. Sort the column
        2. Find all boundary points (value where class label and value changes)
        3. Start out with one bin containing all values, and a variable current_caim = 0
        4. For every boundary point: compute the CAIM value if we were to add that point into the bin list
            CAIM(C, D | F) = (1/n) * sum(max(q_i) / M_i for i in range n)

            Where C is the classes, D is the current discretization method, F is the current attribute (column),
                n is the number of bins in the current D, max(q_i) is the maximum value in the class frequency counts
                for the bin i, and M_i is the total number of datapoint that are in the bin i
        5. Pick the boundary point with the largest CAIM value c. If c > current_caim, then set current_caim = c,
            add that boundary point into our current list, and repeat from step 3. Otherwise, exit.

    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into when inverse=False, or a dictionary that
        contains the encoding information when inverse=True. The encoding_info is a separate, empty dictionary created
        for each call to the encoding function. Modify as you wish within this function call without fears of
        overwriting other data.
    :param labels: the labels for the data (required). If left as None, an error is raised. Should be a sequence of
        elements the same length as the input data vals, and should be able to be converted to a numpy array, as well
        as have the ability to determine uniqueness with np.unique().
    :param max_bins: if not None, then an int >= 2 specifying the maximum number of bins to use. Will stop the 
        algorithm if the number of bins reaches this value, even if the new CAIM value is greater than the current one.
    :return: a numpy array
    """
    # Enforce that classes is integers
    labels = to_numpy(labels).reshape(-1)
    num_classes = len(np.unique(labels))
    labels, label_labels = pd.factorize(labels) if not np.issubdtype(labels.dtype, np.integer) or \
                                                   num_classes != np.max(labels) - 1 else (labels, None)

    ret_bins = []

    # Go through each column, applying the chimerge algorithm
    for col in vals.T:

        # Sort the col and find all the boundary points, while making sure to copy the labels
        arg_sort = np.argsort(col)
        col = col[arg_sort]
        labels_sorted = labels.copy()[arg_sort]

        # Left such that if boundary_indices[i] = x, then labels[x - 1] != labels[x], but labels[x - 1] may be equal
        #    to labels[x - 2]. IE: the i-th partition is the labels range [boundary_indices[i - 1], boundary_indices[i])
        boundary_indices = np.argwhere(col[:-1] != col[1:]).reshape(-1) + 1

        current_caim = 0
        current_counts = np.unique(label, return_counts=True)[1].reshape([-1, 1])
        bins = []

def _get_best_cut_point(labels, boundary_points, bins, num_classes, current_counts):
    best_point = None
    best_val = -100
    best_old_c, best_new_c = None
    for i, b in enumerate(boundary_points):
        # Find the bin we are testing
        idx = np.searchsorted(bins, b, side='right') - 1
        larger_side_left = b - bins[idx] >= bins[idx + 1] - b
        min_idx, max_idx = bins[idx], bins[idx + 1]

        old_c = current_counts[:, idx].copy()
        new_c = np.zeros(shape=[num_classes])

        un, c = np.unique(labels[b:max_idx] if larger_side_left else labels[min_idx:b], return_counts=True)
        old_c[un] -= c
        new_c[un] += c

        caim_val = (1 / len(bins)) * sum([(np.max(current_counts[:, i]) / np.sum(current_counts[:, i]))
                                          if i != idx else 0 for i in range(len(bins) - 1)])
        caim_val += (1 / len(bins)) * sum([(np.max(a) / np.sum(a)) for a in [old_c, new_c]])

        if caim_val > best_val:
            best_point = b
            best_val = caim_val
            best_old_c = old_c
            best_new_c = new_c

    new_counts =
    return best_point, best_val

        