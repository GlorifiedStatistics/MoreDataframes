"""
Implements the ChiMerge algorithm.

Resources:
https://medium.com/@nithin_rajan/data-discretization-using-chimerge-55c8ade3cfda

Paper:
https://dl.acm.org/doi/abs/10.5555/1867135.1867154
"""
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo, Optional, Union, NDArray, Any, List, Sequence
from moredataframes.mdfutils import to_numpy, check_for_numba
from moredataframes.encodings.binning import binning_function
from moredataframes.constants import ENCODING_INFO_BINS_KEY
import numpy as np
import pandas as pd
from scipy.stats import chi2


@binning_function((ENCODING_INFO_BINS_KEY, (tuple, list, np.ndarray)), max_bins=(None, int), min_bins=(None, int),
                  threshold=float)
def chi_merge(vals: ArrayLike, encoding_info: EFuncInfo, labels: ArrayLike = None,
              max_bins: Optional[Union[int, None]] = None, min_bins: Optional[Union[int, None]] = None,
              threshold: Optional[float] = 0.05) -> NDArray[Any]:
    """
    Implements the chimerge binning algorithm seen here:
    https://dl.acm.org/doi/abs/10.5555/1867135.1867154

    Algorithm overview:
        1. Sort the column
        2. Partition at every boundary point (value where class label changes)
        3. Compute the chi-square value on given labels between all adjacent partitions
            a. Sum is computed as:
                sum(sum( (A_ij - E_ij)^2 / E_ij for j in range(k)) for i in range(2))

                Where:
                    - k is the number of classes
                    - A_ij is the number of samples for the i-th interval with the j-th class label
                    - E_ij is the expected frequency of A_ij
                        * Defined as R_i * (C_j / N) where R_i is the number of samples in the i-th interval, C_j is
                            the total number of samples for class j over the two bins, and N is the total number of
                            samples in the two bins
                        * If E_ij is 0, then the division fails. So, we instead set E_ij to an arbitrary small value 0.1

        4. Merge the two partitions with the lowest chi-square value
        5. Repeat until some stopping criterion is met

    :param vals: the ArrayLike object to encode
    :param encoding_info: a dictionary to add encoding information into when inverse=False, or a dictionary that
        contains the encoding information when inverse=True. The encoding_info is a separate, empty dictionary created
        for each call to the encoding function. Modify as you wish within this function call without fears of
        overwriting other data.
    :param labels: the labels for the data (required). If left as None, an error is raised. Should be a sequence of
        elements the same length as the input data vals, and should be able to be converted to a numpy array, as well
        as have the ability to determine uniqueness with np.unique(). Assumes these labels have already been factorized
        (which should happen automatically as part of the @binning_function decorator)
    :param max_bins: if not None, then an int >= 2 specifying the maximum number of bins to use. Will continue the
        merging algorithm until there are <= this number of bins, even if the threshold value has been passed.
    :param min_bins: if not None, then an int >= 2 specifying the minimum number of bins to use. Will stop the merging
        algorithm if the next merge would cause the number of bins to fall below this value, even if the current
        lowest chi-square p-value is less than the threshold.
    :param threshold: a float in range (0, 1] specifying the p-value of the chi-square test at which, if the current
        lowest chi-square p-value is greater than this threshold, the merging algorithm is halted. This is overridden
        if min_bins or max_bins is not None. Choosing larger threshold values makes the merging process tend to go on
        longer, resulting in fewer bins and larger intervals.
    :return: a numpy array
    """
    vals, labels = to_numpy(vals), to_numpy(labels)
    min_bins, max_bins = 0 if min_bins is None else min_bins, np.inf if max_bins is None else max_bins

    # Check to make sure the lengths are the same
    if len(vals) != len(labels):
        raise ValueError("The labels should be the same length as the values to bin. Instead values length is: "
                         "%d while labels length is: %d" % (len(vals), len(labels)))

    # Check to make sure threshold makes sense
    if not 0 < threshold <= 1:
        raise ValueError("Threshold should be in the range (0, 1], instead is: %d" % threshold)

    ret = []

    # Find the number of classes, and the chi2 value of our threshold/p-value. For this, we use the inverse
    #   survival function of the chi2 distribution
    classes = np.unique(labels)
    chi2_threshold = chi2.isf(threshold, len(classes) - 1)

    # Go through each column, applying the chimerge algorithm
    for col in vals.T:

        # Sort the col and find all the boundary points, while making sure to copy the labels
        arg_sort = np.argsort(col)
        col = col[arg_sort]
        labels_sorted = labels.copy()[arg_sort]

        # Left such that if boundary_indices[i] = x, then labels[x - 1] != labels[x], but labels[x - 1] may be equal
        #    to labels[x - 2]. IE: the i-th partition is the labels range [boundary_indices[i - 1], boundary_indices[i])
        boundary_indices = np.argwhere(labels_sorted[:-1] != labels_sorted[1:]).reshape(-1) + 1

        # Compute all the chi-square values
        chi_squares = _chi_square_all(boundary_indices, labels_sorted, classes)

        # Find the minimum chi-square index and value
        min_idx = np.argmin(chi_squares)
        min_val = chi_squares[min_idx]

        # Iterate until stopping criterion is met
        while True:

            # Stop if we have min_bins bins, or if both the min_val is less than the threshold and we have <= max_bins
            #   bins (or in our case, len(boundary_indices) < max_bins since there are len(boundary_indices) + 1 bins
            if len(boundary_indices) < min_bins or (min_val > chi2_threshold and len(boundary_indices) < max_bins):
                break

            # Otherwise, we can merge the smallest chi2 value bins, and recompute their left and right chi2 values

            new_left = None
            new_right = None

            # Add in the new chi2 values, and update min_idx and min_val
            if new_left is not None:




    # Need to transpose since they were columns
    return np.array(ret).T


def _chi_square_all(boundaries: NDArray[Any], labels: NDArray[Any], classes: Sequence[Any]) -> List[float]:
    """
    Computes the chi-square values between each partition and returns them as a list.
    This function is left separate than _chi_square() to speed up computation using numpy's vectorized methods.
    :param boundaries: the indicies describing the boundaries of a partition, such that if boundaries[i] = x, then
        labels[x - 1] is in a different partition than labels[x], however it may be in the same partition as
        labels[x - 2]
    :param labels: the class labels
    :param classes: the list of unique classes in the entire column
    :return: a list chi-squares such that chi-squares[i] is the chi-square value between the 0-th and 1-st partition
        of labels
    """
    # Enforce that classes is integers
    labels, label_labels = pd.factorize(labels) if not np.issubdtype(labels.dtype, np.integer) or \
                                                   len(classes) != np.max(labels) - 1 else (classes, None)

    return _numba_accelerate_chi_square_all(boundaries, labels, len(classes))


@check_for_numba()
def _numba_accelerate_chi_square_all(b, l, num_classes):
    ret = np.empty(len(b))
    class_counts_a = np.zeros(num_classes)
    class_counts_b = np.zeros(num_classes)

    flip = True

    # Fill the first class_counts_a
    for j in range(b[0]):
        class_counts_a[l[j]] += 1

    for i in range(len(b)):
        start = 0 if i == 0 else b[i - 1]
        end = len(l) if i == len(b) - 1 else b[i + 1]

        # Get class counts for each
        for j in range(end - b[i]):
            if flip:
                class_counts_b[l[j + b[i]]] += 1
            else:
                class_counts_a[l[j + b[i]]] += 1

        # Get the total class counts
        class_counts_total_div_n = (class_counts_a + class_counts_b) / (end - start)

        chi_2 = 0.0

        # Work on sum for parta
        # Build the a's
        e = np.clip((b[i] - start) * class_counts_total_div_n, 0.1, np.inf)
        sub = (class_counts_a if flip else class_counts_b) - e
        chi_2 += np.sum(sub * sub / e)

        # Work on sum for partb
        e = np.clip((end - b[i]) * class_counts_total_div_n, 0.1, np.inf)
        sub = (class_counts_b if flip else class_counts_a) - e
        chi_2 += np.sum(sub * sub / e)

        # Add this new total sum into out list
        ret[i] = chi_2

        # If we were currently using class_counts_a as parta
        if flip:
            class_counts_a.fill(0)
        else:
            class_counts_b.fill(0)
        flip = not flip

    return ret
