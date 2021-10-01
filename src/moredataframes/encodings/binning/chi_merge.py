"""
Implements the ChiMerge algorithm.

Resources:
https://medium.com/@nithin_rajan/data-discretization-using-chimerge-55c8ade3cfda

Paper:
https://dl.acm.org/doi/abs/10.5555/1867135.1867154
"""
from moredataframes.mdfutils.typing import ArrayLike, EFuncInfo, Optional, Union, NDArray, Any, List
from moredataframes.mdfutils import to_numpy
from moredataframes.encodings.binning import binning_function
from moredataframes.constants import ENCODING_INFO_BINS_KEY
import numpy as np


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
        as have the ability to determine uniqueness with np.unique()
    :param max_bins: if not None, then an int >= 2 specifying the maximum number of bins to use. Will continue the
        merging algorithm until there are <= this number of bins, even if the threshold value has been passed.
    :param min_bins: if not None, then an int >= 2 specifying the minimum number of bins to use. Will stop the merging
        algorithm if the next merge would cause the number of bins to fall below this value, even if the current
        lowest chi-square p-value is less than the threshold.
    :param threshold: a float in range (0, 1] specifying the p-value of the chi-square test at which, if the current
        lowest chi-square p-value is greater than this threshold, the merging algorithm is halted. This is overridden
        if min_bins or max_bins is not None.
    :return: a numpy array
    """
    vals, labels = to_numpy(vals), to_numpy(labels)

    # Check to make sure the lengths are the same
    if len(vals) != len(labels):
        raise ValueError("The labels should be the same length as the values to bin. Instead values length is: "
                         "%d while labels length is: %d" % (len(vals), len(labels)))

    ret = []

    # Go through each column, applying the chimerge algorithm
    for col in vals.T:

        # Sort the col and find all the boundary points, while making sure to copy the labels
        arg_sort = np.argsort(col)
        col = col[arg_sort]
        labels_sorted = labels.copy()[arg_sort]

        # Left such that if boundary_indices[i] = x, then labels[x - 1] != labels[x], but labels[x - 1] may be equal
        #    to labels[x - 2]. IE: the i-th partition is the labels range [boundary_indices[i - 1], boundary_indices[i])
        boundary_indices = np.argwhere(labels[:-1] != labels[1:]).reshape(-1) + 1

        # Compute all the chi-square values
        classes = np.unique(labels)
        chi_squares = _chi_square_all(boundary_indices, labels, classes)

    # Encode each column
    return np.array([10])


def _chi_square_all(boundaries: NDArray[Any], labels: NDArray[Any], num_classes: Sequence[Any]) -> List[float]:
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

