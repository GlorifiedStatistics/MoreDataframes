"""
Speed tests for the binning algorithms.
"""
import numpy as np
import pandas as pd
from moredataframes.mdfutils import check_for_numba
from speedtester import speedtest


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


def speedtest_chi_square_all():
    """
    Implementations of the _chi_square_all() algorithm used in the chi_merge() function.

    Function Description (may be old):

    '''
    Computes the chi-square values between each partition and returns them as a list.
    This function is left separate than _chi_square() to speed up computation using numpy's vectorized methods.
    :param boundaries: the indicies describing the boundaries of a partition, such that if boundaries[i] = x, then
        labels[x - 1] is in a different partition than labels[x], however it may be in the same partition as
        labels[x - 2]
    :param labels: the class labels
    :param classes: the list of unique classes in the entire column
    :return: a list chi-squares such that chi-squares[i] is the chi-square value between the 0-th and 1-st partition
        of labels
    '''


    Chi-square calculation should be:

    Sum is computed as:
        sum(sum( (A_ij - E_ij)^2 / E_ij for j in range(k)) for i in range(2))

        Where:
            - k is the number of classes
            - A_ij is the number of samples for the i-th interval with the j-th class label
            - E_ij is the expected frequency of A_ij
                * Defined as R_i * (C_j / N) where R_i is the number of samples in the i-th interval, C_j is
                    the total number of samples for class j over the two bins, and N is the total number of
                    samples in the two bins
                * If E_ij is 0, then the division fails. So, we instead set E_ij to an arbitrary small value 0.1
    """

    def slow_python(boundaries, labels, classes):
        ret = []

        for i in range(len(boundaries)):
            parta = labels[0 if i == 0 else boundaries[i - 1]: boundaries[i]]
            partb = labels[boundaries[i]: len(labels) if i == len(boundaries) - 1 else boundaries[i + 1]]
            both_parts = labels[0 if i == 0 else boundaries[i - 1]: len(labels) if i == len(boundaries) - 1 else boundaries[i + 1]]

            n = len(parta) + len(partb)
            cj_dict = {k: v for k, v in zip(*np.unique(both_parts, return_counts=True))}
            cj_dict.update({k: 0 for k in [c for c in classes if c not in cj_dict]})

            s = 0
            for part in [parta, partb]:
                for c in classes:
                    a = len(part[part == c])
                    e = max(0.1, len(part) * (cj_dict[c] / n))
                    s += (a - e)**2 / e

            ret.append(s)

        return ret

    numba_accelerated = check_for_numba()(_numba_accelerate_chi_square_all)

    def numba_python(boundaries, labels, classes):
        # Enforce that classes is integers
        labels, label_labels = pd.factorize(labels) if not np.issubdtype(labels.dtype, np.integer) or \
                                         len(classes) != np.max(labels) - 1 else (classes, None)

        return numba_accelerated(boundaries, labels, len(classes))

    failed_numba = check_for_numba(test_no_numba=True)(_numba_accelerate_chi_square_all)

    def numba_failed(boundaries, labels, classes):
        # Enforce that classes is integers
        labels, label_labels = pd.factorize(labels) if not np.issubdtype(labels.dtype, np.integer) or \
                                         len(classes) != np.max(labels) - 1 else (classes, None)

        return failed_numba(boundaries, labels, len(classes))

    np.random.seed(123456)

    test_vals = [
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        np.array([0, 0, 0, 1, 1, 2]),
        np.random.randint(0, 100, size=1_000),
        np.random.randint(0, 100, size=10_000),
        np.array([3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 1, 1, 1]),
        np.repeat(np.random.choice(range(100), 100, replace=False), np.random.randint(1, 100, size=100))
    ]
    speed_input_labels = [
        'test_labels',
        'small_manual',
        'small_random_labels',
        'medium_random_labels',
        'small_long_string',
        'medium_long_string',
    ]

    funcs = [
        slow_python,
        numba_python,
        numba_failed,
    ]
    
    speed_inputs = [[(np.argwhere(s[:-1] != s[1:]).reshape(-1) + 1, s, np.unique(s)), {}] for s in test_vals]

    speedtest(speed_inputs, speed_input_labels, funcs)

