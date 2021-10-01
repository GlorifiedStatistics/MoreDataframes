"""
Speed tests for the binning algorithms.
"""
import numpy as np
from timeit import default_timer
import pandas as pd
import numba

@numba.njit()
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

    def numba_python(boundaries, labels, classes):
        # Enforce that classes is integers
        labels, label_labels = pd.factorize(labels) if not np.issubdtype(labels.dtype, np.integer) or \
                                         len(classes) != np.max(labels) - 1 else (classes, None)

        return _numba_accelerate_chi_square_all(boundaries, labels, len(classes))

    _labels_test = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    _labels_test_expected = np.array([2 for i in range(len(_labels_test) - 1)])

    np.random.seed(123456)

    _labels_list = [
        _labels_test,
        np.array([0, 0, 0, 1, 1, 2]),
        np.random.randint(0, 100, size=1_000),
        np.random.randint(0, 100, size=10_000),
    ]
    _labels_names = [
        'test_labels',
        'small_manual',
        'small_random_labels',
        'medium_random_labels',
    ]

    _func_list = [
        slow_python,
        numba_python,
    ]

    def _get_boundaries_and_classes(_l):
        _b = np.argwhere(_l[:-1] != _l[1:]).reshape(-1) + 1
        _c = np.unique(_l)
        return _b, _c

    func_times = []

    outputs = []

    for func in _func_list:
        # Check that the algorithm might work (NOTE: this is not a full testing my any means), as well as call the
        #   function once to make sure numba can compile to not mess with timings
        _boundaries, _classes = _get_boundaries_and_classes(_labels_test)
        np.testing.assert_array_almost_equal(func(_boundaries, _labels_test, _classes), _labels_test_expected)

        times = []
        these_outputs = []

        for _labels in _labels_list:
            _boundaries, _classes = _get_boundaries_and_classes(_labels)

            t = default_timer()
            these_outputs.append(func(_boundaries, _labels, _classes))
            times.append(default_timer() - t)

        func_times.append(times)
        outputs.append(these_outputs)

    for i in range(1, len(outputs)):
        for j in range(len(outputs[0])):
            np.testing.assert_array_almost_equal(outputs[i][j], outputs[i - 1][j])

    print("Timings:\n")
    print(pd.DataFrame(func_times, columns=_labels_names, index=[f.__name__ for f in _func_list]).to_markdown(floatfmt='0.4f'))


if __name__ == '__main__':
    speedtest_chi_square_all()
