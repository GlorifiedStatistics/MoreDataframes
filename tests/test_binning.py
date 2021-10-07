"""
Unit tests for the encodings.binning package
"""
import pytest
import numpy as np
from moredataframes.encodings.binning import fixed_length, digitize, chi_merge
from moredataframes.constants import BIN_RIGHT_EDGE_OFFSET
from moredataframes.errors import UserFunctionCallError


@pytest.mark.parametrize('v, num_bins, expected, bins', [
    ([], 1, [], None),
    ([], 10, [], None),
    ([1], 1, [0], [[1, 1 + BIN_RIGHT_EDGE_OFFSET]]),
    (range(10), 9, list(range(9)) + [8], [list(range(9)) + [9.0 + BIN_RIGHT_EDGE_OFFSET]])
])
def test_fixed_length(v, num_bins, expected, bins):
    output, extra_info = fixed_length(v, num_bins=num_bins)
    np.testing.assert_array_equal(output, np.array(expected).reshape(output.shape))
    if bins is not None:
        np.testing.assert_array_equal(extra_info['bins'], bins)

    output2, extra_info = fixed_length(v, num_bins=num_bins)
    np.testing.assert_array_equal(output2, np.array(expected).reshape(output2.shape))
    if bins is not None:
        np.testing.assert_array_equal(extra_info['bins'], bins)


def test_fixed_length_failures():
    # Raise error on num_bins <= 0
    with pytest.raises(ValueError):
        _, __ = fixed_length([1], num_bins=0)
    with pytest.raises(ValueError):
        _, __ = fixed_length([1], num_bins=-1)

    # Raise error on num_bins != 1 and len(vals) == 1
    with pytest.raises(ValueError):
        _, __ = fixed_length([1], num_bins=2)


@pytest.mark.parametrize('vals, bins, edge, expected, new_bins', [
    ([], [1, 2], 'warn', [], None),
    ([], [1, 2], 'extend', [], None),
    ([], [1, 2], 'closest', [], None),
    ([0, 1, 2, 3, 4], [0, 1, 3, 5], 'closest', [0, 1, 1, 2, 2], None),
    ([0, 1, 2.3, 3, 4], [0, 1.1, 3, 5], 'extend', [0, 0, 1, 2, 2], None),
    ([-1, 0, 1, 2, 3, 4, 5, 7, 5.00000001], [0, 1, 3, 5], 'closest', [0, 0, 1, 1, 2, 2, 2, 2, 2], None),
    ([-1, 0, 1, 2, 3, 4, 5 + BIN_RIGHT_EDGE_OFFSET, 7, 5 + BIN_RIGHT_EDGE_OFFSET / 2],
        [0, 1, 3, 5 + BIN_RIGHT_EDGE_OFFSET], 'extend', [0, 1, 2, 2, 3, 3, 4, 4, 3],
        [-1, 0, 1, 3, 5 + BIN_RIGHT_EDGE_OFFSET, 7 + BIN_RIGHT_EDGE_OFFSET]),
])
def test_digitize(vals, bins, edge, expected, new_bins):
    bins = np.array(bins)
    if bins.ndim == 1:
        bins = bins.reshape([1, -1])
    new_bins = bins if new_bins is None else np.array(new_bins).reshape([1, -1])

    enc_info = {'bins': bins}
    output, enc_info = digitize(vals, enc_info, edge_behavior=edge)
    np.testing.assert_array_equal(output, np.array(expected).reshape(output.shape))
    np.testing.assert_array_equal(enc_info['bins'], new_bins)

    output2, enc_info = digitize(vals, enc_info, edge_behavior=edge)
    np.testing.assert_array_equal(output2, np.array(expected).reshape(output.shape))
    np.testing.assert_array_equal(enc_info['bins'], new_bins)


def test_digitize_failure():
    with pytest.raises(TypeError):
        _, __ = digitize([1, 2])
    with pytest.raises(TypeError):
        _, __ = digitize([1, 2], encoding_info={'bins': None})

    encoding_info = {'bins': [[0, 1, 2, 3]], 'edge_behavior': 'warn'}

    with pytest.raises(ValueError):
        _, __ = digitize([-1, 0, 1, 2], encoding_info=encoding_info, edge_behavior='warn')
    with pytest.raises(ValueError):
        _, __ = digitize([0, 1, 2, 5], encoding_info=encoding_info)


def test_decode():
    vals = []
    output, encoding_info = fixed_length(vals)
    dec, encoding_info = fixed_length(output, encoding_info=encoding_info, inverse=True)
    assert dec.size == 0

    vals = [1, 2, 3, 4]
    output, encoding_info = fixed_length(vals, num_bins=2)
    dec, encoding_info = fixed_length(output, encoding_info=encoding_info, inverse=True)
    a, b = '[%f, %f)' % (1, 2.5), '[%f, %f)' % (2.5, 4 + BIN_RIGHT_EDGE_OFFSET)
    np.testing.assert_array_equal(dec, np.array([[a], [a], [b], [b]]))
    np.testing.assert_array_equal(encoding_info['bins'], [[1, 2.5, 4 + BIN_RIGHT_EDGE_OFFSET]])

    dec, encoding_info = fixed_length(output, encoding_info=encoding_info, inverse=True, decode_method='left')
    np.testing.assert_array_equal(dec, np.array([[1], [1], [2.5], [2.5]]))
    np.testing.assert_array_equal(encoding_info['bins'], [[1, 2.5, 4 + BIN_RIGHT_EDGE_OFFSET]])

    dec, encoding_info = fixed_length(output, encoding_info=encoding_info, inverse=True, decode_method='right')
    np.testing.assert_array_equal(dec, np.array([[2.5], [2.5], [4 + BIN_RIGHT_EDGE_OFFSET], [4 + BIN_RIGHT_EDGE_OFFSET]]))
    np.testing.assert_array_equal(encoding_info['bins'], [[1, 2.5, 4 + BIN_RIGHT_EDGE_OFFSET]])

    dec, encoding_info = fixed_length(output, encoding_info=encoding_info, inverse=True, decode_method='mid')
    a, b = (1 + 2.5) / 2, (2.5 + 4 + BIN_RIGHT_EDGE_OFFSET) / 2
    np.testing.assert_array_equal(dec, np.array([[a], [a], [b], [b]]))
    np.testing.assert_array_equal(encoding_info['bins'], [[1, 2.5, 4 + BIN_RIGHT_EDGE_OFFSET]])

    f = lambda low, high: low + 1
    dec, encoding_info = fixed_length(output, encoding_info=encoding_info, inverse=True, decode_method=f)
    np.testing.assert_array_equal(dec, np.array([[2], [2], [3.5], [3.5]]))
    np.testing.assert_array_equal(encoding_info['bins'], [[1, 2.5, 4 + BIN_RIGHT_EDGE_OFFSET]])


def test_decode_failure():
    with pytest.raises(UserFunctionCallError):
        vals = [1, 2, 3, 4]
        f = lambda low: 3
        output, encoding_info = fixed_length(vals, num_bins=2)
        _, __ = fixed_length(vals, encoding_info=encoding_info, decode_method=f, inverse=True)

    with pytest.raises(UserFunctionCallError):
        vals = [1, 2, 3, 4]
        f = lambda low, high: 3
        output, encoding_info = fixed_length(vals, num_bins=2)
        _, __ = fixed_length(vals, encoding_info=encoding_info, decode_method=f, inverse=True)

    with pytest.raises(UserFunctionCallError):
        vals = [1, 2, 3, 4]
        f = lambda low, high: low[:-1]
        output, encoding_info = fixed_length(vals, num_bins=2)
        _, __ = fixed_length(vals, encoding_info=encoding_info, decode_method=f, inverse=True)

    with pytest.raises(UserFunctionCallError):
        vals = [1, 2, 3, 4]
        f = lambda low, high: low[0][:-1]
        output, encoding_info = fixed_length(vals, num_bins=2)
        _, __ = fixed_length(vals, encoding_info=encoding_info, decode_method=f, inverse=True)


def chim():
    vals = [1, 2, 3, 4, 5, 6]
    labels = [0, 1, 0, 1, 0, 1]
    output, encoding_info = chi_merge(vals, labels=labels, max_bins=3)

    size = 100_0000
    vals = np.random.randint(0, 10000, size=[size])
    labels = np.where(vals < 1_000, 0, 1)

    from timeit import default_timer
    t = default_timer()
    output, encoding_info = chi_merge(vals, labels=labels)
    print(default_timer() - t)
    print(encoding_info['bins'])


if __name__ == "__main__":
    #print("No current test")
    chim()
