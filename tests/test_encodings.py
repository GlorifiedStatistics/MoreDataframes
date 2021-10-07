"""
Unit tests for the encoding_functions.py file.
"""
import numpy as np
import pytest
from moredataframes.encodings import noop, drop, factorize


def test_noop():
    """
    noop() Should only call to_numpy(), so just need to check to make sure things are passed correctly to get full
        coverage.
    """
    output, extra_info = noop(np.array([1, 2, 3]))
    np.testing.assert_equal(output, np.array([1, 2, 3]).reshape([-1, 1]))

    output, extra_info = noop(np.array([1, 2, 3]), inverse=True)
    np.testing.assert_equal(output, np.array([1, 2, 3]).reshape([-1, 1]))


def test_drop():
    """
    drop() should always return None, just need to check this always happens (so I don't change it later without
        having to think about it) and that parameters can be passed correctly.
    """
    output, extra = drop(np.array([1, 2, 3]))
    assert output is None

    output, extra = drop(np.array([1, 2, 3]), inverse=True)
    assert output is None


@pytest.mark.parametrize("v, expected", [
    ([1, 2, 3], [0, 1, 2]),
    ([], []),
    (['a', 'b', 'c', 'a', 'c', 'a'], [0, 1, 2, 0, 2, 0]),
    (np.array([['1', '2'], ['3', '4']]), [[0, 0], [1, 1]]),
])
def test_factorize(v, expected):
    output, extra_info = factorize(v)
    np.testing.assert_equal(output, np.array(expected).reshape(output.shape))

    output2, extra_info = factorize(v, encoding_info=extra_info)
    np.testing.assert_equal(output2, np.array(expected).reshape(output2.shape))

    orig, extra_info = factorize(output, encoding_info=extra_info, inverse=True)
    np.testing.assert_equal(orig, np.array(v).reshape(orig.shape))

    orig2, extra_info = factorize(output2, encoding_info=extra_info, inverse=True)
    np.testing.assert_equal(orig2, np.array(v).reshape(orig2.shape))


def test_extra_kwargs_supplied():
    """
    If you pass extra kwargs to an encoding function (needed because of the decorator)
    """
    with pytest.raises(TypeError):
        _, __ = factorize([1, 2, 3], fail_cause_bad_kwargs=True)
