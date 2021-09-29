"""
Unit tests for the mdfutils subpackage.
"""
import numpy as np
import pandas as pd
from moredataframes.mdfutils import to_numpy

try:
    import pytest
except ImportError:
    print("Pytest is not installed. Install to run tests using 'pip install pytest'.")
    exit(-1)


@pytest.mark.parametrize("array_like, expected", [
    ([1, 2, 3], np.array([[1, 2, 3]]).T),
    ([[1], [2], [3]], np.array([[1, 2, 3]]).T),
    ((1, 2, 3), np.array([[1, 2, 3]]).T),
    (((1,), (2,), (3,)), np.array([[1, 2, 3]]).T),
    ([7.4], np.array([[7.4]]).T),
    ([True, False, False], np.array([[True, False, False]]).T),
    ([i for i in range(10)], np.array([[i for i in range(10)]]).T),
    (range(10), np.array([[i for i in range(10)]]).T),
    (np.array([1, 2, 3]), np.array([1, 2, 3]).reshape([-1, 1])),
    (np.array([[]]), np.array([[]]).reshape([-1, 1])),
    (pd.DataFrame(), np.array([[]]).reshape([-1, 1])),
    ([], np.array([[]]).reshape([-1, 1])),
    (tuple(), np.array([[]]).reshape([-1, 1])),

    # Pandas defaults it to int64
    (pd.DataFrame({'a': [1, 2, 3], 'x': [4, 5, 6]}), np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int64)),
])
def test_to_numpy_valid(array_like, expected):
    """
    Tests if the given 1d or 2d ArrayLike object gets converted into the expected numpy array.
    :param array_like: Some ArrayLike to convert to numpy
    :param expected: A numpy ndarray of the expected output
    """
    output = to_numpy(array_like)
    np.testing.assert_array_equal(output, expected)
    assert type(output) == np.ndarray
    assert output.dtype == expected.dtype


_NUMPY_INVALID_OBJECT_ERROR = TypeError
_NUMPY_INVALID_DIMENION_ERROR = ValueError


@pytest.mark.parametrize("not_array_like, error", [
    (1, _NUMPY_INVALID_OBJECT_ERROR),
    (np.array(None), _NUMPY_INVALID_OBJECT_ERROR),
    (None, _NUMPY_INVALID_OBJECT_ERROR),
    ({'a': [5]}, _NUMPY_INVALID_OBJECT_ERROR),
    (np.array([[[]]]), _NUMPY_INVALID_DIMENION_ERROR),
    (np.array([[[1], [2]]]), _NUMPY_INVALID_DIMENION_ERROR),
    (np.array(range(16)).reshape([2, 2, 2, 2]), _NUMPY_INVALID_DIMENION_ERROR)
])
def test_to_numpy_error(not_array_like, error):
    """
    Tests if the given inputs raise the given error
    :param not_array_like: Some type that is not ArrayLike to fail in conversion
    :param error: The error to be raised
    """
    with pytest.raises(error):
        to_numpy(not_array_like)
