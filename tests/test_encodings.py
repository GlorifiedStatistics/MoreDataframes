"""
Unit tests for the encodings.py file.
"""
import numpy as np
# import pandas as pd
import pytest
from moredataframes.encodings import to_numpy


@pytest.mark.parametrize("arraylike,expected", [
    ([1, 2, 3], np.array([[1, 2, 3]]).T),
])
def test_to_numpy(arraylike, expected):
    """
    Tests if the given 1d or 2d ArrayLike object gets converted into the expected numpy array.
    :param arraylike: Some ArrayLike to convert to numpy
    :param expected: A numpy ndarray of the expected output
    """
    output = to_numpy(arraylike)
    np.testing.assert_array_equal(output, expected)
    assert type(output) == np.ndarray
