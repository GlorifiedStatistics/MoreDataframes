"""
Unit tests for the encoding_functions.py file.
"""
import numpy as np
from moredataframes.encodings.encoding_functions import noop, drop


def test_noop():
    """
    noop() Should only call to_numpy(), so just need to check to make sure things are passed correctly to get full
        coverage.
    """
    extra_info = {}
    output = noop(np.array([1, 2, 3]), extra_info, inverse=False)
    np.testing.assert_equal(output, np.array([1, 2, 3]).reshape([-1, 1]))
    assert len(list(extra_info.keys())) == 0

    output = noop(np.array([1, 2, 3]), extra_info, inverse=True)
    np.testing.assert_equal(output, np.array([1, 2, 3]).reshape([-1, 1]))
    assert len(list(extra_info.keys())) == 0


def test_drop():
    """
    drop() should always return None, just need to check this always happens (so I don't change it later without
        having to think about it) and that parameters can be passed correctly.
    """
    extra_info = {}
    output = drop(np.array([1, 2, 3]), extra_info, inverse=False)
    assert len(list(extra_info.keys())) == 0 and output is None

    output = drop(np.array([1, 2, 3]), extra_info, inverse=False)
    assert len(list(extra_info.keys())) == 0 and output is None
