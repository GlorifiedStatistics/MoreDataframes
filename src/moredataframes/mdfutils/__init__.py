from .conversion import to_numpy
from .param_utils import string_param


NUMBA_INSTALLED = False
NUMBA_WARNED = False


def _failed_numba(func):
    """
    If numba is not installed, then this shows a warning on first call to a numba function telling the user that
        there may be significant speed improvements if numba is installed.
    """
    global NUMBA_WARNED
    if not NUMBA_WARNED:
        NUMBA_WARNED = True
        print("Warning: Numba is not installed and a numba-accelerated function is called. You may see significant"
              " speed improvements if you install numba. Try out 'pip install numba'.")
    return func


# Build the numba jit function decorator
try:
    import numba


    def check_for_numba(*args, **kwargs):
        if 'njit' in kwargs and kwargs['njit']:
            kwargs['nopython'] = True

        if 'test_no_numba' in kwargs and kwargs['test_no_numba']:
            return _failed_numba

        NUMBA_INSTALLED = True
        def _ret(func):
            return numba.jit(func, *args, **kwargs)

        return _ret
except ImportError:
    def check_for_numba(*args, **kwargs):
        return _failed_numba

__all__ = ['to_numpy', 'string_param', 'check_for_numba', 'NUMBA_INSTALLED']
