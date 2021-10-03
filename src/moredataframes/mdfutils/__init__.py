from .conversion import to_numpy
from .param_utils import string_param, check_encoding_info, check_kwargs_types


# Build the numba jit function decorator
try:
    import numba
    NUMBA_INSTALLED = True


    def check_for_numba(*args, **kwargs):
        if 'njit' in kwargs and kwargs['njit']:
            kwargs['nopython'] = True

        if 'test_no_numba' in kwargs and kwargs['test_no_numba']:
            return lambda x: x

        def _ret(func):
            return numba.jit(func, *args, **kwargs)

        return _ret
except ImportError:
    NUMBA_INSTALLED = False


    def check_for_numba(*args, **kwargs):
        return lambda x: x

__all__ = ['to_numpy', 'string_param', 'check_encoding_info', 'check_kwargs_types', 'check_for_numba',
           'NUMBA_INSTALLED']
