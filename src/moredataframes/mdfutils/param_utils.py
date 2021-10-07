"""
Utility functions for checking correctness of user-provided parameters.
"""
from moredataframes.mdfutils.typing import Sequence, Union, EFuncInfo, Dict, Any
from moredataframes.errors import MissingEncodingInfoError


def string_param(param: Union[str, Any], accept_list: Sequence[str], ignores: Union[str, Sequence[str], None] = '-_',
                 ignore_case: bool = True) -> bool:
    """
    Returns true if the given param is a string, and if that string appears in the given accept_list, false otherwise.
    :param param: the parameters to check (should be a string)
    :param accept_list: the list of strings to accept
    :param ignores: characters that can be ignored. For example,
        string_param('par_am', ['param', 'param2'], ignores=None) would return False while
        string_param('par_am', ['param', 'param2'], ignores='_') would return True
    :param ignore_case: if True, ignores case on both the param and accept_list
    :return: bool
    """

    # Check that param is a str
    if not isinstance(param, str):
        return False

    # Convert ignores to a string
    if not isinstance(ignores, str):
        ignores = '' if ignores is None else '|'.join([i for i in ignores])

    # Clean up accept_list and param if need be
    if ignore_case:
        param = param.lower()
        accept_list = [a.lower() for a in accept_list]
    accept_list = [a.replace(ignores, '') for a in accept_list]
    param = param.replace(ignores, '')

    # Now check if our param is in this new accept_list
    return param in accept_list
