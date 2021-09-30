"""
Utility functions for checking correctness of user-provided parameters.
"""
from moredataframes.mdfutils.typing import Sequence, Union, ExpectedType, EFuncInfo, Dict, Any, Tuple, \
    EncodingInfoExpectedType
from moredataframes.errors import MissingEncodingInfoError


def string_param(param: str, accept_list: Sequence[str], ignores: Union[str, Sequence[str], None] = '-_',
                 ignore_case: bool = True) -> bool:
    """
    Returns true if the given param is a string, and if that string appears in the given accept_list, false otherwise.
    :param param: the parameters to check
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


def check_encoding_info(d: EFuncInfo, tc: Sequence[EncodingInfoExpectedType]) -> None:
    """
    Raises an error if the given encoding_info dictionary does not contain the correct keys, or if the values are
        the wrong type.
    :param d: the encoding_info dictionary
    :param tc: an arbitrary number of 2-tuples of strings (the keys) and either types or
        sequences of types (the types). Ensures the dictionary passed to the binning function contains the given
        keys and each key has a value that is of the specified type (or one of any of the specified types if the
        types is a sequence in that tuple). The type can be object to allow any type, but just check for key.
        Types can also be None for ensuring or allowing NoneType. An empty tuple for type is considered the same
        as Any or type object.
    :raises:
        MissingEncodingInfoError: if one of the keys does not exist in the given encoding_info
        TypeError: if the value for a given key in the encoding_info is not of the correct type, or if d is not
            a dictionary
    """

    # Check the encoding_info is a dictionary
    if not isinstance(d, dict):
        raise TypeError("encoding_info should be a dictionary, instead is: %s" % type(d))

    # Check each key for existance, and for type
    for key, _type in tc:

        # Check for key
        if key not in d:
            raise MissingEncodingInfoError(key)

        if not _check_type(_type, d[key]):
            raise TypeError("Expected key '%s' to have one of types %s in encoding_info, instead is: %s"
                            % (key, _type, type(d[key])))


def check_kwargs_types(kwargs_types: Dict[str, ExpectedType], **kwargs: Any):
    """
    Raises an error if the args or kwargs types given do not match expected
    :param kwargs_types: checks that the given kwargs have the given types. Each key should be a parameter kwarg that
        could be passed to the function, and value should be either a type or sequence of types where the value could
        be any one of the given types. If the type is object, then Any type will be allowed
    :param kwargs: the kwargs
    :raises:
        TypeError: if the types do not match
    """

    # Check the kwargs, assume all the kwargs exist
    for k, et in kwargs_types.items():
        if not _check_type(et, kwargs[k]):
            raise TypeError("Expected kwarg '%s' to have type %s, instead is: %s" % (k, et, type(kwargs[k])))


def _check_type(_type: ExpectedType, v: Any) -> bool:
    """
    Helper to check type
    :param _type: the type
    :param v: the value
    :return: bool
    """
    # Make _type a list no matter what it was
    if isinstance(_type, type) or _type is None:
        _type = (_type,)

    # Allow _type to be object to allow Any, or None for None
    if object in _type or (None in _type and v is None) or len(_type) == 0:
        return True

    # Remove None from _type for isinstance check
    _type = tuple([t for t in _type if t is not None])
    return len(_type) == 0 or not isinstance(v, _type)
