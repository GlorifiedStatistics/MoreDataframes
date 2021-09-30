"""
Classes for different common errors that are not covered by basic TypeError, ValueError, etc.
"""


class UserFunctionCallError(Exception):
    """
    An unknown error occurring from a call to a user-supplied function.
    """

    def __init__(self, string: str = ''):
        super().__init__(string)


class MissingEncodingInfoError(Exception):
    """
    An error occurring when a decoding function expects a key to exist in its encoding_info, but that key was not found.
    """

    def __init__(self, key: str):
        super().__init__("Encoding_info expected to have key '%s', but key was not found." % key)