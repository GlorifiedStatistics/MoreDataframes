from typing import Union, Callable, Any, Dict, List, TYPE_CHECKING, Tuple, Optional, TypeVar, Sequence
from typing_extensions import Protocol

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    EncodingFunction = Callable[[ArrayLike, Dict[str, Any], bool], ArrayLike]
    EFuncInfo = Dict[Union[str, int], Any]

    EncodingDict = Dict[Union[str, int], EncodingFunction]
    ExpectedType = Union[type, Sequence[type]]
    EncodingInfoExpectedType = Tuple[str, ExpectedType]

else:
    ArrayLike = List
    NDArray = List
    EncodingFunction = Any
    EFuncInfo = Any
    EncodingDict = Any
    DecodeBinArgs = Any
    ExpectedType = Any
    EncodingInfoExpectedType = Any

__all__ = ['ArrayLike', 'NDArray', 'Union', 'Callable', 'Any', 'Dict', 'EncodingFunction', 'EFuncInfo', 'EncodingDict',
           'List', 'Tuple', 'Optional', 'TypeVar', 'Sequence', 'ExpectedType', 'EncodingInfoExpectedType']
