from typing import Union, Callable, Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    EncodingFunction = Callable[[ArrayLike, Dict[str, Any], bool], ArrayLike]
    EFuncInfo = Dict[Union[str, int], Any]

    EncodingDict = Dict[Union[str, int], EncodingFunction]
else:
    ArrayLike = List
    NDArray = List
    EncodingFunction = Any
    EFuncInfo = Any
    EncodingDict = Any

__all__ = ['ArrayLike', 'NDArray', 'Union', 'Callable', 'Any', 'Dict', 'EncodingFunction', 'EFuncInfo', 'EncodingDict',
           'List']
