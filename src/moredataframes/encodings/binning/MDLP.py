"""
Code modified from: https://github.com/tmadl/sklearn-expertsys/blob/master/Discretization
"""
import numpy as np
from moredataframes.mdfutils.typing import ArrayLike, NDArray, Any, Tuple, Union, List, Optional, TypeVar
from moredataframes.mdfutils import to_numpy


_T = TypeVar('_T')
_U = TypeVar('_U')
_LOG2of3 = np.log2(3)


def mdlp_bin(x: ArrayLike, y: ArrayLike) -> NDArray[Any]:
    """
    Does MDLP discretization on data x given classes y.
    """
    x, y = to_numpy(x), to_numpy(y)

    def _sort(_x: NDArray[_T], _y: NDArray[_U]) -> Tuple[NDArray[_T], NDArray[_U]]:
        args = np.argsort(_x)
        return _x[args], _y[args]

    all_cuts = [list(sorted(_column_cutpoints(*_sort(x[:, i], y)))) for i in range(x.shape[1])]
    # return apply_cutpoints(x, y, all_cuts)
    return np.array(all_cuts)


"""
def apply_cutpoints(x, y, all_cuts):
    '''
    Discretizes data by applying bins according to self._cuts.
    '''
    bin_label_collection = []
    for cuts in all_cuts:
        if len(cuts) == 0:
            self._data[attr] = 'All'
            bin_label_collection.append(['All'])
        else:
            cuts = [-np.inf] + cuts + [np.inf]
            start_bin_indices = range(0, len(cuts) - 1)
            bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i + 1])) for i in start_bin_indices]
            bin_label_collection.append(bin_labels)
            self._data[attr] = pd.cut(x=self._data[attr].values, bins=cuts, right=False, labels=bin_labels,
                                      precision=6, include_lowest=True)
"""


def _column_cutpoints(x: NDArray[Any], y: NDArray[Any], start: Optional[Union[int, None]] = None,
                      end: Optional[Union[int, None]] = None) -> List[int]:
    '''
    Computes the cuts for binning a feature according to the MDLP criterion.
    Assumes x and y are already sorted.
    :param start: the starting index in x to consider for this partition, else None to use all of x
    :param end: the ending index in x to consider for this partition, else None to use all of x
    '''
    start, end = 0 if start is None else start, len(x) if end is None else end

    # determine whether to cut and where
    args = np.where(y[start:end][:-1] != y[start:end][1:], 1, 0).astype(bool)
    candidates = (x[start:end][:-1][args] + x[start:end][1:][args]) / 2
    if len(candidates) == 0:
        return []

    cut_candidates = [(cut, _cut_point_information_gain(x[start:end], y[start:end], cut)) for cut in candidates]
    cut_idx = max(cut_candidates, key=lambda v: v[1])[0]
    decision = _mdlpc_criterion(x[start:end], y[start:end], cut_idx)

    cuts = []
    if decision and cut_idx > 0 and cut_idx < end - start - 1:
        cuts += _column_cutpoints(x, y, start=start, end=cut_idx) \
                + _column_cutpoints(x, y, start=cut_idx, end=end) \
                + [cut_idx]

    return cuts


def _mdlpc_criterion(x: NDArray[Any], y: NDArray[Any], cut_point: int) -> bool:
    '''
    Determines whether a partition is accepted according to the MDLPC criterion. Assumes x and y are already sorted.
    '''
    idx = int(np.searchsorted(x, cut_point, side='right'))
    cut_point_gain = _cut_point_information_gain(x, y, cut_point)

    def _ent(a: NDArray[Any]) -> float:
        return len(np.unique(a)) * _entropy(a)

    # I chose 50 here because it's a nice round number where the difference between the real and approximate functions
    #   is small (within floating point rounding error) and the difference in time is negligable
    logval = np.log2(3 ** len(y) - 2) if len(y) > 50 else (_LOG2of3 * len(y))
    delta = logval - _ent(y) + _ent(y[:idx]) + _ent(y[idx:])

    # to split or not to split
    return bool(cut_point_gain > (np.log2(len(x) - 1) + delta) / len(x))


def _entropy(y: NDArray[Any]) -> float:
    '''
    Computes the entropy of a set of classes
    '''
    def _nlog(x: float) -> float:
        return float(-x * np.log2(x))
    return sum([_nlog(len(y[y == c]) / len(y)) for c in np.unique(y)])


def _cut_point_information_gain(x: NDArray[Any], y: NDArray[Any], cut_point: int) -> float:
    '''
    Returns the information gain obtained by splittinga numeric attribute in two according to cut_point.
    Assumes x and y to be already sorted.
    '''
    idx = int(np.searchsorted(x, cut_point, side='right'))
    return _entropy(y) - (idx / len(x)) * _entropy(x[:idx]) - ((len(x) - idx) / len(x)) * _entropy(x[idx:])
