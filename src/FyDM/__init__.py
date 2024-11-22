"""Created on Feb 21 22:49:36 2024"""

from typing import Callable, Optional, Union

from numpy.typing import NDArray

TOLERANCE = 1e-8
N_DECIMAL = 6
MAX_ITER = 500

DOC_STYLE = 'numpy_napoleon_with_merge'

Func = Callable
NdArray = NDArray
IFloat = Union[float, int]

FList = list[IFloat]

LList = list[FList]
FTuple = tuple[IFloat, ...]
LLList = list[LList]
LFunc = list[Func]

FListOrLList = Union[FList, LList]
IFloatOrFList = Union[IFloat, FList]

OptFunc = Optional[Func]
OptList = Optional[FList]
OptIFloat = Optional[IFloat]
