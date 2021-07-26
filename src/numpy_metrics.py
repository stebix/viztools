import numpy as np

from typing import Union, Dict, List, Tuple


def void(arg, *args, **kwargs):
    return arg


def assert_idshape(array_a: np.ndarray, array_b: np.ndarray) -> None:
    # assert identical shape with nice message :)
    msg = (f'Expecting identical shape, but got shape {array_a.shape} '
           f'and {array_b.shape}!')
    assert array_a.shape == array_b.shape, msg
    return None


def is_binary(array: np.ndarray) -> bool:
    # check if entries are all 0 or 1
    boolmap = np.logical_or(
        np.isclose(array, np.ones_like(array)),
        np.isclose(array, np.zeros_like(array))
    )
    return boolmap.all()


def canonical_reduction_dims(array: np.ndarray) -> Tuple:
    # get canonical reduction dims for 4D/5D tensors of
    # specific (N x C x [... spatial ...]) layout
    if array.ndim == 4:
        return (0, 2, 3)
    elif array.ndim == 5:
        return (0, 2, 3, 4)
    else:
        raise ValueError((f'Non-canonical layout: {array.shape}! '
                          f'Expecting 4D (N x C x H x W) or 5D '
                          f'((N x C x D x H x W)) tensors'))



class _BinaryClassificationMetric:
    """
    Base class for binary classification metric with NumPy backend.
    """
    reductions = {
        'mean' : np.mean,
        'sum' : np.sum,
        'none' : void,
        None : void
    }
    # long string name
    lname = None

    def __init__(self, reduction: str = 'none',
                 square: bool = False,
                 checkargs: bool = True) -> None:

        if not reduction in self.reductions.keys():
            msg = (f'Unsupported reduction argument: {reduction},'
                   f'reduction argument must one of {self.valid_reductions}')
            raise ValueError(msg)

        self.reduction = reduction
        self._reduction_op = self.reductions[reduction]
        self.checkargs = checkargs
        self.square = square
    

    def __call__(self, *args):
        raise NotImplementedError


    def _preprocess(self, inpt: np.ndarray, trgt: np.ndarray) -> None:
        """
        Preprocesses the numpy.ndarray pair provided in the __call__ method
        via shape and value checking.
        Also computes the reduction dims if a reduction operation is utilized.

        May raise ValueError or AssertionError
        """
        # check for shape congruence
        assert_idshape(inpt, trgt)
        # enforce binarity i.e. tensor values must be either 0 or 1
        if self.checkargs:
            for i, array in enumerate((inpt, trgt)):
                assert is_binary(array), f'Array at posarg {i} is not binary!'
        
        if self.reduction not in ['none', None]:
            red_dims = canonical_reduction_dims(inpt)
        else:
            red_dims = None
        


class TP(_BinaryClassificationMetric):
    """
    Compute true positive classification metric between input
    `inpt` and target `trgt`. 
    """
    lname = 'true_positive'

    def __call__(self, inpt: np.ndarray, trgt: np.ndarray) -> np.ndarray:
        red_dims = self._preprocess(inpt, trgt)
        tp = inpt * trgt

        if self.square:
            tp = np.square(tp)

        return self._reduction_op(tp, axis=red_dims, keepdims=False)


class FP(_BinaryClassificationMetric):
    """
    Compute true positive classification metric between input
    `inpt` and target `trgt`. 
    """
    lname = 'false_positive'

    def __call__(self, inpt: np.ndarray, trgt: np.ndarray) -> np.ndarray:
        red_dims = self._preprocess(inpt, trgt)
        fp = inpt * (1 - trgt)

        if self.square:
            fp = np.square(fp)

        return self._reduction_op(fp, axis=red_dims, keepdims=False)



class FN(_BinaryClassificationMetric):
    """
    Compute false negative classification metric between input
    `inpt` and target `trgt`. 
    """
    lname = 'false_negative'


    def __call__(self, inpt: np.ndarray, trgt: np.ndarray) -> np.ndarray:
        red_dims = self._preprocess(inpt, trgt)
        fn = (1 - inpt) * trgt

        if self.square:
            fn = np.square(fn)

        return self._reduction_op(fn, axis=red_dims, keepdims=False)



class TN(_BinaryClassificationMetric):
    """
    Compute true negative classification metric between input
    `inpt` and target `trgt`. 
    """
    lname = 'true_negative'

    def __call__(self, inpt: np.ndarray, trgt: np.ndarray) -> np.ndarray:
        red_dims = self._preprocess(inpt, trgt)
        tn = (1 - inpt) * (1 - trgt)

        if self.square:
            tn = np.square(tn)

        return self._reduction_op(tn, axis=red_dims, keepdims=False)
