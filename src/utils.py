"""
Home for various utils that live here until this gets cleaned up into multiple separate
sub-modules.

The code is mainly concernened with providing eas-of-use for working with
predictions on datasets. 
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py

import nrrd

from typing import List, Tuple, Union, Dict, Iterable, NewType, Optional, Callable

import src.numpy_metrics as npm


PathLike = NewType('PathLike', Union[str, pathlib.Path])


def relabel(array, src_label, trgt_label):
    assert array.dtype in (np.int, np.int64, np.int32, np.uint, np.uint8)
    relabeled_array = array[array == src_label] = trgt_label
    return relabeled_array


def merge_foreground(label_array):
    """
    Merge the foreground semantic classes into one label: [1, 2, ..., N] -> [1].
    The label_array is assumed to be integer-like, where the background is
    assumed to have the 0 label and the foreground semantic classes to
    have i = 1, 2, 3, ... as label.
    """
    assert label_array.dtype in (np.int, np.int64, np.int32,
                                 np.uint, np.uint8), ('Expecting integer array as label_array!')
    return np.where(label_array > 0, 1, 0)


def expand_dims(array):
    """Expand a purely spatial array to the N x C x [spatial] layout."""
    return array[np.newaxis, np.newaxis, ...]


def get_extent(fpath):
    """Extent directly from HDF5 metadata as 3-tuple of slice objects."""
    extents = []
    with h5py.File(fpath, mode='r') as f:
        for key, value in f['label/label-0'].attrs.items():
            if key.lower().endswith('extent') and isinstance(value, np.ndarray):
                extents.append(value)
    
    extents = np.stack(extents, axis=0)
    maxs = np.max(extents, axis=0)
    mins = np.min(extents, axis=0)
    axis_slices = []
    for min_, max_ in zip(mins[::2], maxs[1::2]):
        axis_slices.append(slice(min_, max_, 1))
    return tuple(axis_slices)


def pad_to_source(array, axis_slices, pad_width, source_shape):
    """Pad array to original size."""
    # expand the pad width to per-axis pre & post width
    if isinstance(pad_width, int):
        pad_width = 3 * (2 * (pad_width,),)
    elif (isinstance(pad_width, (list, tuple))
          and all(isinstance(elem, int) for elem in pad_width)):
        pad_width = tuple(2 * (int(elem),) for elem in pad_width)
    else:
        assert len(pad_width) == 3, 'Requiring 3 pre & post pad specifications'
        assert all(isinstance(elem, (list, tuple)) for elem in pad_width)
        
    src_pad_width = []
    for (axis_slice,
         axis_pad,
         source_axis_size) in zip(axis_slices, pad_width, source_shape):
        
        pre_pad = axis_slice.start - axis_pad[0]
        post_pad = max(0, source_axis_size - (axis_slice.stop + axis_pad[1]))
        src_pad_width.append((pre_pad, post_pad))
        
    src_pad_width = tuple(src_pad_width)
    # print(src_pad_width)
    padded_array = np.pad(
        array, pad_width=src_pad_width, mode='constant',
        constant_values=np.min(array)
    )
    return padded_array 




class lazyproperty:

    def __init__(self, func: Callable) -> None:
        self.func = func

    
    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value




class Result:

    int_pred_path = 'prediction/prediction-0'
    int_norm_path = 'prediction/normalization-0'
    int_raw_path = 'raw/raw-0'
    int_label_path = 'label/label-0'


    def __init__(self,
                 data_path: Optional[PathLike],
                 prediction_path: Optional[PathLike],
                 pad_width: int = 25,
                 threshold: Optional[float] = None) -> None:

        if data_path:
            self.data_path = pathlib.Path(data_path)
            self._raw_shape = self.get_array_shape(self.data_path, self.int_raw_path)
            self._label_shape = self.get_array_shape(self.data_path, self.int_label_path)
            # TODO: Clean this up!
            self._extent = get_extent(self.data_path)
        else:
            self.data_path = None
            self._raw_shape = None
            self._label_shape = None
            self._extent = None

        if prediction_path:
            self.prediction_path = pathlib.Path(prediction_path)
            self._pred_shape = self.get_array_shape(self.prediction_path, self.int_pred_path)
            self._norm_shape = self.get_array_shape(self.prediction_path, self.int_norm_path)
        else:
            self.prediction_path = prediction_path
            self._pred_shape = None
            self._norm_shape = None
        
        self.pad_width = pad_width
        self.threshold = threshold

        assert self.data_path.is_file(), f'Prediction path {self.data_path} is not a file!'
        assert self.prediction_path.is_file(), f'Prediction path {self.prediction_path} is not a file!'

        self.TP_calc = npm.TP(reduction='sum')
        self.FP_calc = npm.FP(reduction='sum')
        self.FN_calc = npm.FN(reduction='sum')
        self.TN_calc = npm.TN(reduction='sum')



    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(data_path={self.data_path.resolve()}, '
                    f'prediction_path={self.prediction_path.resolve(9)}), '
                    f'pad_width={self.pad_width}, threshold={self.threshold}')
        return repr_str


    def get_prediction(self) -> np.ndarray:
        normalized_pred = self.get_normalized_prediction().squeeze()
        padded_pred = pad_to_source(
            array=normalized_pred, axis_slices=self._extent,
            pad_width=self.pad_width, source_shape=self._raw_shape
        )
        assert padded_pred.shape == self._raw_shape, 'Shape mismatch: Raw to padded pred!'
        
        if self.threshold:
            return np.where(padded_pred >= self.threshold, 1, 0)
        else:
            return padded_pred

    
    def get_normalization(self) -> np.ndarray:
        normalization = self.get_dataset(self.prediction_path, self.int_norm_path)
        return normalization

    
    def get_raw(self) -> np.ndarray:
        raw = self.get_dataset(self.data_path, self.int_raw_path)
        return raw


    def get_label(self) -> np.ndarray:
        label = self.get_dataset(self.data_path, self.int_label_path)
        return label



    def get_normalized_prediction(self) -> np.ndarray:
        pred = self.get_dataset(self.prediction_path, self.int_pred_path)
        norm = self.get_dataset(self.prediction_path, self.int_norm_path)
        return pred / norm

    
    @lazyproperty
    def TP(self) -> np.int32:
        assert self.threshold, 'Threshold value required for binary classification metric!'
        pred = expand_dims(self.get_prediction())
        trgt = expand_dims(merge_foreground(self.get_label()))
        return self.TP_calc(pred, trgt)

    @lazyproperty
    def TN(self) -> np.int32:
        assert self.threshold, 'Threshold value required for binary classification metric!'
        pred = expand_dims(self.get_prediction())
        trgt = expand_dims(merge_foreground(self.get_label()))
        return self.TN_calc(pred, trgt)


    @lazyproperty
    def FP(self) -> np.int32:
        assert self.threshold, 'Threshold value required for binary classification metric!'
        pred = expand_dims(self.get_prediction())
        trgt = expand_dims(merge_foreground(self.get_label()))
        return self.FP_calc(pred, trgt)


    @lazyproperty
    def FN(self) -> np.int32:
        assert self.threshold, 'Threshold value required for binary classification metric!'
        pred = expand_dims(self.get_prediction())
        trgt = expand_dims(merge_foreground(self.get_label()))
        return self.FN_calc(pred, trgt)


    @lazyproperty
    def dice(self) -> np.float32:
        denom = 2 * self.TP + self.FP + self.FN
        return 2 * self.TP / denom
    

    @lazyproperty
    def jaccard(self) -> np.float32:
        return self.TP / (self.TP + self.FP + self.FN)

    
    @lazyproperty
    def sensitivity(self) -> np.float32:
        return self.TP / (self.TP + self.FN)
    

    @lazyproperty
    def specificity(self) -> np.float32:
        return self.TN / (self.TN + self.FP)
    

    @lazyproperty
    def precision(self) -> np.float32:
        return self.TP / (self.TP + self.FP)


    @lazyproperty
    def volsim(self) -> np.float32:
        """Volumetric similarity"""
        denom = 2 * self.TP + self.FP + self.FN
        return 1 - np.abs(self.FN - self.FP) / denom


    @staticmethod
    def get_array_shape(filepath: PathLike, internal_path: str) -> Tuple:
        with h5py.File(filepath, mode='r') as f:
            shape = f[internal_path].shape
        return shape

    
    @staticmethod
    def get_dataset(filepath: PathLike, internal_path: str) -> np.ndarray:
        """
        Load the dataset from the HDF5 file.
        """
        with h5py.File(filepath, mode='r') as f:
            array = f[internal_path][...]
        return array


    @staticmethod
    def _nrrd_export(filepath: PathLike, arr: np.ndarray) -> None:
        filepath = pathlib.Path(filepath)
        if filepath.is_file():
            raise FileExistsError(f'Preexisting file at {filepath.resolve()}')
        nrrd.write(str(filepath), arr)
        return None


    def export_label(self, filepath: PathLike) -> None:
        self._nrrd_export(filepath, self.get_label())
        return None

    
    def export_prediction(self, filepath: PathLike) -> None:
        self._nrrd_export(filepath, self.get_prediction())
        return None

