import pathlib
import logging
import numpy as np
from typing import (List, Tuple, Union, Dict, Iterable, NewType, Optional,
                    Callable, Any)

import h5py
import nrrd

import src.numpy_metrics as npm
from src.utils import get_extent, pad_to_source, expand_dims, merge_foreground


PathLike = NewType(name='PathLike', tp=Union[str, pathlib.Path])
_log = logging.getLogger(__name__)

def init_logger(logger=None):
    """
    Set up the module logger. 
    """
    global _log
    if logger is not None:
        _log = logger
    else:
        pass



class lazyproperty:
    """
    Descriptor class for lazily computed evaluation metrics of
    segmentation results.
    """

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

    # canonical internal HDF5 paths for the different datasets
    # these depend on the setting of the segmentation_net package
    int_pred_path = 'prediction/prediction-0'
    int_norm_path = 'prediction/normalization-0'
    int_raw_path = 'raw/raw-0'
    int_label_path = 'label/label-0'


    def __init__(self,
                 data_path: Optional[PathLike],
                 prediction_path: Optional[PathLike],
                 pad_width: int = 25,
                 threshold: Optional[float] = None,
                 id_: Any = None) -> None:

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
        self.id_ = id_

        assert self.data_path.is_file(), f'Path to data {self.data_path} is not a file!'
        assert self.prediction_path.is_file(), f'Path to prediction data {self.prediction_path} is not a file!'

        self.TP_calc = npm.TP(reduction='sum')
        self.FP_calc = npm.FP(reduction='sum')
        self.FN_calc = npm.FN(reduction='sum')
        self.TN_calc = npm.TN(reduction='sum')



    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(id={self.id_}, '
                    f'data_path={self.data_path.resolve()}, '
                    f'prediction_path={self.prediction_path.resolve(9)}), '
                    f'pad_width={self.pad_width}, threshold={self.threshold}')
        return repr_str


    def get_prediction(self) -> np.ndarray:
        """
        Get the final prediction. This includes normalization and padding
        towards the initial shape.
        """
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
        """
        Get the voxel-wise normalization. Due to the sliding-window approach,
        some voxels are predicted multiple times. The output activations
        have to be normalized with the number of predictions.
        """
        normalization = self.get_dataset(self.prediction_path, self.int_norm_path)
        return normalization

    
    def get_raw(self) -> np.ndarray:
        """
        Get the voxel raw data.
        """
        raw = self.get_dataset(self.data_path, self.int_raw_path)
        return raw


    def get_label(self) -> np.ndarray:
        """
        Get the label data.
        """
        label = self.get_dataset(self.data_path, self.int_label_path)
        return label


    def get_normalized_prediction(self) -> np.ndarray:
        """
        Get the normalized but unpadded voxel prediction data.
        """
        pred = self.get_dataset(self.prediction_path, self.int_pred_path)
        norm = self.get_dataset(self.prediction_path, self.int_norm_path)
        return pred / norm

    
    def get_label_prediction(self, use_roi: bool = False) -> Tuple:
        """
        Convenience access to both label and prediction as a tuple.
        """
        label = self.get_label()
        prediction = self.get_prediction()
        if use_roi:
            roi_spec = self._extent
        else:
            roi_spec = np.s_[...]
        return (label[roi_spec], prediction[roi_spec])

    
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
        """
        Access the shape attribute of a HDF5 dataset.
        """
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
    

    @classmethod
    def from_pathmap(cls, pathmap: Dict, kwargs: Optional[Dict] = None) -> 'List[Result]':
        """
        Programmatically construct Result objects from a raw-to-prediction
        pathmap. 

        Parameters
        ----------

        pathmap: Dict
            Nested dictionary mapping IDs to path dicts that hold
            the paths for the keys 'data_path' and 'prediction_path'

        kwargs: optional, Dict
            Optional keyword argument dict passed through to
            the `Result` object `__init__`
        """
        kwargs = kwargs or {}
        created_results = []
        for id_, pathdict in pathmap.items():
            created_results.append(
                cls(id_=id_, **pathdict, **kwargs)
            )
        return created_results





def raw_pred_pathmap(raw_dir: PathLike, prediction_dir: PathLike,
                     splitchar: str = '_') -> Dict:
    """
    Produce a mapping from integer indices deduced from filenames to
    pairs of filepaths that point to the raw and prediction HDF5 datasets.
    Intended use: Simple construction of Result objects from directories. 
    """
    raw_dir = pathlib.Path(raw_dir)
    prediction_dir = pathlib.Path(prediction_dir)
    raw_paths = {}
    path_mapping = {}
    
    for item in raw_dir.iterdir():
        _log.debug(f'Raw dir item: {item.resolve()}')
        if item.name.endswith('.hdf5'):
            raw_paths[int(item.stem)] = item.resolve()
            _log.debug(f'Added raw path item: {item.resolve()}')
            
    for item in prediction_dir.iterdir():
        _log.debug(f'Prediction dir item: {item.resolve()}')
        if item.name.endswith('.hdf5'):
            _log.debug(f'Attempting to parse prediction dir item: {item.resolve()}')
            try:
                int_id = int(item.name.split(splitchar)[0])
            except ValueError:
                _log.warning(f'Failed to parse file with path: {item.resolve()} -> Skipping item')
                continue
            try:
                raw_path = raw_paths[int_id]
            except KeyError:
                _log.warning(f'Raw to prediction file mismatch for pred dir item: {item.resolve()}')
                continue
            path_mapping[int_id] = {
                'data_path' : raw_path,
                'prediction_path' : item.resolve()
            }
    return path_mapping
