"""
Home for various utils that live here until this gets cleaned up into multiple separate
sub-modules.

The code is mainly concernened with providing eas-of-use for working with
predictions on datasets. 
"""

import pathlib
import numpy as np
import h5py

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





