import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets

from itertools import cycle
from typing import Union, Dict, List, Tuple, Iterable, Optional

from src.utils import Result
from src.vizfuncs import create_alpha_cmap, axis_idx_slider_factory


class SliceDisplay:

    def __init__(self, array_data):
        pass


    @staticmethod
    def create_figure(subplots_kwargs: Union[Dict, None] = None) -> matplotlib.figure.Figure:
        """
        Create the figure instance.

        Parameters
        ----------

        subplots_kwargs : dictionary, optional
            Keyword arguments that are forwarded to
            plt.supplots(). Defaults to None.

        
        Returns
        -------

        figure : matplotlib.figure.Figure
            The figure, i.e. the top level container for all plot elements.
        """
        if subplots_kwargs is None:
            subplots_kwargs = {}
        return plt.subplots(**subplots_kwargs)




class SegmentationDisplay:

    def __init__(self,
                 raw_volume: np.ndarray,
                 segments: Iterable[np.ndarray],
                 axis: int = -1,
                 subplots_kwargs: Optional[Dict] = None) -> None:

        # data attributes    
        self.raw_volume = raw_volume
        self.raw_min = np.min(raw_volume)
        self.raw_max = np.max(raw_volume)
        self.segments = segments
        self.arrays = self.initialize_arrays()

        # plotting attributes
        self.subplots_kwargs = subplots_kwargs or {}
        self.axis = axis

        # plotting settings not exposed currently
        self.initial_slice = np.s_[..., 0]
        self.segment_color_cycle = [(0,0,1), (0,1,0), (1,0,0)]
        self.raw_cmap = 'gray'

        self.fig, self.ax, self.axes_imgs = self.initialize_figure()


    def initialize_figure(self) -> Tuple:
        fig, ax = plt.subplots(**self.subplots_kwargs)
        ax.set_axis_off()

        axes_imgs = []
        # first entry is the raw data
        axes_imgs.append(
            ax.imshow(self.raw_volume[self.initial_slice], cmap=self.raw_cmap)
        )
        # other entries are the segment overlays
        for color, segment in zip(cycle(self.segment_color_cycle), self.segments):
            cmap = create_alpha_cmap(base_color=color)
            axes_imgs.append(
                ax.imshow(segment[self.initial_slice], vmin=0, vmax=1, cmap=cmap)
            )
        return (fig, ax, axes_imgs)

    
    def initialize_arrays(self) -> List[np.ndarray]:
        arrays = [self.raw_volume]
        arrays.extend(self.segments)
        return arrays


    def create(self) -> Tuple:
        slider = axis_idx_slider_factory(
            fig=self.fig, axes_imgs=self.axes_imgs,
            arrays=self.arrays, axis=self.axis
        )
        return (self.fig, slider)

    
    @classmethod
    def from_result(cls, result: Result,
                    subplots_kwargs: Optional[Dict] = None) -> 'SegmentationDisplay':
        """
        Directly build visualization from a Result object.
        """
        return cls(raw_volume=result.get_raw(), segments=[result.get_prediction()],
                   subplots_kwargs=subplots_kwargs)
        