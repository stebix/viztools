import collections.abc
import numpy as np
import matplotlib
import matplotlib.colors
import ipywidgets as widgets


def hex_to_rgb(hexcolor):
    """
    Convert hexadecimal color strings to RGB float tuple.
    #32a852 -> (0.20, 0.66, 0.32)

    Parameters
    ----------

    hexcolor : str
        Color as string specification.

    Returns
    -------

    rgb : tuple of float
        The converted RGB float tuple.
    """
    hexcolor = hexcolor.lstrip('#')
    hexcolor_rgb = []
    for i in (0, 2, 4):
        elem = int(hexcolor[i:i+2], 16) / 255
        hexcolor_rgb.append(elem)
    return tuple(hexcolor_rgb)


def create_alpha_cmap(base_color, max_alpha=1.0, alpha_ticks=50):
    """
    Construct a single-color colormap that provides a smooth
    increase of the alpha value from 0 to max_alpha for the
    desired color.

    Parameters
    ----------
    
    base_color : tuple
        RGB color specification as tuple of three floats
        in [0, 1]
    
    max_alpha : float, optional
        Maximum alpha value. Set < 1 if
        transparaency at max value is desired.
    
    alpha_ticks : int, optional
        Number of steps along the alpha
        transparency axis.
    
    Returns
    -------

    alpha_map : matplotlib.colors.Colormap
        The colormap instance.
    """
    alpha_range = np.linspace(0, max_alpha, num=alpha_ticks)[:, np.newaxis]
    rgb_arr = np.array((base_color))[np.newaxis, :]    
    
    colorspec = np.concatenate(
        [np.broadcast_to(rgb_arr, (alpha_range.shape[0], 3)), alpha_range],
        axis=-1
    )
    msg = f'Expected {(alpha_ticks, 4)}, got {colorspec.shape}'
    assert colorspec.shape == (alpha_ticks, 4), msg
    return matplotlib.colors.ListedColormap(colorspec)




def axis_idx_slider_factory(fig, axes_imgs, arrays, axis=None, slider_desc='Index'):
    """
    Create a slider enabling the synchronized display of slices of
    multiple 3D arrays in corresponding AxesImage instances.

    This is a factory function using closures to create fully synchronized
    ipywidgets.IntSlider list that selects the slice index along the given axis
    and triggers redraw of the canvas. Due to the full linking, only the first,
    i.e. primary slider ist returned.

    Parameters
    ----------

    fig : matplotlib.figure.Figure
        Figure instance in which the `matplotlib.axes.Axes` live.
        Used for the refreshing `fig.canvas.draw_idle()` call.
    
    axes_imgs :  list of matplotlib.image.AxesImage
        AxesImage instances trough which the new image
        data is set.
    
    arrays : list of numpy.ndarray
        The data sources. All arrays are expected to be of the
        same shape to avert IndexErrors.
    
    axis : int, optional
        The axis along which the IntSlider selects the
        the slice. Defaults to 0, i.e. first axis.
    
    slider_desc : str, optional
        The description string displayed by the Slider
        widget. Defaults to 'Index'.
    
    Returns
    -------

    ipywidgets.IntSlider
        The integer slider instance that controls the
        synchronized slice display.
    """
    if not isinstance(axes_imgs, collections.abc.Sequence):
        axes_imgs = [axes_imgs]
    if not isinstance(arrays, collections.abc.Sequence):
        arrays = [arrays]
    assert len(axes_imgs) == len(arrays), 'Requires one source array per mappable!'

    shapeset = set()
    for array in arrays:
        shapeset.add(array.shape)
    msg = f'All arrays have to be of the same shape! Found shapes: {shapeset}'
    assert len(shapeset) == 1, msg

    if axis is not None:
        axswapped_arrays = []
        for array in arrays:
            axswapped_arrays.append(np.swapaxes(array, 0, axis))
        arrays = axswapped_arrays

    # closure factory function
    def modify_idx_factory(mappable, array):
        def modify_idx(change):
            mappable.set_data(array[change['new'], ...])
            fig.canvas.draw_idle()
        return modify_idx

    sliders = []
    for mappable, array in zip(axes_imgs, arrays):
        handler = modify_idx_factory(mappable, array)
        slider = widgets.IntSlider(value=0, min=0, max=array.shape[0] - 1,
                                   description=slider_desc)
        slider.observe(handler, names='value')
        sliders.append(slider)

    # couple/link all together
    primary_slider = sliders[0]
    links = [
        widgets.link((primary_slider, 'value'), (slider, 'value'))
        for slider in sliders[1:]
    ]
    return primary_slider


def cmap_color_factory(fig, axes_img, description=''):
    """
    Create a slider for maximum alpha value and a picker for color value.

    Parameters
    ----------

    fig : matplotlib.figure.Figure
        Figure instance in which the `matplotlib.axes.Axes` live.
        Used for the refreshing `fig.canvas.draw_idle()` call.
    
    axes_img : matplotlib.image.AxesImage
        The AxesImage as returned by `ax.imshow(...)` et al.
    
    description : str, optional
        Description string that is prepended to the color picker
        and slider description.

    Returns
    -------

    (cpicker, alphaslider) : tuple of ipywidgets widget instances
        The tuple of constructed widgets.
        cpicker -> Color picker widget
        alphaslider -> Maximal alpha value selection slider
    """
    alphaslider = widgets.FloatSlider(value=1.0, min=0,
                                      max=1.0, step=0.1,
                                      description=description + ' alpha')
    cpicker = widgets.ColorPicker(concise=False,
                                  description=description + ' color',
                                  value='#ff5733')
    
    def modify_cmap_color(change):
        """
        Set the colormap of `axes_img` to a alpha colormap with
        the base color deduced from `change`.
        """
        color = hex_to_rgb(change['new'])
        cmap = create_alpha_cmap(color, alphaslider.value)
        axes_img.set_cmap(cmap)
        fig.canvas.draw_idle()

    def modify_cmap_alpha(change):
        """
        Set the colormap of `axes_img` to a new alpha colormap
        with the `max_alpha` value deduced from change.
        """
        color = hex_to_rgb(cpicker.value)
        cmap = create_alpha_cmap(color, change['new'])
        axes_img.set_cmap(cmap)
        fig.canvas.draw_idle()
    
    cpicker.observe(modify_cmap_color, names='value')
    alphaslider.observe(modify_cmap_alpha, names='value')

    return (cpicker, alphaslider)