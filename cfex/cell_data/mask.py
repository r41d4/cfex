import typing as t
import pandas as pd
import numpy as np
from skimage.measure import find_contours, regionprops
from skimage.draw import polygon
import pyclipper

from cfex.cell_data.geometry import calculate_image_center

# TODO: implement a CellMaskGenerator class


def create_nucleus_mask(nuclei_labels_image: np.ndarray) -> np.ndarray:
    """
    Create a mask of cell nucleus.

    Returns cell nucleus label.

    Parameters
    ----------
    nuclei_labels_image : ndarray
        Multi-label image of a WSI region defined by cell bounding box.

    Returns
    -------
    ndarray
        Cell nucleus label.
    """
    image_center_coordinates = calculate_image_center(nuclei_labels_image)
    focused_nucleus_color = nuclei_labels_image[image_center_coordinates]
    nucleus_mask = nuclei_labels_image.copy()
    nucleus_mask[nucleus_mask[:][:] != focused_nucleus_color] = 0
    return nucleus_mask.clip(0, 1)


def create_cell_expansion_mask(
    nucleus_mask: np.ndarray, expansion_size: t.Optional[int] = 6
) -> np.ndarray:
    """
    Create a mask of cell expansion (area around the nucleus).

    Returns cell expansion label.

    Parameters
    ----------
    nucleus_mask : ndarray
        Cell nucleus label.
    expansion_size : int, optional, default 6
        Factor by which the initial area is offset.

    Returns
    -------
    ndarray
        Cell expansion label.
    """
    nucleus_mask_polygon = find_contours(nucleus_mask)
    if nucleus_mask_polygon == []:
        return np.zeros_like(nucleus_mask)
    nucleus_mask_polygon = nucleus_mask_polygon[0]
    clip_offset = pyclipper.PyclipperOffset()
    clip_offset.AddPath(
        nucleus_mask_polygon, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON
    )
    dilated_polygon = np.array(clip_offset.Execute(expansion_size))[0]
    expansion_r, expansion_c = polygon(
        dilated_polygon[:, 0], dilated_polygon[:, 1], nucleus_mask.shape
    )
    cell_expansion_mask = nucleus_mask.copy()
    cell_expansion_mask[expansion_r, expansion_c] = 1
    return cell_expansion_mask.clip(0, 1)


def create_cell_outline_mask(
    nucleus_mask: np.ndarray, cell_expansion_mask: np.ndarray
) -> np.ndarray:
    """
    Create a mask of cell outline (cell expansion - cell nucleus).

    Returns cell outline label.

    Parameters
    ----------
    nucleus_mask : ndarray
        Cell nucleus label.
    cell_expansion_mask : ndarray
        Cell expansion label.

    Returns
    -------
    ndarray
        Cell outline label.
    """
    return cell_expansion_mask - nucleus_mask


def create_object_masks_data(
    cell_detected_nucleus_list: t.Sequence[np.ndarray],
    to_dataframe: t.Optional[bool] = False,
):
    """
    Create masks of cell objects.

    Returns a dictionary (or a DataFrame if to_dataframe is True):
    NucleusMask value contains a list of cell nuclei labels.
    CellExpansionMask value contains a list of cell expansion labels.
    CellOutlineMask value contains a list of cell outline (expansion - nucleus) labels.

    Parameters
    ----------
    cell_detected_nucleus_list : array-like of ndarray
        List of cell labels.
    to_dataframe : bool, optional, default False
        Flag for converting resulting dictionary to a DataFrame.

    Returns
    -------
    dict or DataFrame
        Data instance containing cell objects masks.
    """
    nucleus_mask_list = [
        create_nucleus_mask(nuclei_labels_image)
        for nuclei_labels_image in cell_detected_nucleus_list
    ]
    cell_expansion_mask_list = [
        create_cell_expansion_mask(nucleus_mask) for nucleus_mask in nucleus_mask_list
    ]
    cell_outline_mask_list = cell_outline_mask_list = [
        create_cell_outline_mask(nucleus_mask, cell_expansion_mask)
        for (nucleus_mask, cell_expansion_mask) in zip(
            nucleus_mask_list, cell_expansion_mask_list
        )
    ]
    object_masks = {
        "NucleusMask": nucleus_mask_list,
        "ExpansionMask": cell_expansion_mask_list,
        "OutlineMask": cell_outline_mask_list,
    }
    if to_dataframe:
        return pd.DataFrame.from_dict(object_masks)
    return object_masks


def create_object_image_data(
    # TODO: exclude undetected cells from running mask creation
    cell_image_list: t.Sequence[np.ndarray],
    cell_detected_nucleus_list: t.Sequence[np.ndarray],
    to_dataframe: t.Optional[bool] = False,
):
    """
    Assemble cell object image data structure with generated masks.

    Returns a dictionary (or a DataFrame if to_dataframe is True):
    ColorImage value contains a list of colored cell images.
    NucleusMask value contains a list of cell nuclei labels.
    CellExpansionMask value contains a list of cell expansion labels.
    CellOutlineMask value contains a list of cell outline (expansion - nucleus) labels.

    Parameters
    ----------
    cell_image_list : array-like of ndarray
        List containing cell images.
    cell_detected_nucleus_list : array-like of ndarray
        List of cell labels.
    to_dataframe : bool, optional, default False
        Flag for converting resulting dictionary to a DataFrame.

    Returns
    -------
    dict or DataFrame
        Data instance containing cell images and cell objects masks.
    """
    object_masks = create_object_masks_data(cell_detected_nucleus_list, to_dataframe)
    object_masks["ColorImage"] = cell_image_list
    return object_masks
