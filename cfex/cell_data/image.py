import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import slideio as sio

from typing import Optional, Union, List
from cfex.enums import CellImageLoadBackend
from cfex.cell_data.geometry import calculate_centroid, calculate_cell_roi_bounding_box


def _load_cell_images_slideio(
    wsi_path: str,
    cell_data: pd.DataFrame,
    bounding_box_margin: Optional[int],
    show_progress: Optional[bool],
) -> List[np.ndarray]:
    slide = sio.open_slide(wsi_path, "SVS")
    scene = slide.get_scene(0)
    cell_data_iterable = cell_data.iterrows()
    if show_progress:
        cell_data_iterable = tqdm(cell_data_iterable)
    cell_image_list = []
    for _, single_cell_data in cell_data_iterable:
        cell_polygon = single_cell_data["CellPolygon"]
        cell_centroid = calculate_centroid(cell_polygon).astype(int)
        cell_box = calculate_cell_roi_bounding_box(cell_centroid, bounding_box_margin)
        image = scene.read_block(cell_box)
        cell_image_list.append(image)
    return cell_image_list


# TODO: implement a CellImageLoader class


def load_cell_images(
    wsi_path: Union[str, Path],
    cell_data: pd.DataFrame,
    cell_image_load_backend: str,
    bounding_box_margin: Optional[int] = 50,
    show_progress: Optional[bool] = False,
) -> List[np.ndarray]:
    """
    Load WSI regions containing given cells to memory.
    Regions are represented as cell bounding boxes defined by a given margin.

    Returns a list of cell images.

    Parameters
    ----------
    wsi_path : str or Path
        Path to the WSI from which the cell objects are analyzed.
    cell_data : DataFrame
        DataFrame containing cell polygons.
    cell_image_load_backend : str
        Name of the supported WSI load backend.
    bounding_box_margin : int, optional, default 50
        Distance from the cell centroid to the side of the desired bounding box.
    show_progress : bool, optional, default False
        Flag for printing image loading progress to stdout.

    Returns
    -------
    list of ndarray
        List containing cell images.
    """
    if cell_image_load_backend in CellImageLoadBackend.values():
        load_cell_images_func = globals()[
            f"_load_cell_images_{cell_image_load_backend}"
        ]
        return load_cell_images_func(
            str(wsi_path), cell_data, bounding_box_margin, show_progress
        )
