import numpy as np
from tqdm import tqdm
from typing import Union, Optional, Sequence, Tuple, List, Dict 

from cfex.enums import CellDetectionBackend
from cfex.cell_data.geometry import calculate_image_center
from skimage.io import imsave
import imageio


def get_cell_box_segmentation_status(cell_box_labels: np.ndarray) -> bool:
    cell_box_image_center = calculate_image_center(cell_box_labels)
    return bool(cell_box_labels[cell_box_image_center])


def _detect_cells_stardist(
    cell_image_list: Sequence[np.ndarray],
    stash_undetected: Optional[bool],
    show_progress: Optional[bool],
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict]]:
    from stardist.plot import render_label
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained("2D_versatile_he")
    segmented_count = 0
    unsegmented_cell_data = []
    cell_detected_nucleus_list = []
    if len(cell_image_list) == 1:
        image = cell_image_list[0]
        labels, _ = model.predict_instances(
            normalize(image, 1, 99.8, axis=(0, 1, 2)),
            show_tile_progress=False,
            verbose=False,
        )
        imsave("test.png", render_label(labels, image))
        # imsave("test.png", image)
        return labels
    else: 
        cell_image_iterable = enumerate(cell_image_list)
        if show_progress:
            cell_image_iterable = tqdm(cell_image_iterable)
        for i, image in cell_image_iterable:
            labels, _ = model.predict_instances(
                normalize(image, 1, 99.8, axis=(0, 1, 2)),
                show_tile_progress=False,
                verbose=False,
            )
            segmentation_status = get_cell_box_segmentation_status(labels)
            cell_detected_nucleus_list.append(labels)
            if segmentation_status:
                segmented_count += 1
            else:
                predictions_overlayed = render_label(labels, img=image)
                unsegmented_cell_data.append(
                    {"id": i, "labeled_image": predictions_overlayed}
                )
        # TODO: check why cells are correctly segmented by
        # StarDist more often (46 > 43) on smaller cell boxes
        if stash_undetected:
            return (cell_detected_nucleus_list, unsegmented_cell_data)
        return cell_detected_nucleus_list



def detect_cells(
    cell_image_list: Sequence[np.ndarray],
    cell_detection_backend: str,
    stash_undetected: Optional[bool] = False,
    show_progress: Optional[bool] = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict]]:
    """
    Run cell instance segmentation on a given list of images.

    Returns a list of cell labels, or if stash_undetected is True:
    a tuple containing a list of cell labels and a dictionary with undetected
    cell indices as keys and cell images with overlayed labels as values.

    Parameters
    ----------
    cell_image_list : array-like of ndarray
        List containing cell images.
    cell_detection_backend : str
        Name of the supported cell detection backend.
    stash_undetected : bool, optional, default False
        Flag for stashing unsegmented cell instances in a dictionary
        to be returned by the function as a second element of a tuple.
    show_progress: bool, optional, default False
        Flag for printing cell instance segmentation progress to stdout.

    Returns
    -------
    list or tuple of list and dict
        List of cell labels or a tuple with a list of cell labels as the first element
        and a dictionary with keys being undetected cell indices and values being
        cell images with overlayed labels as the second element
    """
    if cell_detection_backend in CellDetectionBackend.values():
        detect_cells_func = globals()[f"_detect_cells_{cell_detection_backend}"]
        return detect_cells_func(
            cell_image_list=cell_image_list,
            show_progress=show_progress,
            stash_undetected=stash_undetected,
        )
