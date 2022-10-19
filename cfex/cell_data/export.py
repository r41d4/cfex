import arrow
import typing as t
from pathlib import Path
import pandas as pd
from skimage.io import imsave
from tqdm import tqdm

from cfex.cell_data.geometry import calculate_centroid


def save_cell_objects_image_data(
    cell_data: pd.DataFrame,
    export_path: t.Union[Path, str],
    include_masks: t.Optional[t.Sequence[str]] = ["nucleus", "outline"],
    show_progress: t.Optional[bool] = False,
):
    """
    Save images of cells within the bounds of regions described in data.

    Parameters 
    ----------
    cell_data : DataFrame
        DataFrame containing cell polygon, image and label data.
    export_path : Path or str
        Path to the output directory
    include_masks : array-like of str, optional, default ["nucleus", "outline"]
        Sequence of mask types to include in the export.
    show_progress: bool, optional, default False
        Flag for printing export progress to stdout.

    Returns
    -------
    Path
        Path to the timestamped directory with image files.
    """
    cell_data_iterator = cell_data.iterrows()
    if show_progress: 
        cell_data_iterator = tqdm(cell_data_iterator)
    cell_images_dirname = Path(f"cells_{arrow.now().isoformat()}")
    cell_images_path = Path(export_path) / cell_images_dirname
    cell_images_path.mkdir(parents=True, exist_ok=True)
    for i, cell_data_point in cell_data_iterator:
        try:
            target_name = cell_data_point.Target
        except:
            target_name = "n"
        scan_name = cell_data_point.WSI
        centroid_x, centroid_y = calculate_centroid(
            cell_data_point.NucleusPolygon
        ).astype("int")
        cell_image_filename = (
            f"cell{i}_{target_name}_{centroid_x}_{centroid_y}_{scan_name}"
        )
        imsave(
            cell_images_path / f"{cell_image_filename}.png",
            cell_data_point.ColorImage,
            check_contrast=False,
        )
        if include_masks:
            for mask_type in include_masks:
                mask_name = f"{mask_type.capitalize()}Mask"
                imsave(
                    cell_images_path / f"{cell_image_filename}_{mask_name}.tif",
                    cell_data_point[mask_name].astype("uint16"),
                    check_contrast=False,
                )
    return cell_images_path