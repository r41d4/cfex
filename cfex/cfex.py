import click

# import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd

from cfex.cell_data.extract import extract_cell_data
from cfex.cell_data.image import load_cell_images
from cfex.cell_data.detect import detect_cells
from cfex.cell_data.mask import create_object_image_data
from cfex.cell_data.export import save_cell_objects_image_data
from cfex.feature_extraction.extract import extract_measurements

# TODO: make the script launch faster by restructuring entry points and local imports

# TODO: implement an alternative way to call this function
def load_data(
    wsi_path: Path,
    cell_data_path: Path,
    size: Optional[int],
    cell_image_load_backend: str,
    bounding_box_margin: Optional[int] = 50,
    extract_measurements: bool = False,
    silent: bool = False,
):
    verbose_print(
        "[loading input data]",
        ":: WSI path:",
        wsi_path,
        ":: Cell data path:",
        cell_data_path,
        sep="\n",
    )
    wsi_path, cell_data_path = Path(wsi_path).resolve(), Path(cell_data_path).resolve()
    wsi_name = wsi_path.stem.split(".")[0]
    verbose_print(":: Extracting cell data... ", end="", flush=True)
    with open(cell_data_path) as cell_data:
        cell_data = extract_cell_data(
            cell_data, data_format="qupath", extract_measurements=extract_measurements
        )[:size]
        cell_data["WSI"] = wsi_name
        verbose_print("Done!")
    verbose_print(f":: Cell object count: {len(cell_data.index)}")
    verbose_print(
        f":: Loading WSI regions defined by a {bounding_box_margin * 2}x{bounding_box_margin * 2} pixels bounding box...",
    )
    cell_image_list = load_cell_images(
        wsi_path=wsi_path,
        cell_data=cell_data,
        cell_image_load_backend="slideio",
        bounding_box_margin=bounding_box_margin,
        show_progress=not silent,
    )
    return cell_data, cell_image_list


# TODO: account for cell boxes around the edges of the slide
def get_segmentation_data(cell_image_list, cell_detection_backend, silent=True):
    verbose_print(
        "[instance segmentation]",
        ":: Running cell instance segmentation...",
        sep="\n",
    )
    cell_detected_nucleus_list, unsegmented_cell_data = detect_cells(
        cell_image_list=cell_image_list,
        cell_detection_backend=cell_detection_backend,
        show_progress=not silent,
        stash_undetected=True,
    )
    verbose_print(
        f":: Found cell instances: {len(cell_detected_nucleus_list) - len(unsegmented_cell_data)}",
    )
    return cell_detected_nucleus_list, unsegmented_cell_data


def get_image_object_data(cell_image_list, cell_detected_nucleus_list, silent=True):
    verbose_print(
        "[object mask generation]", ":: Creating cell object masks...", sep="\n"
    )
    image_object_data = create_object_image_data(
        cell_image_list, cell_detected_nucleus_list, to_dataframe=True
    )
    # TODO: add modes (e.g. cell box image extraction, cell feature extraction)
    return image_object_data


def export_to_files(cell_data, image_object_data, export_path):
    verbose_print("[export]", ":: Saving cell images...", sep="\n")
    cell_data_extended = pd.concat([cell_data, image_object_data], axis=1).dropna()
    cell_images_path = save_cell_objects_image_data(
        cell_data_extended,
        export_path,
        show_progress=True,
    )
    return cell_images_path


def extract_features(
    cell_images_path,
    feature_extraction_backend,
    output_path,
    cell_profiler_pipeline_path,
):
    verbose_print("[extraction]", ":: Extracting cell features...", sep="\n")
    extract_measurements(
        cell_images_path=cell_images_path,
        feature_extraction_backend=feature_extraction_backend,
        output_path=output_path,
        cell_profiler_pipeline_path=cell_profiler_pipeline_path,
    )


@click.command()
@click.option(
    "-w",
    "--wsi",
    prompt="WSI path",
    type=click.Path(resolve_path=True, exists=True, file_okay=True),
    required=False,
    help="Path to the WSI file",
)
@click.option(
    "-d",
    "--data",
    type=click.Path(resolve_path=True, exists=True, file_okay=True),
    prompt="Cell object data file path",
    help="Path to the cell object data file",
)
@click.option(
    "-s",
    "--size",
    type=int,
    required=False,
    help="Amount of objects to be loaded for feature extraction",
)
@click.option(
    "-p",
    "--protocol",
    required=False,
    type=click.Path(resolve_path=True, exists=True, file_okay=True),
    help="Path to the protocol JSON file",
)
@click.option(
    "--cell-image-export-path",
    type=click.Path(resolve_path=True, exists=True, dir_okay=True),
    required=False,
    help="Path to the output directory for cell images and masks",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(resolve_path=True, exists=True, dir_okay=True),
    required=False,
    help="Path to the output directory for cell feature data",
)
@click.option(
    "--cell-profiler-pipeline-path",
    type=click.Path(resolve_path=True, exists=True, file_okay=True),
    required=False,
    help="Path to the cell profiler pipeline file",
)
@click.option(
    "-m",
    "--measurement-extraction",
    is_flag=True,
    default=False,
    help="Extract existing measurements from the cell object data file as additional features",
)
@click.option(
    "--silent",
    is_flag=True,
    default=False,
    help="Suppress messages when going through processing stages",
)
def run_extraction(
    wsi,
    data,
    size,
    measurement_extraction,
    protocol,
    cell_image_export_path,
    output_path,
    cell_profiler_pipeline_path,
    silent,
):
    """Extract features from cell data"""
    global verbose_print
    verbose_print = print if not silent else lambda *args, **kwargs: None
    cell_data, cell_image_list = load_data(
        wsi_path=wsi,
        cell_data_path=data,
        size=size,
        cell_image_load_backend="slideio",
        extract_measurements=measurement_extraction,
        silent=silent,
    )
    cell_detected_nucleus_list, _ = get_segmentation_data(
        cell_image_list, cell_detection_backend="stardist", silent=silent
    )
    image_object_data = get_image_object_data(
        cell_image_list=cell_image_list,
        cell_detected_nucleus_list=cell_detected_nucleus_list,
        silent=silent,
    )
    # TODO: alternative calls to functions called below
    cell_images_path = export_to_files(
        cell_data=cell_data,
        image_object_data=image_object_data,
        export_path=Path(cell_image_export_path),
    )
    extract_features(
        cell_images_path=cell_images_path,
        feature_extraction_backend="cellprofiler",
        output_path=Path(output_path),
        cell_profiler_pipeline_path=cell_profiler_pipeline_path,
    )


def main():
    # TODO: add a mode to export cell box images with or without masks
    run_extraction()


if __name__ == "__main__":
    main()
