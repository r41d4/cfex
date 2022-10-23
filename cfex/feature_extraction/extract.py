import os
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Union, List

from cfex.enums import CellFeaturesBackend


# TODO: refactor batch processing


def _prepare_data_cellprofiler(cell_images_path: Path) -> List:
    single_batch = {
        "png": [path.as_uri() for path in list(cell_images_path.glob("*.png"))],
        "tif_nucleus": [
            path.as_uri() for path in list(cell_images_path.glob("*NucleusMask.tif"))
        ],
        "tif_outline": [
            path.as_uri() for path in list(cell_images_path.glob("*OutlineMask.tif"))
        ],
    }
    batches = [single_batch]
    print(
        "\n".join(
            (
                f":: Images: {len(single_batch['png'])}",
                f":: Nuclei masks: {len(single_batch['tif_nucleus'])}",
                f":: Nucleus outline masks: {len(single_batch['tif_outline'])}",
            ),
        )
    )
    return batches


def _run_pipeline_cellprofiler(batches: List, pipeline_path: Path, output_path: Path):
    import cellprofiler_core.pipeline
    import cellprofiler_core.preferences
    import cellprofiler_core.utilities.java

    cellprofiler_core.preferences.set_headless()
    cellprofiler_core.utilities.java.start_java()
    output_path_pipeline = output_path / "pipeline"
    output_path_pipeline.mkdir(exist_ok=True)
    print(f":: Running pipeline: {pipeline_path.name}...")
    for batch in tqdm(batches):
        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.load(pipeline_path)
        pipeline.clear_urls()
        pipeline.read_file_list(batch["png"])
        pipeline.read_file_list(batch["tif_nucleus"])
        pipeline.read_file_list(batch["tif_outline"])
        pipeline.modules()[-1].directory.value = f"Elsewhere...|{output_path}"
        pipeline_output = pipeline.run()
        try:
            os.rename(
                output_path / "exported_Experiment.csv",
                output_path_pipeline
                / f"exported_Experiment_{batches.index(batch)}.csv",
            )
            os.rename(
                output_path / "exported_Nucleus.csv",
                output_path_pipeline / f"exported_Nucleus_{batches.index(batch)}.csv",
            )
            os.rename(
                output_path / "exported_NucleusObject.csv",
                output_path_pipeline
                / f"exported_NucleusObject_{batches.index(batch)}.csv",
            )
            os.rename(
                output_path / "exported_Outline.csv",
                output_path_pipeline / f"exported_Outline_{batches.index(batch)}.csv",
            )
            os.rename(
                output_path / "exported_OutlineObject.csv",
                output_path_pipeline
                / f"exported_OutlineObject_{batches.index(batch)}.csv",
            )
            os.rename(
                output_path / "exported_Image.csv",
                output_path_pipeline / f"exported_Image_{batches.index(batch)}.csv",
            )
        except:
            batch_num = batches.index(batch)
            print(
                f":: Processing of batch {batch_num} did not succeed, check the measure_cells_batch_{batch_num}_contents.json file for batch contents."
            )
            with open(
                f"measure_cells_batch_{batch_num}_contents.json", "a"
            ) as log_file:
                json.dump(log_file, batch)
    cellprofiler_core.utilities.java.stop_java()
    return pipeline_output


def _filter_data_cellprofiler(batches: List, output_path: Path):
    output_path_filtered = output_path / "filtered"
    output_path_filtered.mkdir(exist_ok=True)
    output_path_pipeline = output_path / "pipeline"
    object_names = ("NucleusObject", "OutlineObject")
    metadata_column_regex = "Metadata|FileName|PathName|Number_Object_Number|Parent_Cell|ImageNumber|ObjectNumber"
    measurement_axis_column_regex = "[\S]+_X|Y|Z$"
    processed_batches = {object_name: [] for object_name in object_names}
    for i, object_name in enumerate(object_names):
        print(f":: Processing {object_name} cell_features_data...")
        for i in range(len(batches)):
            cell_features_data = pd.read_csv(
                output_path_pipeline / f"exported_{object_name}_{i}.csv"
            )
            centroid_coordinates = cell_features_data["FileName_Color"].apply(
                lambda x: x.split("_")[2:4]
            )
            slide_name = cell_features_data["FileName_Color"].apply(
                lambda x: x.replace(x, str(Path(x).stem.split("_")[4]))
            )
            metadata_columns = list(
                cell_features_data.filter(regex=metadata_column_regex)
            )
            measurement_axis_columns = list(
                cell_features_data.filter(regex=measurement_axis_column_regex)
            )
            redundant_columns = metadata_columns + measurement_axis_columns
            cell_features_data.drop(columns=redundant_columns, inplace=True)
            cell_features_data.columns = f"{object_name}_" + cell_features_data.columns
            cell_features_data["CentroidCoordinates"] = centroid_coordinates
            cell_features_data["SlideName"] = slide_name
            processed_batches[object_name].append(cell_features_data)
    print(":: Merging batches for each of the objects...")
    for object_name, data_list in processed_batches.items():
        cell_features_data = pd.concat(data_list, ignore_index=True)
        processed_batches[object_name] = cell_features_data
    cell_features_data = pd.concat(processed_batches.values(), axis=1)
    rows = len(cell_features_data.index)
    result_meta_columns = ["CentroidCoordinates", "SlideName"]
    feature_count = len(cell_features_data.drop(columns=result_meta_columns).columns)
    cell_features_filename = (
        output_path_filtered / f"filtered_on_n{rows}_nf{feature_count}.csv"
    )
    cell_features_data.to_csv(cell_features_filename)
    print(":: Done. Cell features data:", cell_features_filename, sep="\n")
    return cell_features_data


# TODO: load images from memory into the pipeline without exporting to files separately
# TODO: remove batch loading and refactor error reporting to account for different culprits other than unfinished runs
def _extract_measurements_cellprofiler(
    cell_images_path: Path,
    output_path: Path,
    pipeline_path: Path,
):
    batches = _prepare_data_cellprofiler(cell_images_path=cell_images_path)
    pipeline_output = _run_pipeline_cellprofiler(
        batches=batches, pipeline_path=pipeline_path, output_path=output_path
    )
    _filter_data_cellprofiler(batches, output_path=output_path)


def extract_measurements(
    cell_images_path: Union[str, Path],
    feature_extraction_backend: str,
    output_path: Union[str, Path],
    cell_profiler_pipeline_path: Union[str, Path],
):
    if feature_extraction_backend in CellFeaturesBackend.values():
        extract_measurements_func = globals()[
            f"_extract_measurements_{feature_extraction_backend}"
        ]
        return extract_measurements_func(
            cell_images_path=Path(cell_images_path),
            output_path=Path(output_path),
            pipeline_path=Path(cell_profiler_pipeline_path),
        )
