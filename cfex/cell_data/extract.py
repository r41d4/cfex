import pandas as pd
import numpy as np
import typing as t
import io

from cfex.enums import CellDataFormat


def format_feature_name(feature_name: str) -> str:
    """
    Build new feature name in Pascal case format.

    Example: "Nucleus: circularity" -> "NucleusCircularity"

    Returns a formatted feature name string.

    Parameters
    ----------
    feature_name : str
        Name of the feature to be formatted.

    Returns
    -------
    str
        Formatted feature name string.

    >>> format_feature_name("Nucleus: circularity")
    "NucleusCircularity"
    """
    capitalized_name = []
    for word in feature_name.split():
        if word[0].isascii():
            word = word.capitalize()
        capitalized_name.append(word)
    capitalized_name = " ".join(capitalized_name)
    return "".join([letter for letter in capitalized_name if letter.isalnum()])


def _extract_cell_measurements_qupath(data: t.Union[str, t.Dict]) -> pd.DataFrame:
    cell_measurements = []
    data = pd.read_json(data)
    for extracted_cell_data in data["properties"]:
        cell_data = {}
        for measurement in extracted_cell_data.get("measurements"):
            column_name = format_feature_name(measurement["name"])
            cell_data[column_name] = measurement["value"]
        try:
            cell_data["Target"] = extracted_cell_data["classification"]["name"]
        except:
            cell_data["Target"] = np.nan
        cell_measurements.append(cell_data)
    return cell_measurements


def extract_cell_measurements(
    data: t.Union[str, t.Dict], data_format: str
) -> pd.DataFrame:
    """
    Extract cell measurements from data.

    Returns a dataframe with cell measurements as columns.

    Parameters
    ----------
    data : str or dict
        Cell object data containing additional measurements.
    data_format: str
        Name of the supported format.

    Returns
    -------
    DataFrame
        DataFrame containing measurements for each cell.
    """
    if data_format == CellDataFormat.QUPATH.value:
        cell_measurements = _extract_cell_measurements_qupath(data)
    return pd.DataFrame(cell_measurements)


def _extract_cell_polygons_qupath(data: t.Union[str, t.Dict]) -> pd.DataFrame:
    cell_polygons = []
    data = pd.read_json(data)
    data.dropna(inplace=True)
    for _, single_cell_data in data.iterrows():
        cell_data = {}
        cell_data["CellPolygon"] = np.array(
            single_cell_data["geometry"]["coordinates"], dtype=np.int32
        )
        cell_data["NucleusPolygon"] = np.array(
            single_cell_data["nucleusGeometry"]["coordinates"], dtype=np.int32
        )
        cell_polygons.append(cell_data)
    return cell_polygons


def extract_cell_polygons(data: t.Union[str, t.Dict], data_format: str) -> pd.DataFrame:
    """
    Extract cell geometry from data.
    Returns a dataframe with cell polygons and nucleus polygons as columns.

    Parameters
    ----------
    data : str or dict
        Cell object data containing polygons.
    data_format: str
        Name of the supported format.

    Returns
    -------
    DataFrame
        DataFrame containing cell polygons.
    """
    if data_format in CellDataFormat.values():
        # move to separate func because of duplication
        # TODO: replace globals with scope of this module
        extract_cell_polygons_func = globals()[f"_extract_cell_polygons_{data_format}"]
        cell_polygons = extract_cell_polygons_func(data)
    return pd.DataFrame(cell_polygons)


def extract_cell_data(
    data: t.Union[t.IO, str, t.Dict],
    data_format: str,
    extract_measurements: t.Optional[bool] = False,
) -> pd.DataFrame:
    """
    Extract cell measurements and geometry from data.
    Returns a dataframe with cell measurements, cell polygons, nucleus polygons.

    Parameters
    ----------
    data : str or dict
        Cell object data containing polygons.
    data_format: str
        Supported format name.
    extract_measurements: bool, optional, default False
        Flag for extracting existing cell measurements from data.

    Returns
    -------
    DataFrame
        DataFrame containing cell polygons and (optionally) measurements data.
    """
    if isinstance(data, io.IOBase):
        data = data.read()
    cell_data = extract_cell_polygons(data, data_format)
    if extract_measurements:
        cell_measurements = extract_cell_measurements(data, data_format)
        cell_data = pd.concat([cell_data, cell_measurements], axis=1)
    return cell_data
