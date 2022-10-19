import typing as t
from cv2 import boundingRect
import numpy as np


def calculate_centroid(object_polygon: t.Sequence) -> np.ndarray:
    """
    Calculate centroid coordinate of a cell - mean of polygon points coordinates.

    Returns an array with x, y coordinates of the cell centroid.

    Parameters
    ----------
    object_polygon : array-like
        Two-dimensional array containing x, y coordinates of a cell polygon.

    Returns
    -------
    ndarray
        Two-element array with x, y coordinates of the cell centroid.

    >>> polygon = [[0., 3., 0.], [0., 2., 4.]]
    >>> calculate_centroid(polygon)
    array([1., 2.])
    """
    return np.mean(object_polygon, axis=1).flatten()


def calculate_bound_transform_coordinates(
    roi_bounding_box: t.Sequence, point_coordinates: t.Sequence
) -> t.Tuple[int]:
    """
    Calculate the coordinates of a point in a coordinate system where
    the origin is at the upper-left corner of a given ROI.

    Returns a tuple with transformed x, y coordinates of the cell point 
    relative to the local coordinate system of the bounding box.

    Parameters
    ----------
    roi_bounding_box : array-like
        Sequence, first two numbers of which are x, y coordinates of 
        an upper left corner of a bounding box.
    point_coordinates: array-like
        Two-element sequence with x, y coordinates of the cell point.

    Returns
    -------
    tuple
        Two-element array with transformed x, y coordinates of the cell point.
    """
    bound_x, bound_y, _, _ = roi_bounding_box
    origin_x, origin_y = point_coordinates
    transformed_x = origin_x - bound_x
    transformed_y = origin_y - bound_y
    return transformed_x, transformed_y


def calculate_roi_bounding_box(object_polygons: t.Sequence) -> t.Tuple[int]:
    """
    Calculate a minimum bounding box for all objects with given polygons.

    Returns a tuple with x, y coordinate of a bounding box (relative to WSI origin coordinates), 
    its width and its height.

    Parameters
    ----------
    object_polygons : array-like
        Sequence of two-dimensional arrays with x and y coordinates of polygons.

    Returns 
    -------
    tuple
        Tuple with x and y coordinate of the upper left corner of the bounding box, 
        its width and its height. 
    """
    X = []
    Y = []
    for contour in object_polygons:
        # TODO: reimplement without boundingRect
        x, y, _, _ = boundingRect(contour)
        X.append(x)
        Y.append(y)
    roi_bounding_box = (
        np.min(X),
        np.min(Y),
        np.max(X) - np.min(X) + 200,
        np.max(Y) - np.min(Y) + 200,
    )
    return roi_bounding_box


def calculate_cell_roi_bounding_box(cell_point_coordinates: t.Sequence, bounding_box_margin: int) -> t.Tuple[int]:
    """
    Calculate a bounding box for a cell point with a given distance margin.

    Returns a tuple with x, y coordinate of a bounding box (relative to WSI origin coordinates), 
    its width and its height.

    Parameters
    ----------
    cell_point_coordinates : array-like
        Sequence consisting of x and y coordinates of cell centroid.
    bounding_box_margin : int
        Distance between the cell centroid and the edge of the intended bounding box.

    Returns
    ------
    tuple
        Tuple with x, y coordinate of the upper left corner of the bounding box, 
        its width and its height. 
    """
    cell_point_x, cell_point_y = cell_point_coordinates
    return (
        cell_point_x - bounding_box_margin,
        cell_point_y - bounding_box_margin,
        bounding_box_margin * 2,
        bounding_box_margin * 2,
    )


def calculate_image_center(image: np.ndarray) -> t.Tuple[int]:
    """
    Calculate coordinates of the center of the image.

    Returns a tuple with x, y coordinates of the image center.

    Parameters
    ----------
    image : ndarray
        Two-dimensional array with image data.

    Returns
    ------
    tuple
        Tuple with x, y coordinate of the image center.
    """
    image_dimensions = image.shape
    image_center_x = image_dimensions[0] // 2
    image_center_y = image_dimensions[1] // 2
    return image_center_x, image_center_y
