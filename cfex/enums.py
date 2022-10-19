from enum import Enum


class ListedEnum(Enum):
    """
    Provides a simple way to list the values of its members.
    """

    @classmethod
    def values(cls):
        return [member.value for member in cls]


class CellDataFormat(ListedEnum):
    """
    Enumerates cell measurement data formats
    available for import into the Feature Extractor.

    QUPATH
        GeoJSON data exported from QuPath cell detection objects (non-feature collection).
    """

    QUPATH = "qupath"


class CellImageLoadBackend(ListedEnum):
    """
    Enumerates names of components that serve as a backend for loading the WSI or its parts.
    """

    SLIDEIO = "slideio"


class CellDetectionBackend(ListedEnum):
    """
    Enumerates names of components that serve as a backend for cell instance segmentation.
    """

    STARDIST = "stardist"


class CellFeaturesBackend(ListedEnum):
    """
    Enumerates names of components that serve as a backend for extracting cell measurements to be used as features.
    """

    CELLPROFILER = "cellprofiler"


class CellKidneyTumorGradeLabel(Enum):
    """
    Enumerates label colors for grades of kidney tumors.

    F - blue,
    G1 - green,
    G2 - yellow,
    G3 - orange,
    G4 - pink.
    """

    F = (0, 0, 255)
    G1 = (100, 255, 0)
    G2 = (255, 255, 0)
    G3 = (255, 100, 0)
    G4 = (255, 100, 100)
