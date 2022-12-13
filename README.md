# Cell Feature Extractor

**CFEX** is a free open-source utility providing a number of morphometric features extracted from segmented nuclei in a tissue image to be used in your research without any prior annotation work or model training.

## Overview

Program takes two files as input in order to extract features from cell nuclei:

- image of H&E stained tissue in TIFF or SVS format;
- ROI metadata file for specifying the ROI in a given image;

You can find examples of input files under `extra/data_examples` directory.
Providing the metadata file is optional - omitting it will result in the processing of the whole image.

Given an image file `scan.svs` and ROI metadata file `metadata.json`, CFEX can be launched by executing `cfex` with options:

```bash
cfex -w scan.svs -d metadata.json 
```

Or using a protocol file `protocol.json`:

```bash
cfex -p protocol.json
```

By default CFEX outputs a .csv in a directory where it was ran, containing cell nuclei morphometric features as columns in row-ordered cell indices. Default filename of the output file contains a name of the original image file, number of detected nuclei in the image, a number of features extracted from each. You can specify your own output path using `-o` option.

## Installation

### Prerequisites

CFEX currently uses [CellProfiler](https://cellprofiler.org) on the backend as an external dependency, therefore make sure that it is installed in your system.

### Running as a Python package

From PyPI:

```bash
pip install cfex
```

## Interfacing with QuPath

You can provide exported .geojson files with cell object data as input to CFEX. Cell object data can be provided in two forms:

- point data – a set of points representing approximate centers of cell nuclei
- polygon data – a set of arrays representing coordinates of cell polygon's vertices (e.g. if you used some form of instance segmentation)
- ROI data – coordinates of a ROI, in which case features will be extracted from all the cells inside the specified region.

You can find examples of input data under `examples` directory.
