# waterbodies-smp
Water bodies segmentation using Segmentation Models Pytorch

## Code

For each stage, see the README inside the directory for more information.

### 1 - waterbodies-smp/ee_export

Readme includes a links to export scripts on Google Earth Engine (Landsat or Sentinel)


### 2 - waterbodies-smp/annotation_prep

Code for extracting tiles from satellite mosaics for LabelBox annotation

- A. build_input_vrt.sh
Build VRT Virtual Raster Dataset for extracting tiles for annotation.

- B. extract_tiles.py
Extracts random tiles from mosaic. Can control size and count of tiles.

- C. match_tiles.py
Given directory of tiles from "extract_tiles.py", extract matching tiles from other mosaics.
    Useful for extracting data from other satellites and/or from classification maps (for
    example, MapBiomas Agua Water Bodies).

- D. create_lb_project.py
Create Labelbox project and label editor ontology (if necessary). Can also do this from
    the Labelbox web interface.

- E. create_lb_dataset.py
Create Labelbox dataset.

- F. upload_prelabels.py
(Optional): Upload existing labels (e.g. MapBiomas Agua Water Bodies) as starting point
    for labeling.

### 3 - waterbodies-smp/preprocessing

Code for preparing annotated images for training

- prep_smp_dataset.ipynb
Prepare dataset for training


### 4 - waterbodies-smp/train

Includes notebook for training on Google Colab.

- train_smp_segmentation_unet.ipynb
Notebook for training. For running on Google Colab.


### 5 - waterbodies-smp/predict

Scripts for running prediction along with some helper modules

- build_input_vrts_landsat.sh
Build full raster vrts for running predictions

- predict_smp_landsat.py
Script for running prediction

- models.py
Helper module that contains information about model structure

- dataset.py
Helper module that contains information about data for prediction.
