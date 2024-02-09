# Example Usage:

Note: All Google Cloud Storage file paths need to be updated to match your buckets/filepaths.

### A. Create a vrt from a Google Cloud Storage directory:

Note: This requires setting up a Google access key and putting it in ~/.boto

See: https://cloud.google.com/storage/docs/boto-gsutil
https://stackoverflow.com/questions/60540854/how-to-get-gs-secret-access-key-and-gs-access-key-id-in-google-cloud-storage

```
bash build_input_vrt.sh gs://res-id/ee_exports/sentinel2_10m/ s2_10m.vrt nearest
```

The last argument ('nearest' in this case) is the resampling method. It doesn't matter much for rasters that are already at the resolution we're going to be doing our analysis in, but for rasters with courser resolution (e.g. Landsat or 20m Sentinel bands) cubic can be used (see below)

### B. Extract 200 tiles that are 500x500 with 70 pixel padding (640x640 input images):

*Note: Tiles that fall outside the boundaries of brazil (null-value) will not be saved, so the actual number may be smaller than the specified number*

```
mkdir -p out/s2_10m/
python3 extract_tiles.py s2_10m.vrt 200 500 500 70 70 ./out/s2_10m/ --out_prefix='eg_tile_'
```

### C. Extract prelabels from MapBiomas Agua Waterbodies (optional)

If we want to use MapBiomas Agua (or another dataset) classifications as a starting point for our annotations, we can extract matching tiles for our annotation dataset.

```
# MapBiomas Agua Water Bodies
bash build_input_vrt.sh gs://res-id/ee_exports/mb_waterbodies_2017/ mb_wb_2017.vrt nearest
mkdir -p out/mb_wb_2017/
python3 match_tiles.py ./out/s2_10m/grid_indices.csv mb_wb_2017.vrt ./out/mb_wb_2017/ mb_wb_2017 --center_pixels_num 500
```

*Note: This script can also be used to extract data from another satellite (e.g. Landsat) to match the data saved in step B. For example:*
```
# Landsat 8
bash build_input_vrt.sh gs://res-id/ee_exports/landsat8_30m_v2/2017/ ls8_2017.vrt cubic
mkdir -p out/ls8_2017/
python3 match_tiles.py ./out/s2_10m/grid_indices.csv ls8_2017.vrt ./out/ls8_2017/ ls8_2017
```

### D. Create Labelbox Project

First must install Labelbox Python API
```
pip install labelbox labelbox[data]
```

Then, generate API key and save it in this directory as a file named "lb_api_key.txt". Follow directions for creating the API key here: https://docs.labelbox.com/reference/create-api-key

Next, edit the file "label_ontology.py" as needed to match the classes you are labeling. For more information, see: https://docs.labelbox.com/reference/ontology

Now you're ready to create your project. Replace "MyProjectName" with your desired name.
```
python3 create_lb_project.py MyProjectName
```

Make note of the project ID that is generated, you will need it for the next step.


### E. Upload Data to Labelbox

Now that we've created a project, we can upload data and prepare it for labeling:

```
python3 create_lb_dataset.py clrz3trt6068t0716hir65e5y MapBiomasDemoV2 ./out/s2_10m/ out/s2_10m/grid_indices.csv gs://res-id/labelbox_tiles/mb_demo_v2/
```


### F. Upload prelabels to Labelbox (optional)

If you followed Step C, we can now upload those prelabels to Labelbox
   
```
python3 upload_prelabels.py <project-id> MyDataset ./out/mb_wb_2017/
```

