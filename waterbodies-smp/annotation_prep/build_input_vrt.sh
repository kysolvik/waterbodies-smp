#!/bin/bash
# build_input_vrt.sh
# 
# Build VRT Virtual Raster Dataset for extracting tiles for annotation.
# Example Usage:
#     bash build_input_vrt.sh gs://res-id/ee_exports/sentinel2_10m/ s2_10m.vrt nearest

input_gs_dir=$1
output_file=$2
resampling_method=$3

gsutil ls ${input_gs_dir%/}/*.tif > temp_filelist.txt
sed -i 's!gs://!/vsigs/!' temp_filelist.txt
gdalbuildvrt $output_file -r $resampling_method -tap -tr 0.000089831528412 0.000089831528412 -input_file_list temp_filelist.txt

