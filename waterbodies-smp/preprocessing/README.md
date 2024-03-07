# Preprocessing

After running annotation_prep and using LabelBox to manually label training examples, the scripts in this folder can be used to download the results and prepare the data for training.

NOTE: Be sure to include your LabelBox API key in this directory in a plain txt file named "lb_api_key.txt"

### A. Download LabelBox Results (and Get the Matching Original Images)

```
python3 download_lb_results.py <project-id> out/labelbox_results/
```

To make things easier, the script will also get the matching raw data tiles (that you previously generated with annotation_prep/extract_tiles.py) and copy them to the same output directory. To do so, add the "--original_dir" flag and specify the directory containing the Landsat tiles. For example:

```
python3 download_lb_results.py <project-id> out/labelbox_results/ --original_dir ../annotation_prep/out/s2_10m/
```


### B. Preprocess data for training on LabelBox

