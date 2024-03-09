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


### B. Preprocess data for training on Colab

Next, we'll do some basic preprocessing steps, including calculating the mean and std of the images (for scaling, which helps training) and splitting the data into train/validation/test sets. 

This assumes you ran "download_lb_results.py" with the "--original_dir" option so that all the images and masks are in the same location.

```
python3 prep_dataset.py out/labelbox_results/ prepped_data/
```


### C. Upload files needed for training on Google Colab

Next we'll upload the files we need to Google Cloud Storage so that we can easily access them on Google Colab. We have to also make them public so we can read them from Colab

```
gcloud storage cp prepped.zip gs://res-id/cnn/training/
gcloud storage objects update gs://res-id/cnn/training/prepped_data.zip  --add-acl-grant=entity=AllUsers,role=READER
```

