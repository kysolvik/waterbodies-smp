#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create Labelbox Dataset, upload images, and queue batch.

Usage:
    $ python3 create_lb_dataset.py <project-id> <dataset-name> <image-dir> <gs-storage-path>

Example:
    $ python3 create_lb_dataset.py clsdplamz05ip07ynagk8290v  MyDemoDataset ./out/s2_10m/ gs://res-id/labelbox_tiles/mb_demo/

Note: Assumes you have both rgb and ndwi pngs stored in out/s2_10m/

IMPORTANT: Anything inside the gs-storage-path you specify will be made publically readable!
"""
import labelbox as lb
import glob
import os
import argparse
import subprocess as sp
import pandas as pd

with open('lb_api_key.txt') as f:
    lines = f.readlines()
    LB_API_KEY = lines[0].rstrip()


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Create dataset and upload images for annotating',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('project_id',
                   help='Labelbox project ID (for existing project)',
                   type=str)
    p.add_argument('dataset_name',
                   help='Desired name for dataset',
                   type=str)
    p.add_argument('image_dir',
                   help='Path to directory with pngs for annotation',
                   type=str)
    p.add_argument('grid_indices_csv',
                   help='Path to grid_indices csv produced by extract_tiles.py',
                   type=str)
    p.add_argument('gs_storage_path',
                   help='Path to cloud storage directory for images',
                   type=str)

    return p


def main():
    # Get commandline args
    parser = argparse_init()
    args = parser.parse_args()

    # initiate Labelbox client
    client = lb.Client(api_key=LB_API_KEY)

    # Create new dataset
    dataset = client.create_dataset(name=args.dataset_name)

    rgb_list = glob.glob(os.path.join(args.image_dir, '*rgb.png'))

    # Upload files to google cloud
    if args.gs_storage_path[-1] != '/':
        gs_storage_path = args.gs_storage_path + '/'
    else:
        gs_storage_path = args.gs_storage_path
    sp.call(['gsutil', '-m', 'cp', os.path.join(args.image_dir, '*.png'),
             gs_storage_path])
    sp.call(['gcloud', 'storage', 'objects', 'update', '--recursive',
             gs_storage_path,
             '--add-acl-grant=entity=AllUsers,role=READER'])
    storage_url = "https://storage.googleapis.com/" + gs_storage_path[5:]
    global_key_base = args.dataset_name
    grid_df = pd.read_csv(args.grid_indices_csv)
    assets = []
    for f in rgb_list:
        rgb_basename = os.path.basename(f)
        ndwi_basename = rgb_basename.replace('rgb.png', 'ndwi.png')
        raw_name = rgb_basename[:-8]
        gmaps_url = grid_df.loc[grid_df['name']==raw_name, 'gmaps_link'].values[0]
        asset = {
            "row_data": storage_url + rgb_basename,
            "media_type": "IMAGE",
            "global_key": global_key_base + rgb_basename,

            "attachments": [
                {"type": "IMAGE_OVERLAY",
                 "value": storage_url + ndwi_basename
                 },
                {"type": "RAW_TEXT",
                 "value": gmaps_url}]
        }
        assets.append(asset)

    task = dataset.create_data_rows(assets)
    task.wait_till_done()
    print('Data upload errors: ', task.errors)

    # Create batch
    project = client.get_project(args.project_id)
    task = project.create_batches_from_dataset(
            name_prefix=args.dataset_name + " batch-",
            dataset_id=dataset.uid,
            priority=1)
    task.wait_till_done()
    print("Data queue result: ", task.result())

    return


if __name__ == '__main__':
    main()
