#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download all labeled masks from LabelBox

Usage:
    $ python3 download_lb_results.py <project-id> <out-dir>

Example:
    $ python3 download_lb_results.py <project-id> out/labels/
"""
import labelbox as lb
import glob
import PIL.Image as Image
import os
import io
import numpy as np
import uuid
import argparse
from pathlib import Path
import urllib
import shutil

IMAGE_SHAPE = (500, 500)

with open('./lb_api_key.txt') as f:
    lines = f.readlines()
    LB_API_KEY = lines[0].rstrip()


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Download labeled masks from LabelBox',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('project_id',
                   help='Labelbox project ID (for existing project)',
                   type=str)
    p.add_argument('out_dir',
                   help='Path to output directory for saving masks',
                   type=str)
    p.add_argument('--original_dir',
                   help=('Path to directory with original tifs matching masks. '
                         'Optional, if given copies matching tifs to out_dir.'),
                   type=str,
                   default=None)

    return p

def get_object_info(json_row, project_id):
    name = os.path.basename(json_row['data_row']['row_data'])
    object_list = json_row['projects'][project_id]['labels'][0]['annotations']['objects']
    if len(object_list) == 0:
        mask_dict = {
            'name': name,
            'contains_water': False,
            'mask_url': None,
            'object_values_dict': None
        }
    else:
        mask_url = object_list[0]['composite_mask']['url']
        values_color_list = []
        for o in object_list:
            values_color_list.append(
                {o['value']: o['composite_mask']['color_rgb']})
        mask_dict = {
            'name': name,
            'contains_water': True,
            'mask_url': mask_url,
            'values_color_list': values_color_list
        }
    return mask_dict

def download_mask(client, mask_dict, label_values_dict):
    masks_list = []
    req = urllib.request.Request(mask_dict['mask_url'], headers=client.headers)
    image = Image.open(urllib.request.urlopen(req))
    image_ar = np.array(image)
    for name_color_pair in mask_dict['values_color_list']:
        target_color = list(name_color_pair.values())[0]
        target_name = list(name_color_pair.keys())[0]
        mask = np.all(image_ar[:,:,:-1] == target_color, axis=-1)
        mask = (mask * label_values_dict[target_name]).astype('uint8')
        masks_list.append(mask)
    full_mask = np.max(masks_list,axis=0).astype('uint8')

    return full_mask

def write_mask(full_mask, out_dir, mask_dict):
    out_file = os.path.join(out_dir, mask_dict['name']).replace('_rgb.png', '_mask.png')
    im = Image.fromarray(full_mask)
    im.save(out_file)

    return os.path.basename(out_file)

def copy_original_tif(og_dir, out_dir, mask_name):
    og_path = os.path.join(og_dir, mask_name.replace('_mask.png', '_og.tif'))
    shutil.copy(og_path, out_dir)

    return


def main():
    # Get commandline args
    parser = argparse_init()
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # initiate Labelbox client
    client = lb.Client(api_key=LB_API_KEY)
    project = client.get_project(args.project_id)

    # Create export task and get result as json
    # NOTE: In this case, getting images labeled as "Done" or "InReview".
    # Can edit if only want one or the other. 
    # Another possible status is "InRework"
    export_json = []
    for status in ["Done", "InReview"]:
        status_filter = {
          "workflow_status": status 
        }
        export_task = project.export_v2(filters=status_filter)
        export_task.wait_till_done()
        export_json_temp = export_task.result
        export_json = export_json + export_json_temp

    # Get ontology information (label values)
    ontology_info = project.ontology()
    object_type_names = [t.name.lower() for t in ontology_info.tools()]
    object_type_vals = np.arange(1, len(object_type_names)+1)
    label_values_dict = dict(zip(object_type_names, object_type_vals))

    # Download masks
    for json_row in export_json:
        mask_dict = get_object_info(json_row, args.project_id)
        if mask_dict['contains_water']:
            full_mask = download_mask(client, mask_dict, label_values_dict)
            mask_name = write_mask(full_mask, args.out_dir, mask_dict)
        else:
            write_mask(np.zeros(IMAGE_SHAPE, dtype='uint8'), args.out_dir, mask_dict)
            mask_name = write_mask(full_mask, args.out_dir, mask_dict)
        if args.original_dir is not None:
            copy_original_tif(args.original_dir, args.out_dir, mask_name)


if __name__ == '__main__':
    main()
