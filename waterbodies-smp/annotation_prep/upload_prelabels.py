
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload existing prelabels to help with annotation

Usage:
    $ python3 upload_prelabels.py <project-id> <dataset-name> <prelabels-dir>

Example:
    $ python3 upload_prelabels.py <project-id> out/mb_2017/

Note: This file is quite specific to uploading MapBiomas Agua Water Bodies labels.
    If you want to use it for something else, you will need to modify it so that
    the label values and label names match your data and labeling scheme.
"""
import labelbox as lb
import labelbox.types as lb_types
import glob
import PIL.Image as Image
import os
import io
import numpy as np
import uuid
import argparse



LABEL_VALUES_DICT = {
        1: 'Natural',
        2: 'Reservatorios',
        3: 'Hidrelectrica',
        4: 'Mineracao'
        }

FILE_SUFFIX = 'mb_2017.tif'
with open('./lb_api_key.txt') as f:
    lines = f.readlines()
    LB_API_KEY = lines[0].rstrip()


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Upload prelabels to help with annotation',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('project_id',
                   help='Labelbox project ID (for existing project)',
                   type=str)
    p.add_argument('dataset_name',
                   help='Name of Labelbox dataset',
                   type=str)
    p.add_argument('prelabel_dir',
                   help='Path to directory with pngs containing masks',
                   type=str)

    return p


def main():
    # Get commandline args
    parser = argparse_init()
    args = parser.parse_args()

    # initiate Labelbox client
    client = lb.Client(api_key=LB_API_KEY)

    prelabel_list = glob.glob(os.path.join(args.prelabel_dir, '*'))

    global_key_base = args.dataset_name
    label_payloads = []
    for f in prelabel_list:
        basename = os.path.basename(f).replace(FILE_SUFFIX, '')
        global_key = global_key_base + basename
        im_ar = np.asarray(Image.open(f))
        annotations = []
        annotations_exist = False
        for label in range(1, 5):
            label_ar = (im_ar == label)
            if np.sum(label_ar) > 0:
                annotations_exist = True
                pil_im = Image.fromarray(label_ar)
                b = io.BytesIO()
                pil_im.save(b, 'jpeg')
                im_bytes = b.getvalue()
                mask_data = lb.types.MaskData(im_bytes=im_bytes)
                mask_annotation = lb_types.ObjectAnnotation(
                    name=LABEL_VALUES_DICT[label],
                    value=lb_types.Mask(
                        mask=mask_data,
                        color=(255, 0, 0)
                        )
                )
                annotations.append(mask_annotation)

        if annotations_exist:
            label_payloads.append(
                    lb_types.Label(
                        data=lb_types.ImageData(global_key=global_key),
                        annotations=annotations
                                  )
                    )
    # upload prelabel labels for this data row in project
    upload_job = lb.MALPredictionImport.create_from_objects(
        client=client,
        project_id=args.project_id,
        name="prelabel_upload_job" + str(uuid.uuid4()),
        predictions=label_payloads
    )
    upload_job.wait_until_done()

    print(f"Errors: {upload_job.errors}")
    print(f"Status of uploads: {upload_job.statuses}")
