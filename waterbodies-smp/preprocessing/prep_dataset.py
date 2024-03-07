#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perform basic preprocessing and train/val/test split

Usage:
    $ python3 prep_dataset.py <in-dir> <out-dir>

Example:
    $ python3 prep_dataset.py ./out/labelbox_results/ ./out/prepped_data/
"""
import numpy as np
import os
import PIL
import PIL.Image
import glob
from skimage import io
from skimage.transform import resize
import random
import pandas as pd
import subprocess as sp
import argparse
from pathlib import Path

# Can change if needed
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2
random.seed(170)
# By default, select all 4 bands from original images. (B, G, R, NIR)
# Can change to suit your needs
BAND_SELECTION = [0, 1, 2, 3]


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Preprocess and prep dataset for training',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('in_dir',
                   help='Directory with masks (as pngs) and raw data (as tifs)',
                   type=str)
    p.add_argument('out_dir',
                   help='Directory for saving outputs',
                   type=str)

    return p


def split_train_test(img_patterns, test_frac, val_frac):
    """Split data into train, test, val (or just train)

    Returns:
        train_indices, val_indices, test_indices tuple
    """
    total_ims = len(img_patterns)
    if test_frac != 0:

        train_count = round(total_ims * (1 - test_frac - val_frac))
        train_indices = random.sample(range(total_ims), train_count)
        test_val_indices = np.delete(np.array(range(total_ims)), train_indices)

        test_count = round(total_ims * test_frac)
        test_indices = random.sample(list(test_val_indices), test_count)

        if val_frac != 0:
            val_indices = np.delete(np.array(range(total_ims)),
                                    np.append(train_indices, test_indices))

            return train_indices, val_indices, test_indices
        else:
            return train_indices, test_indices
    else:
        return np.arange(total_ims)


def calc_all_mean_std(img_list, band_selection):
    n = 0
    mean = np.zeros(len(band_selection))
    sums = np.zeros(len(band_selection))
    M2 = np.zeros(len(band_selection))
    for fp_base in img_list:
        ar = io.imread('{}og.tif'.format(fp_base))[:, :, band_selection]
        vals = ar.reshape((-1, len(band_selection)))
        n += vals.shape[0]
        sums += np.sum(vals, axis=0)
        delta = vals - mean
        mean += np.sum(delta/n, axis=0)
        M2 += np.sum(delta*(vals - mean), axis=0)
    return np.vstack((sums/n, np.sqrt(M2 / (n - 1))))

def normalize_image(ar, mean_std):
    return (ar - mean_std[0])/mean_std[1]

def select_bands_write_images(fp_base, out_path, band_selection, mean_std):
    ar = io.imread('{}og.tif'.format(fp_base))[:, :, band_selection]

    # Mean std scaling
    ar = normalize_image(ar, mean_std)

    io.imsave(out_path, ar, plugin='pil', compression='tiff_lzw')
    return


def save_images(img_list, input_dir, output_dir, mean_std, 
                band_selection=None):
    if band_selection is None:
        band_selection = np.arange(
                io.imread('{}og.tif'.format(img_list[0])).shape[2]
                )

    for fp_base in img_list:
        # out_path is a little tricky, need to remove _ at end and add in .tif
        out_path = fp_base.replace(input_dir, output_dir)[:-1] + '.tif'
        select_bands_write_images(fp_base, out_path, band_selection, mean_std)


def save_masks(ann_list, input_dir, output_dir):
    for fp_base in ann_list:
        fp = '{}mask.png'.format(fp_base)
        ar = io.imread(fp)

        # Save
        out_path = fp.replace(input_dir, output_dir).replace('_mask.png', '.tif')
        io.imsave(out_path, ar, plugin='pil', compression='tiff_lzw')

    return 

def list_and_split_imgs(input_dir, val_frac, test_frac):
    # First get list of images
    mask_images = glob.glob('{}*mask.png'.format(input_dir))
    mask_images.sort()
    image_patterns = np.array([mi.replace('mask.png', '') for mi in mask_images])

    train_indices, val_indices, test_indices = split_train_test(
            image_patterns,
            test_frac=test_frac,
            val_frac=val_frac)
    train_basename_list = image_patterns[train_indices]
    val_basename_list = image_patterns[val_indices]
    test_basename_list = image_patterns[test_indices]

    return train_basename_list, val_basename_list, test_basename_list

def main():
    # Make temp data dir
    Path('./temp_data').mkdir(parents=False, exist_ok=True)

    # Get commandline args
    parser = argparse_init()
    args = parser.parse_args()

    # Create new directory structure for outputs
    for mid_dir in ['img_dir', 'ann_dir']:
        for end_dir in ['train', 'val', 'test']:
            new_dir = os.path.join(args.out_dir, mid_dir, end_dir)
            Path(new_dir).mkdir(parents=True, exist_ok=True)

    # Split in train/val/test
    train_names, val_names, test_names = list_and_split_imgs(
            args.in_dir, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION)

    # Save train/val/test imgs, and calc mean and std from train for scaling
    mean_std = calc_all_mean_std(train_names, band_selection=BAND_SELECTION)
    print(mean_std)
    save_images(
            train_names,
            args.in_dir,
            '{}/img_dir/train/'.format(args.out_dir),
            mean_std=mean_std,
            band_selection=BAND_SELECTION
            )
    save_images(
            test_names,
            args.in_dir,
            '{}/img_dir/test/'.format(args.out_dir),
            mean_std=mean_std,
            band_selection=BAND_SELECTION
            )
    save_images(
            val_names,
            args.in_dir,
            '{}/img_dir/val/'.format(args.out_dir),
            mean_std=mean_std,
            band_selection=BAND_SELECTION
            )

    np.save('./mean_std.npy', np.vstack(mean_std))

    # Save mask images
    save_masks(train_names,
               args.in_dir,
               output_dir='{}/ann_dir/train/'.format(args.out_dir))
    save_masks(test_names,
               args.in_dir,
               output_dir='{}/ann_dir/test/'.format(args.out_dir))
    save_masks(val_names,
               args.in_dir,
               output_dir='{}/ann_dir/val/'.format(args.out_dir))


if __name__ == '__main__':
    main()
