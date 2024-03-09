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
import shutil

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


def calc_bands_min_max(input_dir, output_dir):
    mask_images = glob.glob(os.path.join(input_dir, '*mask.png'))
    mask_images.sort()
    image_patterns = [mi.replace('mask.png', '') for mi in mask_images]
    band_mins = []
    band_maxes = []

    for image_base in image_patterns:
        ar = io.imread('{}og.tif'.format(image_base))
        img_min = np.min(ar, axis=(0, 1))
        img_max = np.max(ar, axis=(0, 1))
        
        band_mins += [img_min]
        band_maxes += [img_max]

    all_mins = np.stack(band_mins)
    all_maxes = np.stack(band_maxes)
    bands_min_max_all_imgs = np.stack([all_mins, all_maxes], axis=0)
    np.save(os.path.join(output_dir, 'bands_min_max.npy'), bands_min_max_all_imgs)

    return bands_min_max_all_imgs



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

def rescale_to_minmax_uint8(img, bands_min_max):
    img = np.where(img > bands_min_max[1], bands_min_max[1], img)
    img  = (255. * (img.astype('float64') - bands_min_max[0]) / (bands_min_max[1] - bands_min_max[0]))
    img = np.round(img)
    if img.max()>255:
        print(img.max())
        print('Error: overflow')
    return img.astype(np.uint8)


def select_bands_write_images(fp_base, out_path, bands_min_max,
                              band_selection):
    ar = io.imread('{}og.tif'.format(fp_base))

    ar = rescale_to_minmax_uint8(ar, bands_min_max)[:, :, band_selection]

    io.imsave(out_path, ar, plugin='pil', compression='tiff_lzw')

    return ar.reshape((-1, len(band_selection)))


def save_images(img_list, input_dir, output_dir,
                bands_min_max, band_selection=None, calc_mean_std=False):
    if band_selection is None:
        band_selection = np.arange(
                io.imread('{}og.tif'.format(img_list[0])).shape[2]
                )
        print(band_selection)

    if calc_mean_std:
        n = 0
        mean = np.zeros(len(band_selection))
        sums = np.zeros(len(band_selection))
        M2 = np.zeros(len(band_selection))

    for fp_base in img_list:
        # out_path is a little tricky, need to remove _ at end and add in .tif
        out_path = os.path.join(
                output_dir,
                os.path.basename(fp_base)
                )[:-1] + '.tif'
        vals = select_bands_write_images(fp_base, out_path, bands_min_max,
                                         band_selection)
        if calc_mean_std:
            n += vals.shape[0]
            vals = vals
            sums += np.sum(vals, axis=0)
            delta = vals - mean
            mean += np.sum(delta/n, axis=0)
            M2 += np.sum(delta*(vals - mean), axis=0)

    if calc_mean_std:
        return sums/n, np.sqrt(M2 / (n - 1))


def save_masks(ann_list, input_dir, output_dir):
    for fp_base in ann_list:
        fp = '{}mask.png'.format(fp_base)
        ar = io.imread(fp)

        # Save
        out_path = os.path.join(
                output_dir,
                os.path.basename(fp)
                ).replace('_mask.png', '.tif')
        io.imsave(out_path, ar, plugin='pil', compression='tiff_lzw')

    return 


def list_and_split_imgs(input_dir, val_frac, test_frac):
    # First get list of images
    mask_images = glob.glob(os.path.join(input_dir, '*mask.png'))
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
    # Get commandline args
    parser = argparse_init()
    args = parser.parse_args()

    # Create new directory structure for outputs
    for mid_dir in ['img_dir', 'ann_dir']:
        for end_dir in ['train', 'val', 'test']:
            new_dir = os.path.join(args.out_dir, mid_dir, end_dir)
            Path(new_dir).mkdir(parents=True, exist_ok=True)

    # Calculate mins and maxs of bands for scaling
    bands_min_max_all_imgs = calc_bands_min_max(args.in_dir, args.out_dir)
    # Removes top end, since some max values are outliers
    # Can experiment with different values for this
    bands_min_max = np.array([np.min(bands_min_max_all_imgs[0], axis=0),
                              np.percentile(bands_min_max_all_imgs[1], 80, axis=0)])

    # Split in train/val/test
    train_names, val_names, test_names = list_and_split_imgs(
            args.in_dir, val_frac=VAL_FRACTION, test_frac=TEST_FRACTION)

    # Save train/val/test imgs, and calc mean and std from train for scaling
    means_std = save_images(
            train_names,
            args.in_dir,
            os.path.join(args.out_dir, 'img_dir', 'train'),
            bands_min_max,
            calc_mean_std=True,
            band_selection=BAND_SELECTION
            )
    save_images(
            test_names,
            args.in_dir,
            os.path.join(args.out_dir, 'img_dir', 'test'),
            bands_min_max,
            calc_mean_std=False,
            band_selection=BAND_SELECTION
            )
    save_images(
            val_names,
            args.in_dir,
            os.path.join(args.out_dir, 'img_dir', 'val'),
            bands_min_max,
            calc_mean_std=False,
            band_selection=BAND_SELECTION
            )

    np.save(os.path.join(args.out_dir, 'mean_std.npy'), np.vstack(means_std))

    # Save mask images
    save_masks(train_names,
               args.in_dir,
               output_dir=os.path.join(args.out_dir, 'ann_dir', 'train')
               )
    save_masks(test_names,
               args.in_dir,
               output_dir=os.path.join(args.out_dir, 'ann_dir', 'test')
               )
    save_masks(val_names,
               args.in_dir,
               output_dir=os.path.join(args.out_dir, 'ann_dir', 'val')
               )

    # Zip archive
    shutil.make_archive('{}'.format(args.out_dir.strip(os.sep)), 'zip', args.out_dir)

    return


if __name__ == '__main__':
    main()
