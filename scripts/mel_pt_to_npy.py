#!/usr/bin/env python
'''Convert saved torch mels to numpy arrays to feed to HiFi-GAN'''

import glob
import os
import sys

import numpy as np
import torch

if __name__ == '__main__':
    try:
        _, in_dir, out_dir = sys.argv
    except ValueError:
        print("Usage: mel_pt_to_npy.py in_dir out_dir")
        sys.exit()

    npy_dir = os.path.expanduser(out_dir)
    os.makedirs(npy_dir, exist_ok=True)

    pt_dir = os.path.expanduser(in_dir)
    pt_files = glob.glob(os.path.join(pt_dir, '*.pt'))

    for pt_file in pt_files:
        mels_pt = torch.load(pt_file)
        mels_npy = mels_pt.cpu().numpy()
        stem = os.path.splitext(os.path.basename(pt_file))[0]
        npy_file = os.path.join(npy_dir, stem + '.npy')
        np.save(npy_file, mels_npy)

