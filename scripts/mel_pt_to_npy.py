#!/usr/bin/env python
'''Convert saved torch mels to numpy arrays to feed to HiFi-GAN'''

import sys
from pathlib import Path

import numpy as np
import torch

if __name__ == '__main__':
    try:
        _, in_dir, out_dir = sys.argv
    except ValueError:
        print("Usage: mel_pt_to_npy.py in_dir out_dir")
        sys.exit()

    npy_dir = Path(out_dir).expanduser()
    npy_dir.mkdir(parents=True, exist_ok=True)

    pt_dir = Path(in_dir).expanduser()
    pt_files = pt_dir.glob('*.pt')

    for pt_file in pt_files:
        mels_pt = torch.load(pt_file)
        mels_npy = mels_pt.cpu().numpy()
        npy_file = npy_dir / pt_file.with_suffix('.npy').name
        np.save(npy_file, mels_npy)

