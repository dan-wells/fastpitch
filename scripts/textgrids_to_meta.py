#!/usr/bin/env python

import argparse
import glob
import os
import sys

import numpy as np
import tgt
import tqdm


# TODO: this is also defined in extract_mels.py, pull out to common utils?
def parse_textgrid(tier, sampling_rate, hop_length):
    # latest MFA replaces silence phones with "" in output TextGrids
    sil_phones = ["sil", "sp", "spn", ""]
    start_time = tier[0].start_time
    end_time = tier[-1].end_time
    phones = []
    durations = []
    for i, t in enumerate(tier._objects):
        s, e, p = t.start_time, t.end_time, t.text
        if p not in sil_phones:
            phones.append(p)
        else:
            if (i == 0) or (i == len(tier) - 1):
                # leading or trailing silence
                phones.append("sil")
            else:
                # short pause between words
                phones.append("sp")
        durations.append(int(np.ceil(e * sampling_rate / hop_length)
                             - np.ceil(s * sampling_rate / hop_length)))
    return phones, durations, start_time, end_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract transcripts from TextGrid files and write to '
                    'wav file list for extract_mels.py.')
    parser.add_argument('textgrid_dir', type=str,
        help='Directory containing TextGrids to process.')
    parser.add_argument('meta_out', type=str,
        help='Output metadata file to write.')
    parser.add_argument('-sr', '--sampling_rate', type=int, default=22050,
        help='Sampling rate of aligned audio files. Default 22050 Hz.')
    parser.add_argument('--hop_length', type=int, default=256,
        help='Hop length used when extracting mel spectrogram frames. Default 256.')
    args = parser.parse_args()

    with open(args.meta_out, 'w') as outf:
        tg_files = glob.glob(os.path.join(args.textgrid_dir, '*.TextGrid'))
        for tgf in tqdm.tqdm(tg_files):
            try:
                textgrid = tgt.io.read_textgrid(tgf, include_empty_intervals=True)
            except FileNotFoundError:
                # presumably this file did not align successfully
                continue
            phones, durations, start, end = parse_textgrid(
                textgrid.get_tier_by_name("phones"),
                args.sampling_rate,
                args.hop_length)
            # almost certainly wrong given MFA filenames: fix it after
            wav_file = tgf.replace('TextGrid', 'wav')
            outf.write('{}|{}\n'.format(wav_file, ' '.join(phones)))

