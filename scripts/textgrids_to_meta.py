#!/usr/bin/env python

import argparse
import glob
import os

import tgt
import tqdm


def parse_textgrid_tier(tier, skip_sil=False):
    # latest MFA replaces silence phones with '' in output TextGrids
    sil_phones = ['sil', 'sp', 'spn', '']
    symbols = []
    for n, interval in enumerate(tier._objects):
        sym = interval.text
        if sym not in sil_phones:
            symbols.append(sym)
        elif not skip_sil:
            if (n == 0) or (n == len(tier) - 1):
                symbols.append('sil')  # leading or trailing silence
            else:
                symbols.append('sp')  # short pause between words
    return symbols


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract transcripts from TextGrid files and write to '
                    'wav file list for prepare_dataset.py.')
    parser.add_argument('textgrid_dir', type=str,
        help='Directory containing TextGrids to process.')
    parser.add_argument('meta_out', type=str,
        help='Output metadata file to write.')
    parser.add_argument('--tier', type=str, choices=['phones', 'words'], default='phones',
        help='Tier to extract symbols from')
    parser.add_argument('--skip-sil', action='store_true',
        help='Exclude silence phones from output')
    args = parser.parse_args()

    with open(args.meta_out, 'w') as outf:
        outf.write('audio|text\n')
        tg_files = sorted(glob.glob(os.path.join(args.textgrid_dir, '*.TextGrid')))
        for tgf in tqdm.tqdm(tg_files):
            try:
                textgrid = tgt.io.read_textgrid(tgf, include_empty_intervals=True)
            except FileNotFoundError:
                continue  # assume failed alignment
            symbols = parse_textgrid_tier(
                textgrid.get_tier_by_name(args.tier), args.skip_sil)
            # note: TextGrids from MFA might not match original audio filenames
            wav_file = tgf.replace('TextGrid', 'wav')
            wav_file = os.path.join('wavs', os.path.basename(wav_file))
            outf.write('{}|{}\n'.format(wav_file, ' '.join(symbols)))
