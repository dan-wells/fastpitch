#!/usr/bin/env python

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm


def convert_meta(fs_meta_dir, out_meta_dir, symbols='phone'):
    stats_meta = os.path.join(fs_meta_dir, 'stats.json')
    stats = load_json(stats_meta)
    _, _, pitch_mean, pitch_std = stats['pitch']
    fp_stats = {'mean': pitch_mean, 'std': pitch_std}
    fp_stats_f = os.path.join(out_meta_dir, 'pitch_{}_stats.json'.format(symbols))
    if not os.path.exists(out_meta_dir):
        os.makedirs(out_meta_dir)
    with open(fp_stats_f, 'w') as f:
        json.dump(fp_stats, f)

    spkr_meta = os.path.join(fs_meta_dir, 'speakers.json')
    spkr_ids = load_json(spkr_meta)

    train_meta = 'train.txt'
    val_meta = 'val.txt'
    for meta in ['train.txt', 'val.txt']:
        in_meta = os.path.join(fs_meta_dir, meta)
        if os.path.exists(in_meta):
            out_meta = os.path.join(out_meta_dir, meta)
            with open(in_meta) as inf, open(out_meta, 'w') as outf:
                for line in tqdm(inf):
                    utt_id, speaker, phones, text = line.strip().split('|')
                    if symbols == 'phone':
                        out_text = phones.strip('{}')
                    else:
                        out_text = text

                    mel_path = convert_data(fs_meta_dir, out_meta_dir, 'mel', speaker, utt_id)
                    dur_path = convert_data(fs_meta_dir, out_meta_dir, 'duration', speaker, utt_id)
                    pitch_path = convert_data(fs_meta_dir, out_meta_dir, 'pitch', speaker, utt_id)

                    if len(spkr_ids) > 1:
                        outf.write('{}|{}|{}|{}|{}\n'.format(
                            mel_path, dur_path, pitch_path, out_text, spkr_ids[speaker]))
                    else:
                        outf.write('{}|{}|{}|{}\n'.format(
                            mel_path, dur_path, pitch_path, out_text))
            

def convert_data(in_dir, out_dir, data_type, speaker, utt_id):
    fs2_path = os.path.join(in_dir, data_type, 
                            '{}-{}-{}.npy'.format(speaker, data_type, utt_id))
    fs2_data = np.load(fs2_path)
    if data_type == 'mel':
        # should be [bins x len]
        fs2_data = fs2_data.T

    fp_data_path = os.path.join(data_type, '{}-{}.pt'.format(speaker, utt_id))
    full_fp_path = os.path.join(out_dir, fp_data_path)

    fp_data_subdir = os.path.join(out_dir, data_type)
    if not os.path.exists(fp_data_subdir):
        os.makedirs(fp_data_subdir)

    fp_data = torch.from_numpy(fs2_data)
    torch.save(fp_data, full_fp_path)

    return fp_data_path


def load_json(json_file):
    with open(json_file) as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fs_meta_dir', type=str, help='Directory holding preprocessed data files from FastSpeech2.')
    parser.add_argument('out_meta_dir', type=str, help='Output directory for FastPitch data files.')
    parser.add_argument('--symbols', type=str, choices=['char', 'phone'], default='phone',
                        help='Input symbol type, either phone or char (text).')
    args = parser.parse_args()

    convert_meta(args.fs_meta_dir, args.out_meta_dir, args.symbols)
