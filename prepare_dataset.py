# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import json
import os

import torch
import dllogger as DLLogger
import numpy as np
from dllogger import StdOutBackend, Verbosity
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastpitch.data_function import TextMelAliLoader


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelists', required=True, nargs='+',
                        type=str, help='Path to file with audio paths and text')
    parser.add_argument('--n-speakers', default=1, type=int,
                        help='Number of speakers labelled in data')
    parser.add_argument('--extract-mels', action=argparse.BooleanOptionalAction,
                        default=True, help='Save mel spectrograms to disk')
    parser.add_argument('--extract-durs', action=argparse.BooleanOptionalAction,
                        default=True, help='Save input symbol durations to disk')
    parser.add_argument('--extract-pitch', action=argparse.BooleanOptionalAction,
                        default=True, help='Save framewise, unnormalized pitch values to disk')
    parser.add_argument('--n-workers', default=4, type=int,
                        help='Number of parallel threads for data processing')
    parser.add_argument('--write-meta', action='store_true',
                        help='Write metadata file pointing to extracted features')
    parser.add_argument('--input-type', type=str, default='char',
                        choices=['char', 'phone', 'unit'],
                        help='Input symbols used, either char (text), phone '
                        'or quantized unit symbols.')
    parser.add_argument('--symbol-set', type=str, default='english_basic',
                        help='Define symbol set for input text')
    parser.add_argument('--text-cleaners', nargs='*',
                        default=[], type=str,
                        help='Type of text cleaners for input text')
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--peak-norm', action='store_true',
                        help='Apply peak normalization to audio')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=512, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=512, type=int,
                        help='Window length')
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--pitch-fmin', default=40.0, type=float,
                        help='Minimum frequency for pitch extraction')
    parser.add_argument('--pitch-fmax', default=600.0, type=float,
                        help='Maximum frequency for pitch extraction')
    parser.add_argument('--pitch-method', default='yin', choices=['yin', 'pyin'],
                        help='Method to use for pitch extraction. Probabilistic YIN '
                        '(pyin) is more accurate but also slower')
    parser.add_argument('--durations-from', default=None, type=str,
                        choices=['textgrid', 'unit_rle', 'attn_prior'],
                        help='Source of input symbol durations. Either Praat TextGrids, '
                        'run-length encoding quantized unit sequences, or attention '
                        'priors derived from text and mel lengths')
    parser.add_argument('--trim-silence-dur', default=None, type=float,
                        help='Trim leading and trailing silences from audio using TextGrids. '
                        'Specify desired silence duration to leave in seconds (0 to trim '
                        'completely)')
    return parser


def passthrough_collate(batch):
    return batch


def calculate_pitch_mean_std(fname_pitch):
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for v in fname_pitch.values()])
    mean = np.mean(nonzeros)
    std = np.std(nonzeros)
    return mean, std


def save_stats(dataset_path, wav_text_filelist, feature_name, mean, std):
    fpath = stats_filename(dataset_path, wav_text_filelist, feature_name)
    with open(fpath, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=4)


def stats_filename(dataset_path, filelist_path, feature_name):
    stem = os.path.splitext(os.path.basename(filelist_path))[0]
    return os.path.join(dataset_path, f'{feature_name}_stats__{stem}.json')


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch TTS Data Pre-processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    DLLogger.init(backends=[StdOutBackend(Verbosity.VERBOSE,
        prefix_format=lambda t: "[{}] ".format(t.strftime('%Y-%m-%d %H:%M:%S')))])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})

    for datum in ('mels', 'durations', 'pitches'):
        os.makedirs(os.path.join(args.dataset_path, datum), exist_ok=True)

    for filelist in args.wav_text_filelists:
        # store all utterance metadata and pitches for normalization
        fname_text = {}
        fname_pitch = {}
        fname_spkr = {}

        load_mel_from_disk = False
        load_durs_from_disk = False
        load_pitch_from_disk = False
        dataset = TextMelAliLoader(
            args.dataset_path, filelist, args.text_cleaners, args.n_mel_channels,
            args.input_type, args.symbol_set, args.n_speakers,
            load_mel_from_disk, load_durs_from_disk, load_pitch_from_disk,
            args.max_wav_value, args.sampling_rate,
            args.filter_length, args.hop_length, args.win_length,
            args.mel_fmin, args.mel_fmax, args.peak_norm,
            args.durations_from, args.trim_silence_dur,
            args.pitch_fmin, args.pitch_fmax, args.pitch_method,
            pitch_mean=None, pitch_std=None, pitch_mean_std_file=None)
        data_loader = DataLoader(dataset, collate_fn=passthrough_collate,
                                 num_workers=args.n_workers,
                                 batch_size=1)  # no need to worry about padding

        label = os.path.splitext(os.path.basename(filelist))[0]
        for i, batch in enumerate(tqdm(data_loader, label)):
            text, mel, text_len, durations, pitch, speaker, fname = batch[0]
            fname_text[fname] = text
            fname_pitch[fname] = pitch
            fname_spkr[fname] = speaker

            if args.extract_mels:
                fpath = os.path.join(args.dataset_path, 'mels', fname + '.pt')
                torch.save(mel, fpath)

            if args.extract_durs:
                fpath = os.path.join(args.dataset_path, 'durations', fname + '.pt')
                torch.save(torch.tensor(durations).squeeze(), fpath)

            if args.extract_pitch:
                fpath = os.path.join(args.dataset_path, 'pitches', fname + '.pt')
                torch.save(torch.from_numpy(pitch), fpath)

        # TODO: consider normalizing per speaker
        mean, std = calculate_pitch_mean_std(fname_pitch)
        save_stats(args.dataset_path, filelist, 'pitches', mean, std)

        if args.write_meta:
            if args.n_speakers > 1:
                meta_header = 'audio|duration|pitch|text|speaker\n'
                meta_line = 'mels/{0}.pt|durations/{0}.pt|pitches/{0}.pt|{1}|{2}\n'
            else:
                meta_header = 'audio|duration|pitch|text\n'
                meta_line = 'mels/{0}.pt|durations/{0}.pt|pitches/{0}.pt|{1}\n'
            meta_file = os.path.join(args.dataset_path, label + '.meta.txt')
            with open(meta_file, 'w') as meta_out:
                meta_out.write(meta_header)
                for fname, text in fname_text.items():
                    meta_out.write(meta_line.format(fname, text, fname_spkr[fname]))
    DLLogger.flush()


if __name__ == '__main__':
    main()
