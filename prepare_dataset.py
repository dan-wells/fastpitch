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
import functools
import json
import os
import time
from itertools import groupby

import librosa
import tgt
import torch
import dllogger as DLLogger
import numpy as np
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from scipy import ndimage
from scipy.stats import betabinom
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastpitch.data_function import TextMelAliLoader


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelist', required=True,
                        type=str, help='Path to file with audio paths and text')
    parser.add_argument('--output-meta-file', default=None, type=str,
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
                        '(pyin) is more accurate but also much slower.')
    parser.add_argument('--pitch-mean', default=None, type=float,
                        help='Mean value to normalize extracted pitch')
    parser.add_argument('--pitch-std', default=None, type=float,
                        help='Standard deviation to normalize extracted pitch')
    parser.add_argument('--durations-from', type=str, default='',
                        choices=['textgrid', 'unit_rle', 'attn_prior'],
                        help='Extract symbol durations from Praat TextGrids or '
                        'by run-length encoding quantized unit sequences')
    parser.add_argument('--trim-silences', default=None, type=float,
                        help='Trim leading and trailing silences from audio using TextGrids. '
                        'Specify desired silence duration to leave (0 to trim completely)')
    return parser


class FilenamesLoader(TextMelAliLoader):
    def __init__(self, filenames, **kwargs):
        self.filenames = filenames
        kwargs['audiopaths_and_text'] = [kwargs['wav_text_filelist']]
        kwargs['load_mel_from_disk'] = False
        super(FilenamesLoader, self).__init__(**kwargs)

    def __getitem__(self, index):
        # separate filename and text
        audiopath, text = self.audiopaths_and_text[index]
        text = self.get_text(text)
        len_text = len(text)
        mel = self.get_mel(audiopath)
        len_mel = mel.shape[1]
        fname = self.filenames[index]
        return (text, mel, len_text, len_mel, fname)


def passthrough_collate(batch):
    return batch


def stats_filename(dataset_path, filelist_path, feature_name):
    stem = os.path.splitext(os.path.basename(filelist_path))[0]
    return os.path.join(dataset_path, f'{feature_name}_stats__{stem}.json')


def extract_durs_from_textgrid(fname, dataset_path, sampling_rate, hop_length, mel_len):
    # TODO: Something better than forcing consistent filepaths between
    # wavs and TextGrids
    tg_path = os.path.join(dataset_path, 'TextGrid', fname + '.TextGrid')
    try:
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
    except FileNotFoundError:
        print('Expected consistent filepaths between wavs and TextGrids, e.g.')
        print('  /path/to/wavs/speaker_uttID.wav -> /path/to/TextGrid/speaker_uttID.TextGrid')
        raise

    phones, durs, start, end = parse_textgrid(
        textgrid.get_tier_by_name('phones'), sampling_rate, hop_length)
    assert sum(durs) == mel_len, f'Length mismatch: {fname}, {sum(durs)} != {mel_len}'
    return durs, phones, start, end


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
    n_samples = end_time * sampling_rate
    n_frames = n_samples / hop_length
    # fix occasional length mismatches at the end of utterances when
    # duration in samples is an integer multiple of hop_length
    if n_frames.is_integer():
        durations[-1] += 1
    return phones, durations, start_time, end_time


def extract_durs_from_unit_sequence(fname, dataset_path, text, mel_len):
    text = text.numpy()
    units, durs = run_length_encode(text)
    total_dur = sum(durs)
    if total_dur != mel_len:
        # Extracted HuBERT feature sequences are frequently 1-frame short
        # because they truncate rather than padding utterances whose
        # durations in samples are not integer multiples of hop_length. If
        # duration is also an integer multiple of FFT filter length, we
        # lose an additional frame (2 frames short) -- catch both cases here
        dur_diff = mel_len - total_dur
        durs[-1] += dur_diff
    assert sum(durs) == mel_len, f'Length mismatch: {fname}, {sum(durs)} != {mel_len}'
    return durs, units


def run_length_encode(symbols):
    units = []
    run_lengths = []
    for unit, run in groupby(symbols):
        units.append(str(unit))
        run_lengths.append(len(list(run)))
    return units, run_lengths


class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def extract_duration_prior(text_len, mel_len):
    binomial_interpolator = BetaBinomialInterpolator()
    attn_prior = binomial_interpolator(mel_len, text_len)
    #attn_prior = beta_binomial_prior_distribution(text_len, mel_len)
    assert mel_len == attn_prior.shape[0]
    return attn_prior


def extract_pitches(fname, durations, dataset_path, fmin=40, fmax=600,
                    sr=None, hop_length=256, method='yin', start=None, end=None):
    fpath = os.path.join(dataset_path, 'pitches', fname + '.pt')
    wav = os.path.join(dataset_path, 'wavs', fname + '.wav')
    pitches = calculate_pitch(
        str(wav), durations, fmin, fmax, sr, hop_length, method, start, end)
    return pitches


def calculate_pitch(wav, mel_len, fmin=40, fmax=600, sr=None, hop_length=256,
                    method='yin', start=None, end=None):
    try:
        trimmed_dur = end - start
    except TypeError:
        # either start or end is None => don't need to calculate final duration
        trimmed_dur = end
    snd, sr = librosa.load(wav, sr=sr, offset=start, duration=trimmed_dur)

    if method == 'yin':
        pitch = librosa.yin(snd, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    elif method == 'pyin':
        pitch, voiced_flags, voiced_probs  = librosa.pyin(
            snd, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, fill_na=0.0)
    assert np.abs(mel_len - pitch.shape[0]) <= 1.0
    return pitch


def average_pitch_per_symbol(pitch, durs, mel_len):
    durs = np.array(durs)
    assert durs.sum() == mel_len
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))
    pitch_char = np.zeros((durs.shape[0],), dtype=float)
    for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
    return pitch_char


def normalize_pitch_vectors(fname_pitch, mean=None, std=None):
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for v in fname_pitch.values()])
    if mean is None:
        mean = np.mean(nonzeros)
    if std is None:
        std = np.std(nonzeros)
    for v in fname_pitch.values():
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0
    return fname_pitch, mean, std


def save_stats(dataset_path, wav_text_filelist, feature_name, mean, std):
    fpath = stats_filename(dataset_path, wav_text_filelist, feature_name)
    with open(fpath, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=4)


def trim_silences(fname, dataset_path, mel, pitch, text, durations,
                  keep_sil_frames, sampling_rate, hop_length):
    if keep_sil_frames > 0:
        keep_sil_frames = np.round(keep_sil_frames * sampling_rate / hop_length)
    keep_sil_frames = int(keep_sil_frames)

    text = np.array(text)
    sil_idx = np.where(text == 'sil')[0]

    durations = np.array(durations)
    sil_durs = durations[sil_idx]
    trim_durs = np.array(
        [d - keep_sil_frames if d > keep_sil_frames else 0 for d in sil_durs],
        dtype=np.int64)
    if trim_durs.size == 2:
        # trim both sides
        trim_start, trim_end = trim_durs
        trim_end = -trim_end
    elif trim_durs.size == 0:
        # nothing to trim
        trim_start, trim_end = None, None
    elif sil_idx[0] == 0:
        # trim only leading silence
        trim_start, trim_end = trim_durs[0], None
    elif sil_idx[0] == len(text) - 1:
        # trim only trailing silence
        trim_start, trim_end = None, -trim_durs[0]
    if trim_end == 0:
        # don't trim trailing silence if already short enough
        trim_end = None

    if keep_sil_frames == 0:
        sil_mask = text != "sil"
    else:
        sil_mask = np.ones_like(text, dtype=bool)
    mel = mel[:, trim_start:trim_end]
    durations.put(sil_idx, sil_durs - trim_durs)
    durations = durations[sil_mask]
    assert mel.shape[1] == sum(durations), \
        "{}: Trimming led to mismatched durations ({}) and mels ({})".format(
            fname, sum(durations), mel.shape[1])
    pitch = pitch[sil_mask]
    assert len(pitch) == len(durations), \
        "{}: Trimming led to mismatched durations ({}) and pitches ({})".format(
            fname, len(durations), len(pitch))
    text = text[sil_mask]
    return mel, pitch, text


def main():
    parser = argparse.ArgumentParser(description='PyTorch TTS Data Pre-processing')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    if args.output_meta_file is None:
        args.output_meta_file = os.path.join(args.dataset_path, 'meta.txt')
    if args.trim_silences is not None:
        assert args.durations_from == 'textgrid', \
            "Can only trim silences based on TextGrid alignments"

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})

    for datum in ('mels', 'durations', 'pitches'):
        os.makedirs(os.path.join(args.dataset_path, datum), exist_ok=True)

    # store all utterance metadata and pitches for normalization
    fname_text = {}
    fname_pitch = {}

    filenames = [os.path.splitext(os.path.basename(l.split('|')[0]))[0]
                 for l in open(args.wav_text_filelist, 'r')]
    dataset = FilenamesLoader(filenames, **vars(args))
    data_loader = DataLoader(dataset, collate_fn=passthrough_collate,
                             batch_size=1)  # no need to worry about padding

    for i, batch in enumerate(tqdm(data_loader, 'Processing audio files')):
        text, mel, text_len, mel_len, fname = batch[0]

        fpath = os.path.join(args.dataset_path, 'mels', fname + '.pt')
        torch.save(mel, fpath)

        start_time, end_time = None, None
        if args.durations_from == 'textgrid':
            durations, text, start_time, end_time = extract_durs_from_textgrid(
                fname, args.dataset_path, args.sampling_rate, args.hop_length, mel_len)
            text = ' '.join(text)
        elif args.durations_from == 'unit_rle':
            durations, text = extract_durs_from_unit_sequence(
                fname, args.dataset_path, text, mel_len)
            text = ' '.join(text)
        elif args.durations_from == 'attn_prior':
            durations = extract_duration_prior(text_len, mel_len)
            text = dataset.tp.ids_to_text(text.numpy())
        fpath = os.path.join(args.dataset_path, 'durations', fname + '.pt')
        torch.save(torch.tensor(durations), fpath)
        # texts have been modified here: silences from textgrids, run-length encoding of units
        fname_text[fname] = text

        pitches = extract_pitches(
            fname, mel_len, args.dataset_path,
            args.pitch_fmin, args.pitch_fmax, args.sampling_rate, args.hop_length,
            args.pitch_method, start_time, end_time)
        if args.durations_from != 'attn_prior':
            pitches = average_pitch_per_symbol(pitches, durations, mel_len)
        fname_pitch[fname] = pitches

        if args.trim_silences is not None:
            mel, pitches, text = trim_silences(
                fname, args.dataset_path, mel, pitches, text, durations,
                args.trim_silences, args.sampling_rate, args.hop_length)
            # possibly trimmed pitches and text
            fname_pitch[fname] = pitches
            fname_text[fname] = text
            # save mels and durs again (sorry)
            torch.save(mel, os.path.join(args.dataset_path, 'mels', fname + '.pt'))
            torch.save(durations, os.path.join(args.dataset_path, 'durations', fname + '.pt'))

    fname_pitch, mean, std = normalize_pitch_vectors(fname_pitch, args.pitch_mean, args.pitch_std)
    # TODO: consider normalizing per speaker
    for fname, pitch in fname_pitch.items():
        fpath = os.path.join(args.dataset_path, 'pitches', fname + '.pt')
        torch.save(torch.from_numpy(pitch), fpath)
    save_stats(args.dataset_path, args.wav_text_filelist, 'pitches', mean, std)

    if args.output_meta_file is not None:
        with open(args.output_meta_file, 'w') as meta_out:
            for fname in filenames:
                meta_out.write(
                    'mels/{0}.pt|durations/{0}.pt|pitches/{0}.pt|{1}\n'.format(
                        fname, fname_text[fname]))
    DLLogger.flush()


if __name__ == '__main__':
    main()
