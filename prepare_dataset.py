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
import time

import parselmouth
import tgt
import torch
import dllogger as DLLogger
import numpy as np
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from torch.utils.data import DataLoader

from common import utils
from tacotron2.data_function import TextMelLoader, TextMelCollate, batch_to_gpu
from common.text.text_processing import TextProcessing, PhoneProcessing, UnitProcessing


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelist', required=True,
                        type=str, help='Path to file with audio paths and text')
    parser.add_argument('--output-meta-file', default=None, type=str,
                        help='Write metadata file pointing to extracted features')
    parser.add_argument('--text-cleaners', nargs='*',
                        default=['english_cleaners'], type=str,
                        help='Type of text cleaners for input text')
    parser.add_argument('--symbol-set', type=str, default='english_basic',
                        help='Define symbol set for input text')
    parser.add_argument('--input-type', type=str, default='char',
                        choices=['char', 'phone', 'unit'],
                        help='Input symbols used, either char (text), phone '
                        'or quantized unit symbols.')
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--peak-norm', action='store_true',
                        help='Apply peak normalization to audio')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--durations-from', type=str, default='',
                        choices=['textgrid', 'unit_rle'],
                        help='Extract symbol durations from Praat TextGrids or '
                        'by run-length encoding quantized unit sequences')
    parser.add_argument('--trim-silences', default=None, type=float,
                        help='Trim leading and trailing silences from audio using TextGrids. '
                        'Specify desired silence duration to leave (0 to trim completely)')
    return parser


class FilenamesLoader(TextMelLoader):
    def __init__(self, filenames, **kwargs):
        # dict_args = vars(args)
        kwargs['audiopaths_and_text'] = [kwargs['wav_text_filelist']]
        kwargs['load_mel_from_disk'] = False
        super(FilenamesLoader, self).__init__(**kwargs)
        if kwargs['input_type'] == 'phone':
            self.tp = PhoneProcessing(kwargs['symbol_set'])
        elif kwargs['input_type'] == 'unit':
            self.tp = UnitProcessing(kwargs['symbol_set'], kwargs['input_type'])
        else:
            self.tp = TextProcessing(kwargs['symbol_set'], kwargs['text_cleaners'])
        self.filenames = filenames

    def __getitem__(self, index):
        mel_text = super(FilenamesLoader, self).__getitem__(index)
        return mel_text + (self.filenames[index],)


def maybe_pad(vec, l):
    assert np.abs(vec.shape[0] - l) <= 3
    vec = vec[:l]
    if vec.shape[0] < l:
        vec = np.pad(vec, pad_width=(0, l - vec.shape[0]))
    return vec


def extract_durs_from_textgrids(mel_lens, fnames, dataset_path, sampling_rate, hop_length, metadata):
    texts = []
    durations = []
    start_times = []
    end_times = []
    for j, mel_len in enumerate(mel_lens):
        # TODO: Something better than forcing consistent filepaths between
        # wavs and TextGrids
        tg_path = os.path.join(dataset_path, 'TextGrid', fnames[j] + '.TextGrid')
        try:
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        except FileNotFoundError:
            print('Expected consistent filepaths between wavs and TextGrids, e.g.')
            print('  /path/to/wavs/speaker_uttID.wav -> /path/to/TextGrid/speaker_uttID.TextGrid')
            raise
        phones, durs, start, end = parse_textgrid(
            textgrid.get_tier_by_name('phones'), sampling_rate, hop_length)
        texts.append(phones)
        start_times.append(start)
        end_times.append(end)
        assert sum(durs) == mel_len, f'Length mismatch: {fnames[j]}, {sum(durs)} != {mel_len}'
        dur = torch.LongTensor(durs)
        durations.append(dur)
        fpath = os.path.join(dataset_path, 'durations', fnames[j] + '.pt')
        torch.save(dur.cpu().int(), fpath)
        if metadata is not None:
            metadata[fnames[j]] = phones
    return durations, texts, start_times, end_times, metadata


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
    # fix occasional length mismatches at the end of utterances
    if n_frames == int(n_frames):
        durations[-1] += 1
    return phones, durations, start_time, end_time


def extract_durs_from_unit_sequences(mel_lens, fnames, texts_padded, text_lens, metadata):
    durations = []
    for j, mel_len in enumerate(mel_lens):
        text = texts_padded[j][:text_lens[j]].cpu().numpy()
        rle_text = run_length_encode(text)
        units, durs = (list(i) for i in zip(*rle_text))  # unpack list of tuples to two sequences
        total_dur = sum(durs)
        if total_dur != mel_len:
            dur_diff = mel_len - total_dur
            durs[-1] += dur_diff
            # TODO: Work out why some utterances are 2 frames short.
            # Expected for extracted HuBERT feature sequences to be
            # consistently 1-frame short because they truncate rather than
            # padding utterances which don't exactly fit into 0.02 s chunks,
            # but not sure where this extra missing frame comes from
            if dur_diff != 1:
                DLLogger.log(step="Feature length mismatch {}: {} mels, {} durs".format(
                    fnames[j], mel_len, total_dur), data={})
        assert sum(durs) == mel_len, f'Length mismatch: {fnames[j]}, {sum(durs)} != {mel_len}'
        dur = torch.LongTensor(durs)
        durations.append(dur)
        fpath = os.path.join(dataset_path, 'durations', fnames[j] + '.pt')
        torch.save(dur.cpu().int(), fpath)
        if metadata is not None:
            metadata[fnames[j]] = [str(i) for i in units]
    return durations, metadata


def run_length_encode(symbols):
    run_lengths = []
    dur = 1
    for u0, u1 in zip(symbols, symbols[1:]):
        if u1 == u0:
            dur += 1
        else:
            run_lengths.append([u0, dur])
            dur = 1
    run_lengths.append([u1, dur])
    return run_lengths


def extract_pitches(pitch_vecs, durations, fnames, dataset_path, trim_silences,
                    start_times, end_times):
    for j, dur in enumerate(durations):
        fpath = os.path.join(dataset_path, 'pitches', fnames[j] + '.pt')
        wav = os.path.join(dataset_path, 'wavs', fnames[j] + '.wav')
        if trim_silences is not None:
            p_char = calculate_pitch(str(wav), dur.cpu().numpy(), start_times[j], end_times[j])
        else:
            p_char = calculate_pitch(str(wav), dur.cpu().numpy())
        pitch_vecs[fnames[j]] = p_char
    return pitch_vecs


def calculate_pitch(wav, durs, start=None, end=None):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))
    snd = parselmouth.Sound(wav)
    snd = snd.extract_part(from_time=start, to_time=end)
    pitch = snd.to_pitch(time_step=snd.duration / (mel_len + 3)
                         ).selected_array['frequency']
    assert np.abs(mel_len - pitch.shape[0]) <= 1.0

    # Average pitch over characters
    pitch_char = np.zeros((durs.shape[0],), dtype=float)
    for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
    pitch_char = maybe_pad(pitch_char, len(durs))

    return pitch_char


def normalize_pitch_vectors(pitch_vecs):
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for v in pitch_vecs.values()])
    mean, std = np.mean(nonzeros), np.std(nonzeros)

    for v in pitch_vecs.values():
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0

    return mean, std


def save_stats(dataset_path, wav_text_filelist, feature_name, mean, std):
    fpath = utils.stats_filename(dataset_path, wav_text_filelist, feature_name)
    with open(fpath, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=4)


def trim_silences(keep_sil_frames, sampling_rate, hop_length, fnames, dataset_path,
                  mel_lens, mels_padded, durations, pitch_vecs, texts, metadata):
    if keep_sil_frames > 0:
        keep_sil_frames = np.round(keep_sil_frames * sampling_rate / hop_length)
    keep_sil_frames = int(keep_sil_frames)
    for j, text in enumerate(texts):
        text = np.array(text)
        sil_idx = np.where(text == 'sil')[0]
        sil_durs = durations[j][sil_idx]
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
        text = text[sil_mask]
        if metadata is not None:
            metadata[fnames[j]] = text
        mel = mels_padded[j][:, :mel_lens[j]].cpu()
        mel = mel[:, trim_start:trim_end]
        dur = durations[j]
        dur = dur.put(torch.tensor(sil_idx), sil_durs - trim_durs)
        dur = dur[sil_mask]
        assert mel.shape[1] == sum(dur), \
            "{}: Trimming led to mismatched durations ({}) and mels ({})".format(
                fnames[j], sum(dur), mel.shape[1])
        pitch = pitch_vecs[fnames[j]]
        pitch = pitch[sil_mask]
        pitch_vecs[fnames[j]] = pitch
        assert len(pitch) == len(dur), \
            "{}: Trimming led to mismatched durations ({}) and pitches ({})".format(
                fnames[j], len(dur), len(pitch))
        # save mels and durs again (sorry)
        torch.save(mel, os.path.join(dataset_path, 'mels', fnames[j] + '.pt'))
        torch.save(dur, os.path.join(dataset_path, 'durations', fnames[j] + '.pt'))
    return pitch_vecs, metadata


def main():
    parser = argparse.ArgumentParser(description='PyTorch TTS Data Pre-processing')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    if args.trim_silences is not None:
        assert args.durations_from == 'textgrid', \
            "Can only trim silences based on TextGrid alignments"

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, args.log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})

    for datum in ('mels', 'durations', 'pitches'):
        os.makedirs(os.path.join(args.dataset_path, datum), exist_ok=True)

    pitch_vecs = {}
    if args.output_meta_file is None:
        metadata = None
    else:
        metadata = {}

    filenames = [os.path.splitext(os.path.basename(l.split('|')[0]))[0]
                 for l in open(args.wav_text_filelist, 'r')]
    # Compatibility with Tacotron2 Data loader
    args.n_speakers = 1
    dataset = FilenamesLoader(filenames, **vars(args))
    # TextMelCollate supports only n_frames_per_step=1
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             sampler=None, num_workers=0,
                             collate_fn=TextMelCollate(1),
                             pin_memory=False, drop_last=False)
    for i, batch in enumerate(data_loader):
        tik = time.time()
        fnames = batch[-1]
        x, _, _ = batch_to_gpu(batch[:-1])
        texts_padded, text_lens, mels_padded, _, mel_lens = x
        start_times, end_times = [], []

        for j, mel in enumerate(mels_padded):
            fpath = os.path.join(args.dataset_path, 'mels', fnames[j] + '.pt')
            torch.save(mel[:, :mel_lens[j]].cpu().clone(), fpath)

        if args.durations_from == 'textgrid':
            durations, texts, start_times, end_times, metadata = extract_durs_from_textgrids(
                mel_lens, fnames, args.dataset_path, args.sampling_rate, args.hop_length,
                metadata)
        elif args.durations_from == 'unit_rle':
            durations, metadata = extract_durs_from_unit_sequences(
                mel_lens, fnames, texts_padded, text_lens, metadata)

        pitch_vecs = extract_pitches(
            pitch_vecs, durations, fnames, args.dataset_path, args.trim_silences, start_times, end_times)

        if args.trim_silences is not None:
            pitch_vecs, metadata = trim_silences(
                args.trim_silences, args.sampling_rate, args.hop_length, fnames, args.dataset_path,
                mel_lens, mels_padded, durations, pitch_vecs, texts, metadata)

        nseconds = time.time() - tik
        DLLogger.log(step=f'{i+1}/{len(data_loader)} ({nseconds:.2f}s)', data={})

    mean, std = normalize_pitch_vectors(pitch_vecs)
    for fname, pitch in pitch_vecs.items():
        fpath = os.path.join(args.dataset_path, 'pitches', fname + '.pt')
        torch.save(torch.from_numpy(pitch), fpath)
    save_stats(args.dataset_path, args.wav_text_filelist, 'pitches', mean, std)

    if args.output_meta_file is not None:
        with open(args.output_meta_file, 'w') as meta_out:
            for fname in filenames:
                meta_out.write(
                    'mels/{0}.pt|durations/{0}.pt|pitches/{0}.pt|{1}\n'.format(
                        fname, ' '.join(metadata[fname])))

    DLLogger.flush()


if __name__ == '__main__':
    main()
