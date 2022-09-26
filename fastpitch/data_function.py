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

import numpy as np
import torch

import common.layers as layers
from common.utils import load_filepaths_and_text, load_wav_to_torch, to_gpu
from common.text.text_processing import TextProcessor, PhoneProcessor, UnitProcessor


class TextMelAliLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, text_cleaners, n_mel_channels,
                 input_type='char', symbol_set='english_basic', n_speakers=1,
                 load_mel_from_disk=True, max_wav_value=None, sampling_rate=None,
                 filter_length=None, hop_length=None, win_length=None,
                 mel_fmin=None, mel_fmax=None, peak_norm=False, **kwargs):
        self.audiopaths_and_text = load_filepaths_and_text(
            dataset_path, audiopaths_and_text,
            has_speakers=(n_speakers > 1))
        self.n_speakers = n_speakers

        self.input_type = input_type
        self.symbol_set = symbol_set
        self.text_cleaners = text_cleaners
        if self.input_type == 'char':
            self.tp = TextProcessor(self.symbol_set, self.text_cleaners)
        elif self.input_type == 'unit':
            self.tp = UnitProcessor(self.symbol_set, self.input_type)
        else:
            self.tp = PhoneProcessor(self.symbol_set, self.input_type)
        self.n_symbols = len(self.tp.symbols)

        self.load_mel_from_disk = load_mel_from_disk
        if not load_mel_from_disk:
            self.max_wav_value = max_wav_value
            self.peak_norm = peak_norm
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.peak_norm:
                audio = (audio / torch.max(torch.abs(audio))) * (self.max_wav_value - 1)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
        return melspec

    def get_text(self, text):
        text_encoded = torch.IntTensor(self.tp.encode_text(text))
        return text_encoded

    def __getitem__(self, index):
        # separate filename and text
        if self.n_speakers > 1:
            audiopath, durpath, pitchpath, text, speaker = self.audiopaths_and_text[index]
            speaker = int(speaker)
        else:
            audiopath, durpath, pitchpath, text = self.audiopaths_and_text[index]
            speaker = None
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        # expect always to load duration and pitch targets from disk
        dur = torch.load(durpath)
        pitch = torch.load(pitchpath)
        return (text, mel, len_text, dur, pitch, speaker)

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelAliCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, symbol_type='char', n_symbols=148, mas=False):
        self.symbol_type = symbol_type
        self.n_symbols = n_symbols
        self.mas = mas

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        if self.symbol_type == 'pf':
            text_padded = torch.FloatTensor(len(batch), max_input_len, self.n_symbols)
        else:
            text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        if self.mas:
            dur_padded = torch.zeros(
                len(batch), max_target_len, max_input_len)
            dur_padded.zero_()
            dur_lens = None
            for i in range(len(ids_sorted_decreasing)):
                dur = batch[ids_sorted_decreasing[i]][3]
                dur_padded[i, :dur.size(0), :dur.size(1)] = dur
        else:
            dur_padded = torch.zeros(
                len(batch), max_input_len, dtype=batch[0][3].dtype)
            dur_lens = torch.zeros(dur_padded.size(0), dtype=torch.int32)
            for i in range(len(ids_sorted_decreasing)):
                dur = batch[ids_sorted_decreasing[i]][3]
                dur_padded[i, :dur.shape[0]] = dur
                dur_lens[i] = dur.shape[0]
                assert dur_lens[i] == input_lengths[i]

        if self.mas:
            pitch_padded = torch.zeros(
                mel_padded.size(0), mel_padded.size(2), dtype=batch[0][4].dtype)
        else:
            pitch_padded = torch.zeros(
                dur_padded.size(0), dur_padded.size(1), dtype=batch[0][4].dtype)
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :pitch.shape[0]] = pitch

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        return (text_padded, input_lengths, mel_padded, output_lengths,
                len_x, dur_padded, dur_lens, pitch_padded, speaker)


def batch_to_gpu(batch, symbol_type='char', mas=False):
    text_padded, input_lengths, mel_padded, output_lengths, \
        len_x, dur_padded, dur_lens, pitch_padded, speaker = batch
    if symbol_type == 'pf':
        text_padded = to_gpu(text_padded).float()
    else:
        text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    dur_padded = to_gpu(dur_padded).long()
    pitch_padded = to_gpu(pitch_padded).float()
    if speaker is not None:
        speaker = to_gpu(speaker).long()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         dur_padded, dur_lens, pitch_padded, speaker]
    if mas:
        y = [mel_padded, input_lengths, output_lengths]
    else:
        dur_lens = to_gpu(dur_lens).long()
        y = [mel_padded, dur_padded, dur_lens, pitch_padded]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
