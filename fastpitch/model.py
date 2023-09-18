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
import torch.nn as nn
import torch.nn.functional as F

from common.layers import ConvReLUNorm, SeparableConv
from common.utils import mask_from_lens
from fastpitch.alignment import mas_width1
from fastpitch.attention import ConvAttention
from fastpitch.transformer import FFTransformer


def regulate_len(durations, enc_out, pace=1.0, mel_max_len=None):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=1), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=1), (1, 0))

    pitch_sums = (torch.gather(pitch_cums, 1, durs_cums_ends)
                  - torch.gather(pitch_cums, 1, durs_cums_starts)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 1, durs_cums_ends)
                    - torch.gather(pitch_nonzero_cums, 1, durs_cums_starts)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
    return pitch_avg


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, sepconv=False):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout, sepconv=sepconv)
            for i in range(n_layers)]
        )
        self.fc = nn.Linear(filter_size, 1, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out.squeeze(-1)


class FastPitch(nn.Module):
    def __init__(self, n_mel_channels, symbol_type, n_symbols, padding_idx,
                 symbols_embedding_dim, use_sepconv, use_mas, tvcgmm_k,
                 in_fft_n_layers, in_fft_n_heads, in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_sepconv, in_fft_output_size, in_fft_post_cond,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_sepconv, out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 dur_predictor_sepconv, p_dur_predictor_dropout,
                 dur_predictor_n_layers,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 pitch_predictor_sepconv, p_pitch_predictor_dropout,
                 pitch_predictor_n_layers,
                 pitch_embedding_kernel_size, pitch_embedding_sepconv,
                 n_speakers, speaker_emb_weight, n_langs, lang_emb_weight):
        super(FastPitch, self).__init__()

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx,
            input_type=symbol_type,
            sepconv=in_fft_sepconv or use_sepconv,
            post_cond=in_fft_post_cond
        )

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        if n_langs > 1:
            self.lang_emb = nn.Embedding(n_langs, symbols_embedding_dim)
        else:
            self.lang_emb = None
        self.lang_emb_weight = lang_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout,
            n_layers=dur_predictor_n_layers,
            sepconv=dur_predictor_sepconv or use_sepconv
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim,
            sepconv=out_fft_sepconv or use_sepconv
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=pitch_predictor_filter_size,
            kernel_size=pitch_predictor_kernel_size,
            dropout=p_pitch_predictor_dropout,
            n_layers=pitch_predictor_n_layers,
            sepconv=pitch_predictor_sepconv or use_sepconv
        )

        if pitch_embedding_sepconv or use_sepconv:
            self.pitch_emb_conv_fn = SeparableConv
        else:
            self.pitch_emb_conv_fn = nn.Conv1d
        self.pitch_emb = self.pitch_emb_conv_fn(
            1, symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size, stride=1,
            padding=int((pitch_embedding_kernel_size - 1) / 2))

        # Store values precomputed for training data within the model
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))
        self.n_mel_channels = n_mel_channels

        self.tvcgmm_k = tvcgmm_k
        if tvcgmm_k:
            # predict 3 bin means + 6 covariance values + 1 mixture weight per k
            self.proj = nn.Linear(out_fft_output_size, n_mel_channels * tvcgmm_k * 10)
        else:
            self.proj = nn.Linear(out_fft_output_size, n_mel_channels)

        # For monotonic alignment search (see forward_mas)
        self.use_mas = use_mas
        if use_mas:
            self.attention = ConvAttention(
                n_mel_channels, 0, symbols_embedding_dim,
                use_query_proj=True, align_query_enc_type='3xconv')

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_out_cpu = np.zeros(attn.data.shape, dtype=np.float32)
            log_attn_cpu = torch.log(attn.data).to(device='cpu', dtype=torch.float32)
            log_attn_cpu = log_attn_cpu.numpy()
            out_lens_cpu = out_lens.cpu()
            in_lens_cpu = in_lens.cpu()
            for ind in range(b_size):
                hard_attn = mas_width1(
                    log_attn_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]])
                attn_out_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]] = hard_attn
            attn_out = torch.tensor(
                attn_out_cpu, device=attn.get_device(), dtype=attn.dtype)
        return attn_out

    def forward(self, inputs, use_gt_durations=True, use_gt_pitch=True,
                pace=1.0, max_duration=75):
        inputs, _, mel_tgt, _, dur_tgt, _, pitch_tgt, speaker, language = inputs
        mel_max_len = mel_tgt.size(2)

        # Calculate speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        if self.lang_emb is None:
            lang_emb = 0
        else:
            lang_emb = self.lang_emb(language).unsqueeze(1)
            lang_emb.mul_(self.lang_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb + lang_emb)

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Predict pitch
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)

        if use_gt_pitch and pitch_tgt is not None:
            pitch_emb = self.pitch_emb(pitch_tgt.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        len_regulated, dec_lens = regulate_len(
            dur_tgt if use_gt_durations else dur_pred,
            enc_out, pace, mel_max_len)

        # Output FFT
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        return mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred

    def forward_mas(self, inputs, use_gt_pitch=True, pace=1.0, max_duration=75,
                    use_gt_durations=None):  # compatibility
        (inputs, input_lens, mel_tgt, mel_lens,
            attn_prior, _, pitch_dense, speaker, language) = inputs

        text_max_len = inputs.size(1)
        mel_max_len = mel_tgt.size(2)

        # Calculate speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        if self.lang_emb is None:
            lang_emb = 0
        else:
            lang_emb = self.lang_emb(language).unsqueeze(1)
            lang_emb.mul_(self.lang_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb + lang_emb)

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Predict pitch
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)

        # Alignment
        text_emb = self.encoder.word_emb(inputs)

        # make sure to do the alignments before folding
        attn_mask = mask_from_lens(input_lens, max_len=text_max_len)
        attn_mask = attn_mask[..., None] == 0
        # attn_mask should be 1 for unused timesteps in the text_enc_w_spkvec tensor

        attn_soft, attn_logprob = self.attention(
            mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask,
            key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)

        attn_hard = self.binarize_attention(attn_soft, input_lens, mel_lens)

        # Viterbi --> durations
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        dur_tgt = attn_hard_dur
        assert torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens))

        # Average pitch over characters
        pitch_tgt = average_pitch(pitch_dense, dur_tgt)

        if use_gt_pitch and pitch_tgt is not None:
            pitch_emb = self.pitch_emb(pitch_tgt.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        len_regulated, dec_lens = regulate_len(
            dur_tgt, enc_out, pace, mel_max_len)

        # Output FFT
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        return (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred,
                pitch_tgt, attn_soft, attn_hard, attn_hard_dur, attn_logprob)

    def infer(self, inputs, pace=1.0, dur_tgt=None, pitch_tgt=None,
              pitch_transform=None, max_duration=75, speaker=0, language=0):

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        if self.lang_emb is None:
            lang_emb = 0
        else:
            language = torch.ones(inputs.size(0)).long().to(inputs.device) * language
            lang_emb = self.lang_emb(language).unsqueeze(1)
            lang_emb.mul_(self.lang_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb + lang_emb)

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        if dur_tgt is not None and self.use_mas:
            # assume we don't have actual target durations (otherwise why
            # use mas?), so generate them here
            attn_prior, mel_tgt, mel_lens, input_lens = dur_tgt
            text_emb = self.encoder.word_emb(inputs)
            attn_mask = mask_from_lens(input_lens, max_len=inputs.size(1))
            attn_mask = attn_mask[..., None] == 0
            attn_soft, attn_logprob = self.attention(
                mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask,
                key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)
            attn_hard = self.binarize_attention(attn_soft, input_lens, mel_lens)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            dur_tgt = attn_hard_dur

        # Pitch over chars
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)

        if pitch_transform is not None:
            if self.pitch_std[0] == 0.0:
                # XXX LJSpeech-1.1 defaults
                mean, std = 218.14, 67.24
            else:
                mean, std = self.pitch_mean[0], self.pitch_std[0]
            pitch_pred = pitch_transform(pitch_pred, enc_mask.sum(dim=(1,2)), mean, std)

        if pitch_tgt is None:
            pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1)).transpose(1, 2)
        else:
            if self.use_mas:
                pitch_tgt = average_pitch(pitch_tgt, dur_tgt)
            pitch_emb = self.pitch_emb(pitch_tgt.unsqueeze(1)).transpose(1, 2)

        enc_out = enc_out + pitch_emb

        len_regulated, dec_lens = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        return mel_out, dec_lens, dur_pred, pitch_pred
