# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import json
import sys
from typing import Optional
from os.path import abspath, dirname

import torch

from common.text.symbols import get_symbols, get_pad_idx
from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from hifigan import AttrDict
from hifigan.model import Generator as HiFiGanGenerator
from tacotron2.model import Tacotron2
from waveglow.model import WaveGlow


def parse_model_args(model_name, parser, add_help=False):
    if model_name == 'FastPitch':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name == 'HiFi-GAN':
        from hifigan.arg_parser import parse_hifigan_args
        return parse_hifigan_args(parser, add_help)
    elif model_name == 'WaveGlow':
        from waveglow.arg_parser import parse_waveglow_args
        return parse_waveglow_args(parser, add_help)
    elif model_name == 'Tacotron2':
        from tacotron2.arg_parser import parse_tacotron2_args
        return parse_tacotron2_args(parser, add_help)
    else:
        raise NotImplementedError(model_name)


def get_model(model_name, model_config, device,
              forward_is_infer=False, jitable=False):
    """Chooses a model based on name"""
    if model_name == 'FastPitch':
        if jitable:
            model = FastPitchJIT(**model_config)
        else:
            model = FastPitch(**model_config)
    elif model_name == 'HiFi-GAN':
        model = HiFiGanGenerator(model_config)
    elif model_name == 'WaveGlow':
        model = WaveGlow(**model_config)
    elif model_name == 'Tacotron2':
        model = Tacotron2(**model_config)
    else:
        raise NotImplementedError(model_name)

    if forward_is_infer and hasattr(model, 'infer'):
        model.forward = model.infer

    return model.to(device)


def get_model_config(model_name, args):
    """ Code chooses a model based on name"""
    if model_name == 'FastPitch':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            symbol_type=args.input_type,
            n_symbols=len(get_symbols(args.symbol_set, args.input_type)),
            padding_idx=get_pad_idx(args.symbol_set, args.input_type),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # model-wide architecture
            use_sepconv=args.use_sepconv,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_sepconv=args.in_fft_sepconv,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_sepconv=args.out_fft_sepconv,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            dur_predictor_sepconv=args.dur_predictor_sepconv,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            pitch_predictor_sepconv=args.pitch_predictor_sepconv,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            pitch_embedding_sepconv=args.pitch_embedding_sepconv,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight
        )
        return model_config
    elif model_name == "HiFi-GAN":
        with open(args.hifigan_config) as f:
            model_config = json.load(f)
        model_config = AttrDict(model_config)
        return model_config
    elif model_name == 'WaveGlow':
        model_config = dict(
            n_mel_channels=args.n_mel_channels,
            n_flows=args.flows,
            n_group=args.groups,
            n_early_every=args.early_every,
            n_early_size=args.early_size,
            WN_config=dict(
                n_layers=args.wn_layers,
                kernel_size=args.wn_kernel_size,
                n_channels=args.wn_channels
            )
        )
        return model_config
    elif model_name == 'Tacotron2':
        model_config = dict(
            # optimization
            mask_padding=args.mask_padding,
            # audio
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=len(get_symbols(args.symbol_set)),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # encoder
            encoder_kernel_size=args.encoder_kernel_size,
            encoder_n_convolutions=args.encoder_n_convolutions,
            encoder_embedding_dim=args.encoder_embedding_dim,
            # attention
            attention_rnn_dim=args.attention_rnn_dim,
            attention_dim=args.attention_dim,
            # attention location
            attention_location_n_filters=args.attention_location_n_filters,
            attention_location_kernel_size=args.attention_location_kernel_size,
            # decoder
            n_frames_per_step=args.n_frames_per_step,
            decoder_rnn_dim=args.decoder_rnn_dim,
            prenet_dim=args.prenet_dim,
            max_decoder_steps=args.max_decoder_steps,
            gate_threshold=args.gate_threshold,
            p_attention_dropout=args.p_attention_dropout,
            p_decoder_dropout=args.p_decoder_dropout,
            # postnet
            postnet_embedding_dim=args.postnet_embedding_dim,
            postnet_kernel_size=args.postnet_kernel_size,
            postnet_n_convolutions=args.postnet_n_convolutions,
            decoder_no_early_stopping=args.decoder_no_early_stopping,
        )
        return model_config
    else:
        raise NotImplementedError(model_name)
