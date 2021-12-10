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

import argparse
import models
import os
import time
import sys
import warnings
from tqdm import tqdm

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from common import utils
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text.symbols import get_pad_idx
from common.text.text_processing import TextProcessing, PhoneProcessing, UnitProcessing
from pitch_transform import pitch_transform_custom
from waveglow import model as glow
from waveglow.denoiser import Denoiser

sys.modules['glow'] = glow


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Full path to the input text (phrases separated by newlines)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str, default='',
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('--vocoder', type=str, default='', choices=['WaveGlow', 'HiFi-GAN'],
                        help='Vocoder model type if generating audio (skip to only generate mels)')
    parser.add_argument('--vocoder-checkpoint', type=str, default='',
                        help='Full path to vocoder model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
                        help='WaveGlow denoising')
    parser.add_argument('--hifigan-config', type=str, default='',
                        help='Path to HiFi-GAN config file')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--save-mels', action='store_true',
                        help='Save generated mel spectrograms from FastPitch')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str, default='',
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    transform = parser.add_argument_group('transform')
    transform.add_argument('--fade-out', type=int, default=10,
                           help='Number of fadeout frames at the end')
    transform.add_argument('--pace', type=float, default=1.0,
                           help='Adjust the pace of speech')
    transform.add_argument('--pitch-transform-flatten', action='store_true',
                           help='Flatten the pitch')
    transform.add_argument('--pitch-transform-invert', action='store_true',
                           help='Invert the pitch wrt mean value')
    transform.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                           help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
    transform.add_argument('--pitch-transform-shift', type=float, default=0.0,
                           help='Raise/lower the pitch by <hz>')
    transform.add_argument('--pitch-transform-custom', action='store_true',
                           help='Apply the transform from pitch_transform.py')

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--input-type', type=str, default='char',
                                 choices=['char', 'phone', 'pf', 'unit'],
                                 help='Input symbols used, either char (text), phone, '
                                 'pf (phonological feature vectors) or unit (quantized '
                                 'acoustic representation IDs)')
    text_processing.add_argument('--symbol-set', type=str, default='english_basic',
                                 help='Define symbol set for input sequences. For '
                                 'quantized unit inputs, pass the size of the vocabulary.')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['english_cleaners'], type=str,
                                 help='Type of text cleaners for input text.')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the model.')

    return parser


def load_model_from_ckpt(checkpoint_path, ema, model):

    checkpoint_data = torch.load(checkpoint_path)
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    elif 'generator' in checkpoint_data:
        # HiFi-GAN checkpoint
        model.load_state_dict(checkpoint_data['generator'])
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')

    return model


def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)

    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        model = load_model_from_ckpt(checkpoint, ema, model)

    if model_name == "WaveGlow":
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        model = model.remove_weightnorm(model)
    elif model_name == "HiFi-GAN":
        model.remove_weight_norm()

    if amp:
        model.half()
    model.eval()
    return model.to(device)


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c:f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, device, input_type, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False,
                           load_pitch=False, load_duration=False,
                           load_speaker=False):
    if input_type == 'char':
        tp = TextProcessing(symbol_set, text_cleaners)
    elif input_type == 'unit':
        tp = UnitProcessing(symbol_set, input_type)
    else:
        tp = PhoneProcessing(symbol_set, input_type)

    if input_type == 'pf':
        fields['text'] = [torch.FloatTensor(tp.encode_text(text))
                          for text in fields['text']]
    else:
        fields['text'] = [torch.LongTensor(tp.encode_text(text))
                          for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    if load_mels:
        assert 'mel' in fields
        fields['mel'] = [
            torch.load(os.path.join(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor([t.size(0) for t in fields['mel']])
    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(os.path.join(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor([t.size(0) for t in fields['pitch']])
    if load_duration:
        assert 'duration' in fields
        fields['duration'] = [
            torch.load(os.path.join(dataset, fields['duration'][i])) for i in order]
    if load_speaker:
        assert 'speaker' in fields
        fields['speaker'] = torch.LongTensor([int(fields['speaker'][i]) for i in order])
    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]
    if 'mel_output' in fields:
        fields['mel_output'] = [fields['mel_output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True,
                                        padding_value=get_pad_idx(symbol_set, input_type))
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'duration' and load_duration:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches


def build_pitch_transformation(args):
    if args.pitch_transform_custom:
        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens)
                    - mean) / std
        return custom_

    fun = 'pitch'
    if args.pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if args.pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if args.pitch_transform_amplify:
        ampl = args.pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f'({fun}) + {hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


class MeasureTime(list):
    def __init__(self, *args, cuda=True, **kwargs):
        super(MeasureTime, self).__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime((sum(ab) for ab in zip(self, other)), cuda=cuda)


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    log_fpath = args.log_file or os.path.join(args.output, 'nvlog_infer.json')
    log_fpath = unique_log_fpath(log_fpath)
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_fpath),
                            StdOutBackend(Verbosity.VERBOSE,
                                          metric_format=stdout_metric_format)])
    init_inference_metadata()
    [DLLogger.log("PARAMETER", {k: v}) for k, v in vars(args).items()]

    device = torch.device('cuda' if args.cuda else 'cpu')

    if args.fastpitch:
        generator = load_and_setup_model(
            'FastPitch', parser, args.fastpitch, args.amp, device,
            unk_args=unk_args, forward_is_infer=True, ema=args.ema,
            jitable=args.torchscript)

        if args.torchscript:
            generator = torch.jit.script(generator)
    else:
        generator = None

    if args.vocoder:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vocoder = load_and_setup_model(
                args.vocoder, parser, args.vocoder_checkpoint, args.amp, device,
                unk_args=unk_args, forward_is_infer=True, ema=args.ema)
        if args.vocoder == 'WaveGlow':
            denoiser = Denoiser(vocoder).to(device)
            vocoder = getattr(vocoder, 'infer', vocoder)
    else:
        vocoder = None

    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    fields = load_fields(args.input)
    batches = prepare_input_sequence(
        fields, device, args.input_type, args.symbol_set, args.text_cleaners,
        args.batch_size, args.dataset_path, load_mels=(generator is None),
        load_pitch=('pitch' in fields), load_duration=('duration' in fields),
        load_speaker=('speaker' in fields))

    # Use real data rather than synthetic - FastPitch predicts len
    if args.warmup_steps:
        for _ in tqdm(range(args.warmup_steps), 'Warmup'):
            with torch.no_grad():
                if generator is not None:
                    b = batches[0]
                    mel, *_ = generator(b['text'])
                if vocoder is not None:
                    if args.vocoder == 'WaveGlow':
                        audios = vocoder(mel, sigma=args.sigma_infer).float()
                        _ = denoiser(audios, strength=args.denoising_strength)
                    elif args.vocoder == 'HiFi-GAN':
                        audios = vocoder(mel)

    gen_measures = MeasureTime(cuda=args.cuda)
    vocoder_measures = MeasureTime(cuda=args.cuda)

    gen_kw = {'pace': args.pace,
              'speaker': args.speaker,
              'pitch_transform': build_pitch_transformation(args)}

    if args.torchscript:
        gen_kw.pop('pitch_transform')
        print('NOTE: Pitch transforms are disabled with TorchScript')

    all_utterances = 0
    all_samples = 0
    all_letters = 0
    all_frames = 0

    reps = args.repeats
    log_enabled = reps == 1
    log = lambda s, d: DLLogger.log(step=s, data=d) if log_enabled else None

    for rep in (tqdm(range(reps), 'Inference') if reps > 1 else range(reps)):
        for n, b in enumerate(batches):
            if generator is None:
                log(rep, {'Synthesizing from ground truth mels'})
                mel, mel_lens = b['mel'], b['mel_lens']
            else:
                gen_kw['dur_tgt'] = b['duration'] if 'duration' in b else None
                gen_kw['pitch_tgt'] = b['pitch'] if 'pitch' in b else None
                gen_kw['speaker'] = b['speaker'] if 'speaker' in b else args.speaker
                with torch.no_grad(), gen_measures:
                    mel, mel_lens, *_ = generator(b['text'], **gen_kw)

                gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
                all_letters += b['text_lens'].sum().item()
                all_frames += mel.size(0) * mel.size(2)
                log(rep, {"fastpitch_frames/s": gen_infer_perf})
                log(rep, {"fastpitch_latency": gen_measures[-1]})

                if args.save_mels and args.output is not None and reps == 1:
                    for i, _mel in enumerate(mel):
                        _mel = _mel[:, :mel_lens[i]]
                        if 'mel_output' in b:
                            fname = b['mel_output'][i]
                        elif 'output' in b:
                            fname = os.path.splitext(b['output'][i])[0] + '.pt'
                        else:
                            fname = f'mel_{(n * args.batch_size) + i}.pt'
                        mel_path = os.path.join(args.output, fname)
                        torch.save(_mel.cpu(), mel_path)

            if vocoder is not None:
                with torch.no_grad(), vocoder_measures:
                    if args.vocoder == 'WaveGlow':
                        audios = vocoder(mel, sigma=args.sigma_infer)
                        audios = denoiser(audios.float(),
                                          strength=args.denoising_strength
                                          ).squeeze(1)
                    elif args.vocoder == 'HiFi-GAN':
                        audios = vocoder(mel)
                        audios = audios.squeeze()

                all_utterances += len(audios)
                all_samples += sum(audio.size(0) for audio in audios)
                vocoder_infer_perf = (
                    audios.size(0) * audios.size(1) / vocoder_measures[-1])

                log(rep, {"vocoder_samples/s": vocoder_infer_perf})
                log(rep, {"vocoder_latency": vocoder_measures[-1]})

                if args.output is not None and reps == 1:
                    for i, audio in enumerate(audios):
                        audio = audio[:mel_lens[i].item() * args.stft_hop_length]

                        if args.fade_out:
                            fade_len = args.fade_out * args.stft_hop_length
                            fade_w = torch.linspace(1.0, 0.0, fade_len)
                            audio[-fade_len:] *= fade_w.to(audio.device)

                        audio = audio / torch.max(torch.abs(audio))
                        fname = b['output'][i] if 'output' in b else f'audio_{(n * args.batch_size) + i}.wav'
                        audio_path = os.path.join(args.output, fname)
                        write(audio_path, args.sampling_rate, audio.cpu().numpy())

            if generator is not None and vocoder is not None:
                log(rep, {"latency": (gen_measures[-1] + vocoder_measures[-1])})

    log_enabled = True
    if generator is not None:
        gm = np.sort(np.asarray(gen_measures))
        rtf = all_samples / (all_utterances * gm.mean() * args.sampling_rate)
        log((), {"avg_fastpitch_letters/s": all_letters / gm.sum()})
        log((), {"avg_fastpitch_frames/s": all_frames / gm.sum()})
        log((), {"avg_fastpitch_latency": gm.mean()})
        log((), {"avg_fastpitch_RTF": rtf})
        log((), {"90%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std()})
        log((), {"95%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std()})
        log((), {"99%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std()})
    if vocoder is not None:
        wm = np.sort(np.asarray(vocoder_measures))
        rtf = all_samples / (all_utterances * wm.mean() * args.sampling_rate)
        log((), {"avg_vocoder_samples/s": all_samples / wm.sum()})
        log((), {"avg_vocoder_latency": wm.mean()})
        log((), {"avg_vocoder_RTF": rtf})
        log((), {"90%_vocoder_latency": wm.mean() + norm.ppf((1.0 + 0.90) / 2) * wm.std()})
        log((), {"95%_vocoder_latency": wm.mean() + norm.ppf((1.0 + 0.95) / 2) * wm.std()})
        log((), {"99%_vocoder_latency": wm.mean() + norm.ppf((1.0 + 0.99) / 2) * wm.std()})
    if generator is not None and vocoder is not None:
        m = gm + wm
        rtf = all_samples / (all_utterances * m.mean() * args.sampling_rate)
        log((), {"avg_samples/s": all_samples / m.sum()})
        log((), {"avg_letters/s": all_letters / m.sum()})
        log((), {"avg_latency": m.mean()})
        log((), {"avg_RTF": rtf})
        log((), {"90%_latency": m.mean() + norm.ppf((1.0 + 0.90) / 2) * m.std()})
        log((), {"95%_latency": m.mean() + norm.ppf((1.0 + 0.95) / 2) * m.std()})
        log((), {"99%_latency": m.mean() + norm.ppf((1.0 + 0.99) / 2) * m.std()})
    DLLogger.flush()


if __name__ == '__main__':
    main()
