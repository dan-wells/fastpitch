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
import copy
import glob
import json
import os
import re
import time
import warnings
from collections import defaultdict, OrderedDict

import librosa
import numpy as np
import torch
import torch.distributed as dist
import torch.distributions as D
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_optimizer import Lamb

import common.tb_dllogger as logger
import models
from fastpitch.attn_loss_function import AttentionBinarizationLoss
from fastpitch.data_function import batch_to_gpu, TextMelAliCollate, TextMelAliLoader
from fastpitch.loss_function import FastPitchLoss, FastPitchMASLoss, FastPitchTVCGMMLoss


def parse_args(parser):
    """Parse commandline arguments"""
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')

    train = parser.add_argument_group('training setup')
    train.add_argument('--epochs', type=int, required=True,
                       help='Number of total epochs to run')
    train.add_argument('--epochs-per-checkpoint', type=int, default=50,
                       help='Number of epochs per checkpoint')
    train.add_argument('--checkpoint-path', type=str, default=None,
                       help='Checkpoint path to resume train')
    train.add_argument('--resume', action='store_true',
                       help='Resume train from the last available checkpoint')
    train.add_argument('--seed', type=int, default=1234,
                       help='Seed for PyTorch random number generators')
    train.add_argument('--amp', action='store_true',
                       help='Enable AMP')
    train.add_argument('--cuda', action='store_true',
                       help='Run on GPU using CUDA')
    train.add_argument('--cudnn-benchmark', action='store_true',
                       help='Enable cudnn benchmark mode')
    train.add_argument('--ema-decay', type=float, default=0,
                       help='Discounting factor for train weights EMA')
    train.add_argument('--grad-accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')

    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('--optimizer', type=str, default='lamb', choices=['adam', 'lamb'],
                     help='Optimization algorithm')
    opt.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                     help='Learning rate')
    opt.add_argument('--weight-decay', default=1e-6, type=float,
                     help='Weight decay')
    opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                     help='Clip threshold for gradients')
    opt.add_argument('-bs', '--batch-size', type=int, required=True,
                     help='Batch size (effective, after multi-GPU and gradient accumulation)')
    opt.add_argument('--warmup-steps', type=int, default=1000,
                     help='Number of steps for lr warmup')
    opt.add_argument('--dur-predictor-loss-scale', type=float, default=0.1,
                     help='Rescale duration predictor loss')
    opt.add_argument('--pitch-predictor-loss-scale', type=float, default=0.1,
                     help='Rescale pitch predictor loss')
    opt.add_argument('--attn-loss-scale', type=float, default=1.0,
                     help='Rescale alignment loss')
    opt.add_argument('--kl-loss-weight', type=float, default=1.0,
                     help='Rescale hard attention loss')
    opt.add_argument('--kl-loss-start-epoch', type=int, default=0,
                     help='Start adding the hard attention loss term')
    opt.add_argument('--kl-loss-warmup-epochs', type=int, default=100,
                     help='Gradually increase the hard attention loss term')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--training-files', type=str, nargs='*', required=True,
                      help='Paths to training filelists.')
    data.add_argument('--validation-files', type=str, nargs='*', required=True,
                      help='Paths to validation filelists')
    data.add_argument('--pitch-mean-std-file', type=str, default=None,
                      help='Path to pitch stats to be stored in the model')
    data.add_argument('--input-type', type=str, default='char',
                      choices=['char', 'phone', 'pf', 'unit'],
                      help='Input symbols used, either char (text), phone, pf '
                      '(phonological feature vectors) or unit (quantized acoustic '
                      'representation IDs)')
    data.add_argument('--symbol-set', type=str, default='english_basic',
                      help='Define symbol set for input sequences. For quantized '
                      'unit inputs, pass the size of the vocabulary.')
    data.add_argument('--text-cleaners', type=str, nargs='*', default=[],
                      help='Type of text cleaners for input text.')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Condition on speaker, value > 1 enables trainable '
                      'speaker embeddings.')
    cond.add_argument('--n-langs', type=int, default=1,
                      help='Condition on language, value > 1 enables trainable '
                      'language embeddings.')

    audio = parser.add_argument_group('log generated audio')
    audio.add_argument('--hifigan', type=str, default='',
                       help='Path to HiFi-GAN audio checkpoint')
    audio.add_argument('--hifigan-config', type=str, default='hifigan/config/config_v1.json',
                       help='Path to HiFi-GAN audio config file')
    audio.add_argument('--sampling-rate', type=int, default=22050,
                       help='Sampling rate for output audio')
    audio.add_argument('--hop-length', type=int, default=256,
                       help='STFT hop length for estimating audio length from mel size')
    audio.add_argument('--audio-interval', type=int, default=5,
                       help='Log generated audio and spectrograms every N epochs')

    dist = parser.add_argument_group('distributed training setup')
    dist.add_argument('--master-addr', type=str, default='localhost',
                             help='IP address of machine hosting master process.')
    dist.add_argument('--master-port', type=int, default=13370,
                             help='Free port on machine hosting master process.')
    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def init_distributed(rank, args):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Rank {}: Initializing distributed training".format(rank))
    torch.cuda.set_device(rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(args.num_gpus)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group(init_method='env://', backend='nccl')
    print("Rank {}: Done initializing distributed training".format(rank))


def last_checkpoint(output):
    saved = sorted(
        glob.glob(f'{output}/fastpitch_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            warnings.warn(f'Cannot load {fpath}')
            return True

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None


def maybe_save_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                          total_iter, config):
    if args.local_rank != 0:
        return

    intermediate = (args.epochs_per_checkpoint > 0
                    and epoch % args.epochs_per_checkpoint == 0)

    if not intermediate and epoch < args.epochs:
        return

    fpath = os.path.join(args.output, f"fastpitch_{epoch}.pt")
    print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
    ema_dict = None if ema_model is None else ema_model.state_dict()
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'ema_state_dict': ema_dict,
                  'optimizer': optimizer.state_dict()}
    if args.amp:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, fpath)


def load_checkpoint(args, model, ema_model, optimizer, scaler, epoch,
                    total_iter, config, filepath):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']

    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if args.amp:
        scaler.load_state_dict(checkpoint['scaler'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])


def load_vocoder(args, device):
    """Load HiFi-GAN vocoder from checkpoint"""
    checkpoint_data = torch.load(args.hifigan)
    vocoder_config = models.get_model_config('HiFi-GAN', args)
    vocoder = models.get_model('HiFi-GAN', vocoder_config, device)
    vocoder.load_state_dict(checkpoint_data['generator'])
    vocoder.remove_weight_norm()
    vocoder.eval()
    if args.amp:
        vocoder.half()
    return vocoder


def validate(model, epoch, total_iter, criterion, valset, batch_size, collate_fn,
             distributed_run, batch_to_gpu, use_gt_durations=False, ema=False,
             mas=False, attention_kl_loss=None, kl_weight=None,
             vocoder=None, sampling_rate=22050, hop_length=256, n_mel=80,
             tvcgmm_k=0, audio_interval=5):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()

    tik = time.perf_counter()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=4, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch, collate_fn.symbol_type, mas=mas)
            y_pred = model(x, use_gt_durations=use_gt_durations)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')

            if mas:
                _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
                binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                meta['kl_loss'] = binarization_loss.clone().detach() * kl_weight
                meta['kl_weight'] = kl_weight
                loss += kl_weight * binarization_loss
                meta['align_loss'] = meta['attn_loss'] + meta['kl_loss']

            if distributed_run:
                for k, v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k, v in meta.items():
                    val_meta[k] += v
                val_num_frames = num_frames.item()

            # log spectrograms and generated audio for first few utterances
            if (i == 0) and (epoch % audio_interval == 0 if epoch is not None else True):
                fnames = batch[-1]
                # reorder utterances by mel length
                if mas:
                    tgt_mel_lens = y[2]
                else:
                    tgt_mel_lens = y[1].sum(axis=1)
                tgt_mel_lens_sorted_idx = [
                    i for i, _ in sorted(enumerate(tgt_mel_lens), key=lambda x: x[1], reverse=True)]

                if tvcgmm_k:
                    mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred = y_pred
                    mel_tgt, dur_tgt, dur_lens, pitch_tgt = y
                    mel_tgt = mel_tgt.transpose(1, 2)
                    min_var = 1.0e-3

                    param_preds = mel_out.reshape(*mel_tgt.shape, tvcgmm_k, 10)
                    scale_tril = torch.diag_embed(
                        F.softplus(param_preds[..., 4:7]) + min_var, offset=0)
                    scale_tril += torch.diag_embed(param_preds[..., 7:9], offset=-1)
                    scale_tril += torch.diag_embed(param_preds[..., 9:10], offset=-2)

                    mix = D.Categorical(F.softmax(param_preds[..., 0], dim=-1))
                    comp = D.MultivariateNormal(param_preds[..., 1:4], scale_tril=scale_tril)
                    mixture = D.MixtureSameFamily(mix, comp)

                    mel_pred = mixture.sample().transpose(2, 3)
                    mel_pred = mel_pred.reshape(batch_size, -1, 3 * n_mel).transpose(1, 2)
                    mel_pred[:, :n_mel, 1:] += mel_pred[:, n_mel:2 * n_mel, :-1]
                    mel_pred[:, 1:n_mel, :] += mel_pred[:, 2 * n_mel:-1, :]
                    mel_pred[:, 1:n_mel, 1:] /= 3  # average overlapping bins
                    mel_pred[:, 0, 1:] /= 2
                    mel_pred[:, 1:, 0] /= 2
                    mel_pred = mel_pred[:, :n_mel, :].transpose(1, 2)
                    y_pred = (mel_pred, *y_pred[1:])  # it's a tuple...

                if epoch == audio_interval:
                    # plot ref and copy synthesis only on first epoch
                    plot_spectrograms(
                        y, fnames, tgt_mel_lens_sorted_idx, total_iter,
                        n=4, label='Reference spectrogram', mas=mas)
                    if vocoder is not None:
                        generate_audio(y, fnames, tgt_mel_lens_sorted_idx, total_iter,
                                       vocoder, sampling_rate, hop_length,
                                       n=4, label='Reference audio', mas=mas,
                                       dataset_path=valset.dataset_path)
                        generate_audio(y, fnames, tgt_mel_lens_sorted_idx, total_iter,
                                       vocoder, sampling_rate, hop_length,
                                       n=4, label='Copy synthesis', mas=mas)
                plot_spectrograms(
                    y_pred, fnames, tgt_mel_lens_sorted_idx, total_iter,
                    n=4, label='Predicted spectrogram', mas=mas)
                if vocoder is not None:
                    generate_audio(y_pred, fnames, tgt_mel_lens_sorted_idx, total_iter,
                                   vocoder, sampling_rate, hop_length, n=4,
                                   label='Predicted audio', mas=mas)
                if mas:
                    plot_attn_maps(
                        y_pred, fnames, tgt_mel_lens_sorted_idx, total_iter,
                        n=4, label='Predicted alignment')

        val_meta = {k: v / len(valset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik

    log_stdout(logger,
               'val_ema' if ema else 'val',
               (epoch,) if epoch is not None else (),
               total_iter,
               val_meta['loss'].item(),
               val_meta['mel_loss'].item(),
               val_meta['duration_predictor_loss'].item(),
               val_meta['pitch_loss'].item(),
               None if not mas else val_meta['align_loss'].item(),
               val_meta['took']
    )

    if was_training:
        model.train()
    return val_meta


def log_stdout(logger, subset, epoch_iters, total_steps, loss, mel_loss,
               dur_loss, pitch_loss, align_loss, took):
    logger_data = [
        ('Loss/Total', loss),
        ('Loss/Mel', mel_loss),
        ('Loss/Duration', dur_loss),
        ('Loss/Pitch', pitch_loss),
        #('Error/Duration', iter_dur_error),
        #('Error/Pitch', iter_pitch_error),
        #('Time/FPS', iter_num_frames / iter_time),
        # only relevant per step, not averaged over epoch
        #('Hyperparameters/Learning rate', optimizer.param_groups[0]['lr']),
    ]
    if align_loss is not None:
        logger_data.extend([
            ('Loss/Alignment', align_loss),
            #('Align/Attention loss', iter_attn_loss),
            #('Align/KL loss', iter_kl_loss),
            #('Align/KL weight', iter_kl_weight),  # step, not avg
        ])
    logger_data.append(('Time/Iter time', took))
    logger.log(epoch_iters,
               tb_total_steps=total_steps,
               subset=subset,
               data=OrderedDict(logger_data)
    )


def plot_spectrograms(y, fnames, sorted_idx, step, n=4, label='Predicted spectrogram', mas=False):
    """Plot spectrograms for n utterances in batch"""
    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    idx = sorted_idx[::s]
    fnames = [fnames[i] for i in idx]
    if label == 'Predicted spectrogram':
        # y: mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred
        mel_specs = y[0][idx].transpose(1, 2).cpu().numpy()
        mel_lens = y[1][idx].squeeze().cpu().numpy().sum(axis=1) - 1
    elif label == 'Reference spectrogram':
        # y: mel_padded, dur_padded, dur_lens, pitch_padded
        mel_specs = y[0][idx].cpu().numpy()
        if mas:
            mel_lens = y[2][idx].cpu().numpy()  # output_lengths
        else:
            mel_lens = y[1][idx].cpu().numpy().sum(axis=1) - 1
    for mel_spec, mel_len, fname in zip(mel_specs, mel_lens, fnames):
        mel_spec = mel_spec[:, :mel_len]
        utt_id = os.path.splitext(os.path.basename(fname))[0]
        logger.log_spectrogram_tb(
            step, '{}/{}'.format(label, utt_id), mel_spec, tb_subset='val')


def generate_audio(y, fnames, sorted_idx, step, vocoder=None, sampling_rate=22050, hop_length=256,
                   n=4, label='Predicted audio', mas=False, dataset_path=''):
    """Generate audio from spectrograms for n utterances in batch"""
    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    idx = sorted_idx[::s]
    fnames = [fnames[i] for i in idx]
    with torch.no_grad():
        if label == 'Predicted audio':
            # y: mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred
            audios = vocoder(y[0][idx].transpose(1, 2)).cpu().squeeze().numpy()
            mel_lens = y[1][idx].squeeze().cpu().numpy().sum(axis=1) - 1
        else:
            # y: mel_padded, dur_padded, dur_lens, pitch_padded
            if label == 'Copy synthesis':
                audios = vocoder(y[0][idx]).cpu().squeeze().numpy()
            elif label == 'Reference audio':
                audios = []
                for fname in fnames:
                    wav = os.path.join(dataset_path, 'wavs/{}.wav'.format(fname))
                    audio, _ = librosa.load(wav, sr=sampling_rate)
                    audios.append(audio)
            if mas:
                mel_lens = y[2][idx].cpu().numpy()  # output_lengths
            else:
                mel_lens = y[1][idx].cpu().numpy().sum(axis=1) - 1
    for audio, mel_len, fname in zip(audios, mel_lens, fnames):
        audio = audio[:mel_len * hop_length]
        audio = audio / np.max(np.abs(audio))
        logger.log_audio_tb(
            step, '{}/{}'.format(label, fname), audio, sampling_rate, tb_subset='val')


def plot_attn_maps(y, fnames, sorted_idx, step, n=4, label='Predicted alignment'):
    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    idx = sorted_idx[::s]
    fnames = [fnames[i] for i in idx]
    _, dec_mask, *_, attn_softs, attn_hards, attn_hard_durs, _ = y
    attn_softs = attn_softs[idx].cpu().numpy()
    attn_hards = attn_hards[idx].cpu().numpy()
    attn_hard_durs = attn_hard_durs[idx].cpu().numpy()
    text_lens = np.count_nonzero(attn_hard_durs, 1)
    mel_lens = dec_mask[idx].cpu().numpy().squeeze(2).sum(1)
    for attn_soft, attn_hard, mel_len, text_len, fname in zip(
            attn_softs, attn_hards, mel_lens, text_lens, fnames):
        attn_soft = attn_soft[:,:mel_len,:text_len].squeeze(0).transpose()
        attn_hard = attn_hard[:,:mel_len,:text_len].squeeze(0).transpose()
        utt_id = os.path.splitext(os.path.basename(fname))[0]
        logger.log_attn_maps_tb(
            step, '{}/{}'.format(label, utt_id), attn_soft, attn_hard, tb_subset='val')


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def apply_ema_decay(model, ema_model, decay):
    st = model.state_dict()
    add_module = hasattr(model, 'module') and not hasattr(ema_model, 'module')
    for k, v in ema_model.state_dict().items():
        if add_module and not k.startswith('module.'):
            k = 'module.' + k
        v.copy_(decay * v + (1 - decay) * st[k])


def train(rank, args):
    args.local_rank = rank
    if args.local_rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    log_fpath = args.log_file or os.path.join(args.output, 'nvlog.json')
    tb_subsets = ['train', 'val']
    if args.ema_decay > 0.0:
        tb_subsets.append('val_ema')

    logger.init(log_fpath, args.output, enabled=(args.local_rank == 0),
                tb_subsets=tb_subsets)
    logger.parameters(vars(args), tb_subset='train')

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.distributed_run:
        init_distributed(rank, args)

    device = torch.device('cuda' if args.cuda else 'cpu')
    model_config = models.get_model_config('FastPitch', args)
    model = models.get_model('FastPitch', model_config, device, forward_mas=args.use_mas)

    # Store pitch mean/std as params to translate from Hz during inference
    with open(args.pitch_mean_std_file, 'r') as f:
        stats = json.load(f)
    model.pitch_mean[0] = stats['mean']
    model.pitch_std[0] = stats['std']

    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), **kw)
    elif args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), **kw)

    scaler = GradScaler(enabled=args.amp)

    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
    else:
        ema_model = None

    if args.distributed_run:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
            #find_unused_parameters=True)

    start_epoch = [1]
    start_iter = [0]

    assert args.checkpoint_path is None or args.resume is False, (
        "Specify a single checkpoint source")
    if args.checkpoint_path is not None:
        ch_fpath = args.checkpoint_path
    elif args.resume:
        ch_fpath = last_checkpoint(args.output)
    else:
        ch_fpath = None

    if ch_fpath is not None:
        load_checkpoint(args, model, ema_model, optimizer, scaler,
                        start_epoch, start_iter, model_config, ch_fpath)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]

    kl_weight = None
    attention_kl_loss = None  # for validation
    if args.use_mas:
        criterion = FastPitchMASLoss(
            dur_predictor_loss_scale=args.dur_predictor_loss_scale,
            pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
            attn_loss_scale=args.attn_loss_scale)
        attention_kl_loss = AttentionBinarizationLoss()  # L_bin
    elif args.tvcgmm_k:
        criterion = FastPitchTVCGMMLoss(k=args.tvcgmm_k, min_var=1.0e-3,
            dur_predictor_loss_scale=args.dur_predictor_loss_scale,
            pitch_predictor_loss_scale=args.pitch_predictor_loss_scale)
    else:
        criterion = FastPitchLoss(
            dur_predictor_loss_scale=args.dur_predictor_loss_scale,
            pitch_predictor_loss_scale=args.pitch_predictor_loss_scale)

    trainset = TextMelAliLoader(audiopaths_and_text=args.training_files, **vars(args))
    valset = TextMelAliLoader(audiopaths_and_text=args.validation_files, **vars(args))

    collate_fn = TextMelAliCollate(
        symbol_type=args.input_type, n_symbols=trainset.n_symbols, mas=args.use_mas)

    if args.distributed_run:
        train_sampler, shuffle = DistributedSampler(trainset), False
    else:
        train_sampler, shuffle = None, True

    train_loader = DataLoader(
        trainset, num_workers=4, shuffle=shuffle, sampler=train_sampler,
        batch_size=int(args.batch_size / args.grad_accumulation),
        pin_memory=False, drop_last=True, collate_fn=collate_fn)

    vocoder = None
    if args.hifigan:
        vocoder = load_vocoder(args, device)

    model.train()

    if args.cuda:
        torch.cuda.synchronize()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_dur_loss = 0.0
        epoch_pitch_loss = 0.0
        epoch_align_loss = 0.0
        epoch_attn_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_dur_error = 0.0
        epoch_pitch_error = 0.0
        epoch_num_frames = 0
        epoch_frames_per_sec = 0.0

        if args.distributed_run:
            train_loader.sampler.set_epoch(epoch)

        accumulated_steps = 0
        iter_loss = 0
        iter_num_frames = 0
        iter_meta = {}

        epoch_iter = 0
        num_iters = len(train_loader) // args.grad_accumulation
        for batch in train_loader:

            if accumulated_steps == 0:
                if epoch_iter == num_iters:
                    break
                total_iter += 1
                epoch_iter += 1
                iter_start_time = time.perf_counter()

                adjust_learning_rate(total_iter, optimizer, args.learning_rate,
                                     args.warmup_steps)

                model.zero_grad()

            x, y, num_frames = batch_to_gpu(batch, args.input_type, args.use_mas)

            with autocast(enabled=args.amp):
                y_pred = model(x, use_gt_durations=True)
                loss, meta = criterion(y_pred, y)
                
                if args.use_mas:
                    if epoch >= args.kl_loss_start_epoch:
                        _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
                        binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                        kl_weight = min((epoch - args.kl_loss_start_epoch) / args.kl_loss_warmup_epochs,
                                        1.0) * args.kl_loss_weight
                        meta['kl_loss'] = binarization_loss.clone().detach() * kl_weight
                        meta['kl_weight'] = kl_weight
                        loss += kl_weight * binarization_loss
                    else:
                        meta['kl_loss'] = torch.zeros_like(loss)
                        meta['kl_weight'] = 0
                        kl_weight = 0
                        binarization_loss = 0
                    meta['align_loss'] = meta['attn_loss'] + meta['kl_loss']

                loss /= args.grad_accumulation

            meta = {k: v / args.grad_accumulation
                    for k, v in meta.items()}

            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.distributed_run:
                reduced_loss = reduce_tensor(loss.data, args.num_gpus).item()
                reduced_num_frames = reduce_tensor(num_frames.data, 1).item()
                meta = {k: reduce_tensor(v, args.num_gpus) for k, v in meta.items()}
            else:
                reduced_loss = loss.item()
                reduced_num_frames = num_frames.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            accumulated_steps += 1
            iter_loss += reduced_loss
            iter_num_frames += reduced_num_frames
            iter_meta = {k: iter_meta.get(k, 0) + meta.get(k, 0) for k in meta}

            if accumulated_steps % args.grad_accumulation == 0:
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    optimizer.step()
                logger.log_grads_tb(total_iter, model)

                if args.ema_decay:
                    apply_ema_decay(model, ema_model, args.ema_decay)

                iter_time = time.perf_counter() - iter_start_time
                iter_mel_loss = iter_meta['mel_loss'].item()
                iter_dur_loss = iter_meta['duration_predictor_loss'].item()
                iter_pitch_loss = iter_meta['pitch_loss'].item()
                iter_dur_error = iter_meta['duration_error'].item()
                iter_pitch_error = iter_meta['pitch_error'].item()
                epoch_loss += iter_loss
                epoch_mel_loss += iter_mel_loss
                epoch_dur_loss += iter_dur_loss
                epoch_pitch_loss += iter_pitch_loss
                epoch_dur_error += iter_dur_error
                epoch_pitch_error += iter_pitch_error
                epoch_num_frames += iter_num_frames
                epoch_frames_per_sec += iter_num_frames / iter_time
                if args.use_mas:
                    iter_align_loss = iter_meta['align_loss'].item()
                    iter_attn_loss = iter_meta['attn_loss'].item()
                    iter_kl_loss = iter_meta['kl_loss'].item()
                    iter_kl_weight = iter_meta['kl_weight']
                    epoch_align_loss += iter_align_loss
                    epoch_attn_loss += iter_attn_loss
                    epoch_kl_loss += iter_kl_loss

                log_stdout(logger,
                           'train',
                           (epoch, epoch_iter, num_iters),
                           total_iter,
                           iter_loss,
                           iter_mel_loss,
                           iter_dur_loss,
                           iter_pitch_loss,
                           None if not args.use_mas else iter_align_loss,
                           iter_time
                )

                accumulated_steps = 0
                iter_loss = 0
                iter_num_frames = 0
                iter_meta = {}

        # Finished epoch
        epoch_time = time.perf_counter() - epoch_start_time

        log_stdout(logger,
                   'train_avg',
                   (epoch,),
                   None,
                   epoch_loss / epoch_iter,
                   epoch_mel_loss / epoch_iter,
                   epoch_dur_loss / epoch_iter,
                   epoch_pitch_loss / epoch_iter,
                   None if not args.use_mas else epoch_align_loss / epoch_iter,
                   epoch_time
        )

        validate(model, epoch, total_iter, criterion, valset, args.batch_size,
            collate_fn, args.distributed_run, batch_to_gpu, use_gt_durations=True,
            mas=args.use_mas, attention_kl_loss=attention_kl_loss, kl_weight=kl_weight,
            vocoder=vocoder, sampling_rate=args.sampling_rate, hop_length=args.hop_length,
            n_mel=args.n_mel_channels, tvcgmm_k=args.tvcgmm_k, audio_interval=args.audio_interval)

        if args.ema_decay > 0:
            validate(ema_model, epoch, total_iter, criterion, valset, args.batch_size,
                collate_fn, args.distributed_run, batch_to_gpu, use_gt_durations=True, ema=True,
                mas=args.use_mas, attention_kl_loss=attention_kl_loss, kl_weight=kl_weight,
                vocoder=vocoder, sampling_rate=args.sampling_rate, hop_length=args.hop_length,
                n_mel=args.n_mel_channels, tvcgmm_k=args.tvcgmm_k, audio_interval=args.audio_interval)

        maybe_save_checkpoint(args, model, ema_model, optimizer, scaler,
                              epoch, total_iter, model_config)
        logger.flush()

    # Finished training
    log_stdout(logger,
               'train_avg',
               (),
               None,
               epoch_loss / epoch_iter,
               epoch_mel_loss / epoch_iter,
               epoch_dur_loss / epoch_iter,
               epoch_pitch_loss / epoch_iter,
               None if not args.use_mas else epoch_align_loss / epoch_iter,
               epoch_time
    )
    validate(model, None, total_iter, criterion, valset, args.batch_size,
        collate_fn, args.distributed_run, batch_to_gpu, use_gt_durations=True,
        mas=args.use_mas, attention_kl_loss=attention_kl_loss, kl_weight=kl_weight,
        vocoder=vocoder, sampling_rate=args.sampling_rate, hop_length=args.hop_length,
        n_mel=args.n_mel_channels, tvcgmm_k=args.tvcgmm_k)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    parser = models.parse_model_args('FastPitch', parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    # TODO: refactor loss/forwards to support TVC-GMM under MAS
    if args.use_mas and args.tvcgmm_k:
        raise NotImplementedError(
            'TVC-GMM prediction is not supported when training with MAS')

    if args.cuda:
        args.num_gpus = torch.cuda.device_count()
        args.distributed_run = args.num_gpus > 1
        args.batch_size = int(args.batch_size / args.num_gpus)
    else:
        args.distributed_run = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.distributed_run:
        mp.spawn(train, nprocs=args.num_gpus, args=(args,))
    else:
        train(0, args)


if __name__ == '__main__':
    main()
