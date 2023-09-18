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

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn

from common.utils import mask_from_lens
from fastpitch.attn_loss_function import AttentionCTCLoss


class FastPitchLoss(nn.Module):
    def __init__(self, dur_predictor_loss_scale=1.0,
                 pitch_predictor_loss_scale=1.0):
        super(FastPitchLoss, self).__init__()
        self.dur_predictor_loss_scale = dur_predictor_loss_scale
        self.pitch_predictor_loss_scale = pitch_predictor_loss_scale

    def forward(self, model_out, targets, is_training=True, meta_agg='mean'):
        mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred = model_out
        mel_tgt, dur_tgt, dur_lens, pitch_tgt = targets
        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)

        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))
        log_dur_tgt = torch.log(dur_tgt.float() + 1)
        loss_fn = F.mse_loss
        dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction='none')
        dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()

        ldiff = mel_tgt.size(1) - mel_out.size(1)
        mel_out = F.pad(mel_out, (0, 0, 0, ldiff, 0, 0), value=0.0)
        mel_mask = mel_tgt.ne(0).float()
        loss_fn = F.mse_loss
        mel_loss = loss_fn(mel_out, mel_tgt, reduction='none')
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()

        ldiff = pitch_tgt.size(1) - pitch_pred.size(1)
        pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
        pitch_loss = (pitch_loss * dur_mask).sum() / dur_mask.sum()

        loss = mel_loss
        loss = (mel_loss + pitch_loss * self.pitch_predictor_loss_scale
                + dur_pred_loss * self.dur_predictor_loss_scale)

        meta = {
            'loss': loss.clone().detach(),
            'mel_loss': mel_loss.clone().detach(),
            'duration_predictor_loss': dur_pred_loss.clone().detach(),
            'pitch_loss': pitch_loss.clone().detach(),
            'duration_error': (torch.abs(dur_pred - dur_tgt).sum()
                          / dur_mask.sum()).detach(),
            'pitch_error': (torch.abs(pitch_pred - pitch_tgt).sum()
                          / dur_mask.sum()).detach(),
        }
        assert meta_agg in ('sum', 'mean')
        if meta_agg == 'sum':
            bsz = mel_out.size(0)
            meta = {k: v * bsz for k,v in meta.items()}
        return loss, meta


class FastPitchMASLoss(nn.Module):
    def __init__(self, dur_predictor_loss_scale=1.0,
                 pitch_predictor_loss_scale=1.0, attn_loss_scale=1.0):
        super(FastPitchMASLoss, self).__init__()
        self.dur_predictor_loss_scale = dur_predictor_loss_scale
        self.pitch_predictor_loss_scale = pitch_predictor_loss_scale
        self.attn_loss_scale = attn_loss_scale
        self.attn_ctc_loss = AttentionCTCLoss()

    def forward(self, model_out, targets, is_training=True, meta_agg='mean'):
        (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt,
         attn_soft, attn_hard, attn_dur, attn_logprob) = model_out

        (mel_tgt, in_lens, out_lens) = targets

        dur_tgt = attn_dur
        dur_lens = in_lens

        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)

        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))
        log_dur_tgt = torch.log(dur_tgt.float() + 1)
        loss_fn = F.mse_loss
        dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction='none')
        dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()

        ldiff = mel_tgt.size(1) - mel_out.size(1)
        mel_out = F.pad(mel_out, (0, 0, 0, ldiff, 0, 0), value=0.0)
        mel_mask = mel_tgt.ne(0).float()
        loss_fn = F.mse_loss
        mel_loss = loss_fn(mel_out, mel_tgt, reduction='none')
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()

        ldiff = pitch_tgt.size(1) - pitch_pred.size(1)
        pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
        pitch_loss = (pitch_loss * dur_mask).sum() / dur_mask.sum()

        # Attention loss (L_ForwardSum)
        attn_loss = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)

        loss = (mel_loss
                + dur_pred_loss * self.dur_predictor_loss_scale
                + pitch_loss * self.pitch_predictor_loss_scale
                + attn_loss * self.attn_loss_scale)

        meta = {
            'loss': loss.clone().detach(),
            'mel_loss': mel_loss.clone().detach(),
            'duration_predictor_loss': dur_pred_loss.clone().detach(),
            'pitch_loss': pitch_loss.clone().detach(),
            'attn_loss': attn_loss.clone().detach(),
            'duration_error': (torch.abs(dur_pred - dur_tgt).sum()
                          / dur_mask.sum()).detach(),
            'pitch_error': (torch.abs(pitch_pred - pitch_tgt).sum()
                          / dur_mask.sum()).detach(),
        }

        assert meta_agg in ('sum', 'mean')
        if meta_agg == 'sum':
            bsz = mel_out.size(0)
            meta = {k: v * bsz for k, v in meta.items()}
        return loss, meta


class FastPitchTVCGMMLoss(nn.Module):
    def __init__(self, k=5, min_var=1.0e-3, 
                 dur_predictor_loss_scale=1.0,
                 pitch_predictor_loss_scale=1.0):
        super(FastPitchTVCGMMLoss, self).__init__()
        self.k = k
        self.min_var = min_var
        self.dur_predictor_loss_scale = dur_predictor_loss_scale
        self.pitch_predictor_loss_scale = pitch_predictor_loss_scale

    def forward(self, model_out, targets, is_training=True, meta_agg='mean'):
        mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred = model_out
        mel_tgt, dur_tgt, dur_lens, pitch_tgt = targets
        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)

        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))
        log_dur_tgt = torch.log(dur_tgt.float() + 1)
        loss_fn = F.mse_loss
        dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction='none')
        dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()

        param_predictions = mel_out.reshape(*mel_tgt.shape, self.k, 10)[:, :dec_mask.shape[1]]
        # in practice we predict the scale_tril (lower triangular factor of the covariance matrix)
        # we predict the parameters for every t,f bin
        # at every bin we predict the joint distribution of t,f t+1,f and t,f+1
        # --> later in sampling we have overlap of one bin with the next time and the next freq bin
        scale_tril = torch.diag_embed(
            nn.functional.softplus(param_predictions[..., 4:7]) + self.min_var, offset=0)
        scale_tril += torch.diag_embed(param_predictions[..., 7:9], offset=-1)
        scale_tril += torch.diag_embed(param_predictions[..., 9:10], offset=-2)

        mix = D.Categorical(nn.functional.softmax(param_predictions[..., 0], dim=-1))
        comp = D.MultivariateNormal(param_predictions[..., 1:4], scale_tril=scale_tril)
        mixture = D.MixtureSameFamily(mix, comp)

        mel_multivariate_targets = torch.zeros([*mel_tgt.shape, 3], device=mel_tgt.device)
        mel_multivariate_targets[..., 0] = mel_tgt # spectrogram
        mel_multivariate_targets[..., :-1, :, 1] = mel_tgt[..., 1:, :] # t shifted spectrogram
        mel_multivariate_targets[..., :, :-1, 2] = mel_tgt[..., :, 1:] # f shifted spectrogram
        mel_loss = -mixture.log_prob(mel_multivariate_targets).masked_select(dec_mask).mean()

        ldiff = pitch_tgt.size(1) - pitch_pred.size(1)
        pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
        pitch_loss = (pitch_loss * dur_mask).sum() / dur_mask.sum()

        loss = mel_loss
        loss = (mel_loss + pitch_loss * self.pitch_predictor_loss_scale
                + dur_pred_loss * self.dur_predictor_loss_scale)

        meta = {
            'loss': loss.clone().detach(),
            'mel_loss': mel_loss.clone().detach(),
            'duration_predictor_loss': dur_pred_loss.clone().detach(),
            'pitch_loss': pitch_loss.clone().detach(),
            'duration_error': (torch.abs(dur_pred - dur_tgt).sum()
                          / dur_mask.sum()).detach(),
            'pitch_error': (torch.abs(pitch_pred - pitch_tgt).sum()
                          / dur_mask.sum()).detach(),
        }
        assert meta_agg in ('sum', 'mean')
        if meta_agg == 'sum':
            bsz = mel_out.size(0)
            meta = {k: v * bsz for k,v in meta.items()}
        return loss, meta
