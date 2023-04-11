# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from common.stft import STFT


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with hifigan """

    def __init__(self, hifigan, filter_length=512, win_length=512, hop_length=256,
                 sr=22050, n_mel=80, mode='zeros', **infer_kw):
        super(Denoiser, self).__init__()

        w = next(p for name, p in hifigan.named_parameters()
                 if name.endswith('.weight'))

        self.stft = STFT(filter_length=filter_length,
                         hop_length=hop_length,
                         win_length=win_length).to(w.device)

        one_sec = int(sr / hop_length)
        mel_init = {'zeros': torch.zeros, 'normal': torch.randn}[mode]
        mel_input = mel_init((1, n_mel, one_sec), dtype=w.dtype, device=w.device)

        with torch.no_grad():
            bias_audio = hifigan(mel_input, **infer_kw).float()

            if len(bias_audio.size()) > 2:
                bias_audio = bias_audio.squeeze(0)
            elif len(bias_audio.size()) < 2:
                bias_audio = bias_audio.unsqueeze(0)
            assert len(bias_audio.size()) == 2

            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.01):
        audio_spec, audio_angles = self.stft.transform(audio.float())
        audio_spec_denoised = audio_spec - (self.bias_spec * strength)
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
