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

import csv
import os
from typing import Optional

import librosa
import numpy as np
import torch
from scipy.io.wavfile import read


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, fnames, delim="|"):
    data_fields = ['audio', 'pitch', 'duration', 'speaker', 'language']
    fields = data_fields + ['text']  # TODO: add fname here
    fpaths_and_text = []
    for fname in fnames:
        with open(fname, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delim, restval=None)
            # return default None for any unspecified fields
            reader.fieldnames.extend(
                i for i in fields if i not in reader.fieldnames)
            for line in reader:
                for k, v in line.items():
                    if k in data_fields and v is not None:
                        data_path = os.path.join(dataset_path, v)
                        if os.path.exists(data_path):
                            line[k] = data_path
                        else:
                            line[k] = v  # pass through speaker/lang ids
                fpaths_and_text.append(line)
    return fpaths_and_text


def load_speaker_lang_ids(fname):
    if fname is None:
        return
    lab2id = {}
    with open(fname) as inf:
        for line in inf:
            lab, id_ = line.strip().split()
            lab2id[lab] = int(id_)
    return lab2id


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def to_device_async(tensor, device):
    return tensor.to(device, non_blocking=True)


def to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x
