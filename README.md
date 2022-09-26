# FastPitch

This repository is based on NVIDIA's reference implementation of FastPitch,
extracted from their [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) repository.

## Data preparation

FastPitch learns to predict mel-scale spectrograms from input symbol sequences
(e.g. text or phones), with explicit duration and pitch prediction per symbol.
For example, you can use `prepare_dataset.py` to extract target features given a
list of audio files and corresponding forced alignments:

```sh
python prepare_dataset.py \
  --dataset-path $DATA_ROOT \
  --wav-text-filelist $FILELIST \
  --durations-from textgrid
```

`$DATA_ROOT` is the directory where all derived features will be stored, in
subdirectories `mels`, `pitches` and `durations`. A file listing global pitch
mean and standard deviation will also be written here, to
`$DATA_ROOT/pitches_stats__${FILELIST_STEM}.json`.

You should make all audio files in `$FILELIST` accessible under
`$DATA_ROOT/wavs`, and corresponding forced alignments represented as Praat
TextGrids under `$DATA_ROOT/TextGrid`. `$FILELIST` should contain audio
filenames and transcripts in your desired symbol set, with lines like:

```
/path/to/data_root/wavs/audio1.wav|this is a text transcript
```

The expected path to the TextGrid file representing alignment information for
this utterance is `/path/to/data_root/TextGrid/audio1.TextGrid`.

Use the `--output-meta-file` option to write a metadata file collecting paths to
all extracted features and preprocessed transcripts per utterance which can be
passed directly to `train.py` for model training. Alternatively, you can put
this file together yourself, with lines like:

```
mels/audio1.pt|durations/audio1.pt|pitches/audio1.pt|<transcript>[|<speaker_id>]
```

Note that paths to feature files are relative to the provided `--dataset-path`,
which will also be passed to `train.py`. Integer speaker IDs are an optional
field in case you want to train a multi-speaker model.

### Supported input representations

Use the `--input-type` and `--symbol-set` options to `prepare_dataset.py` to
specify the input symbols you are using. In the list below, top-level bullets
are possible values for `--input-type` and second-level for `--symbol-set`:

- `char` (raw text)
    * `english_basic`
    * `english_basic_lowercase`
    * `english_expanded`
    * `english_basic_sil`
- `phone` or `pf` (phonological feature vectors)
    * `arpabet`
    * `combilex`
    * `unisyn`
    * `globalphone` (French subset)
    * `ipa`
    * `ipa_all`
    * `xsampa`
    * `english_basic_sil`
- `unit` (integer symbol IDs)
    * Size of symbol vocabulary

See `common/text/symbols.py` for definitions of symbol sets corresponding to
each combination of options above.

Additional notes:

- `char` input can just look like regular English text with spaces between
  words, normalized to whatever degree makes sense given the range of characters
  in your chosen `--symbol-set`
    * Check `common/text/cleaners.py` and the `--text-cleaners` option for
      built-in text normalization options
    * You might want to insert additional space characters at the beginning and
      end of your text transcripts to represent any leading or trailing silence
      in the audio (especially if using  monotonic alignment search for duration
      targets). Pass `add_spaces=True` when setting up your `TextProcessor` to
      do this automatically.
- `phone`, `pf` and `unit` inputs should probably be individual symbols
  separated by spaces, e.g. an `xsampa` rendering of the phrase 'the cat'
  would have a transcript like `D @ k a t`
    * Some support is possible for phone transcripts which may be more
      human-readable when using `ipa{,_all}`, thanks to the way
      [PanPhon](https://github.com/dmort27/panphon)
      handles such strings, e.g. transcripts like `ðə kæt`
- If you want to use phonological feature vectors as input (`--input-type pf`),
  transcripts should be phone strings, with symbols specified by `--symbol-set`
  as above. We convert each symbol to PF vectors using PanPhon.
- `unit` is intended for use with integer symbol IDs, e.g. k-means acoustic
  cluster IDs extracted from raw audio using a self-supervised model such as
  [HuBERT](https://ieeexplore.ieee.org/document/9414460)
    * Pass the size of the symbol vocabulary in `--symbol-set`; if this is e.g.
      100, we expect transcripts to be integer sequences like `79 3 14 14 25`,
      where possible individual symbol values are in the range [0, 99]
- `english_basic_sil` is a special case: it is intended for use with
  character-level alignments, where the pronunciation of each word is
  represented by the characters in that word separated by spaces (e.g. 'cat'
  &rarr; `c a t`), and also includes symbols for silence and short pauses
  between words. This symbol set can be used in two ways:
    * With `char` inputs, pass `handle_sil=True` to your `TextProcessor`
      to treat silence phones as single tokens while still splitting other
      words into character sequences.
    * With `phone` inputs, where your input texts should be preprocessed to
      insert spaces between every character.

### Acoustic feature extraction

Mel spectrogram feature extraction is defined by several parameters passed to
`prepare_dataset.py`:

- `--sampling-rate`, the sampling rate of your audio data (default 22050 Hz)
- `--filter-length`, the number of STFT frequency bins used (default 512)
- `--hop-length`, frame shift of the STFT analysis window (default 256 samples)
- `--win-length`, length of signal attenuation window (default 512)
- `--n-mel-channels`, number of bins in mel-scale filter bank (default 80)
- `--mel-f{min,max}`, minimum and maximum frequency of mel filter bank bins
  (defaults 0, 8000)

These parameters define the frequency and time resolution of acoustic feature
extraction. Each frequency bin in the STFT analysis window spans `sampling-rate /
filter-length` Hz, and each analysis window covers `filter-length /
sampling-rate` seconds of audio. For the default values specified above, this
gives 22050 / 512 = 43 Hz frequency resolution and 512 / 22050 = 23 ms
analysis windows. The frame shift moves the analysis window `hop-length /
sampling-rate` seconds forward each frame, for a stride of 256 / 22050 = 11 ms
(~50% overlap between adjacent frames). For efficiency, `filter-length` should
be some power of 2, and in general `win-length` should match `filter-length`.

See [this document](http://www.add.ece.ufl.edu/4511/references/ImprovingFFTResoltuion.pdf)
for additional discussion of STFT parameters.

If you want to use HuBERT unit sequences as input, then you will need to match
the 50 Hz framerate of that feature extraction process by adjusting the
parameters listed above. If there is a mismatch, then the lengths of your mel
spectrograms will not line up with the durations calculated by run-length
encoding framewise HuBERT codes. For example, if your audio data is sampled at
16 kHz, use `--hop-length 320` for a 20 ms frame shift, i.e. 50 Hz framerate.

Additional options are available for applying peak normalization to audio data
(for example if it was collected across multiple recording sessions) or for
trimming excessive leading or trailing silences found during forced alignment.

### Duration targets

We support three methods for providing duration targets during training:

- Reading segment durations from forced alignments provided in
  Praat TextGrid format
- Run-length encoding frame-level input symbol sequences
- Monotonic alignment search without explicit targets

**Forced alignment:**
Given audio files, transcripts and a pronunciation lexicon (mapping words to
phone strings or just character sequences, depending on your desired input), you
could generate the required [TextGrid alignment files](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html)
per utterance using a tool such as the [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
or our own [KISS Aligner](https://github.com/dan-wells/kiss-aligner).
Then, pass `--durations-from textgrid` to `prepare_dataset.py` to extract
durations per input symbol and save to disk.
This approach is similar to that in the
[original FastPitch paper](https://ieeexplore.ieee.org/abstract/document/9413889).

**Run-length encoding:**
Alternatively, we can extract durations from repeated sequences of input symbols
also specified at the frame level by run-length encoding. This is the expected
method to extract symbol-level duration targets from HuBERT code sequences when
using `--input-type unit`, for example. Pass `--durations-from unit_rle` to use
this method.

**Monotonic alignment search:**
Instead of providing explicit duration targets, we can also use monotonic
alignment search (MAS) to learn the correspondence between input symbols and
acoustic features during training. This approach follows the implementation from
FastPitch 1.1 in the source repo, as described in [this paper](https://ieeexplore.ieee.org/document/9747707).
Pass `--durations-from attn_prior` to `prepare_dataset.py` and `--use-mas` to
`train.py` to use this method. In this case, diagonal attention priors are saved
to disk in the '`durations`' data directory for each utterance.

### Pitch estimation

We default to the [YIN algorithm](https://librosa.org/doc/main/generated/librosa.yin.html)
for fundamental frequency estimation. Framewise estimates are averaged per input
symbol for easier interpretation and more stable performance.

We also have an option to use the more accurate [probabilistic YIN](https://librosa.org/doc/main/generated/librosa.pyin.html),
but this algorithm runs consderably slower. If you know the expected pitch range
in your data, then you can narrow the corresponding hypothesis space for pYIN by
adjusting `--pitch-f{min,max}` (defaults 50--600 Hz) to speed things up a bit.

Framewise pitch values are averaged per input symbol to provide pitch targets
during training. This is done during data preprocessing when extracting target
durations from TextGrid alignments or by run-length encoding input symbol
sequences, and the character-level pitch values are saved to disk in this case.
When training with MAS, we save frame-level pitch values to disk and average
them online during training according to discovered alignments.

## Model training

To train a phone-based system from IPA transcripts, after preparing a dataset
and splitting into train and validation sets:

```sh
python train.py \
  --dataset-path $DATA_ROOT \
  --output $CHECKPOINT_DIR \
  --training-files $DATA_ROOT/train_meta.txt \
  --validation-files $DATA_ROOT/val_meta.txt \
  --pitch-mean-std-file $DATA_ROOT/pitches_stats__${FILELIST_STEM}.json \
  --input-type phone \
  --symbol-set ipa \
  --epochs 100 \
  --epochs-per-checkpoint 10 \
  --batch-size 16 \
  --cuda
```

Model checkpoints will be saved to `$CHECKPOINT_DIR` every 10 epochs, alongside
TensorBoard logs. Make sure to pass `--cuda` to run on GPU if available.

For multi-speaker data, specify `--n-speakers N` to learn additional speaker
embeddings. Input metadata files should then include integer speaker IDs as the
final field on each line, with IDs ranging between [0, `n-speakers` - 1]. You
can assign integer speaker IDs using `scripts/add_speaker_id_to_meta.py`.

To train using monotonic alignment search instead of passing explicit input
symbol duration targets, pass `--use-mas`. 

To reduce model size with (probably) limited impact on performance, pass
`--use-sepconv` to replace all convolutional layers with depthwise separable
convolutions. Additional options are available for automatic mixed precision
(AMP) and gradient accumulation. We also support data-parallel distributed
training at least on a single node -- just set `CUDA_VISIBLE_DEVICES` and point
to a free port on your machine using `--master-{addr,port}`.

## Synthesizing speech

After training has completed, we can predict mel spectrograms from test
transcripts, and optionally generate speech audio using a separate vocoder
model.

First, prepare an input metadata file. This should be a TSV file with a header
row indicating whichever of the following fields you care to provide (so order does
not matter):

- `text`, transcripts of test utterances
- `output`, path to save synthesized speech audio
- `mel_output`, path to save predicted mel spectrogram features
- `speaker`, integer speaker ID per utterance
- `mel`, path to load mel spectrogram features, e.g. reference values for copy
  synthesis
- `pitch`, path to load reference pitch values
- `duration`, path to load reference duration values

To synthesize speech from IPA phone transcripts using the final checkpoint from
our `train.py` run above, using a pre-trained HiFi-GAN vocoder:

```sh
python inference.py \
  --input $DATA_ROOT/test_meta.tsv \
  --output $OUTPUT_DIR \
  --fastpitch $CHECKPOINT_DIR/FastPitch_checkpoint_100.pt \
  --input-type phone \
  --symbol-set ipa \
  --hifigan $PATH_TO_HIFIGAN_CHECKPOINT \
  --hifigan-config hifigan/config/config_v1.json \  # 22.05 kHz audio
  --cuda
```

If `test_meta.tsv` includes an `output` field, synthesized speech will be saved
to corresponding WAV files under `$OUTPUT_DIR`. Otherwise, audio files will be
saved to sequential files named like `$OUTPUT_DIR/audio_1.wav`.

Use the `--sampling-rate` option to ensure output audio files are written at the
correct sampling rate. Predicted mel spectrograms are trimmed to remove noise
(introduced by batching multiple utterances to a fixed length), with target
durations calculated using `--stft-hop-length`, which should match the value of 
`--hop-length` passed to `prepare_dataset.py`.

For a multi-speaker FastPitch model, pass `--n-speakers` to match the value used
with `train.py` and specify a target speaker ID using `--speaker N` to
synthesize all utterances in the same target speaker's voice. If `test_meta.tsv`
includes a `speaker` field then this will take precedence, and individual
utterances can be synthesized each using a different speaker's voice.

If your model checkpoint uses depthwise separable convolutional layers, then
also pass `--use-sepconv` to `inference.py`. Likewise, if trained with monotonic
alignment search then pass `--use-mas` to match checkpoint model architecture.

It should be possible to run any HiFi-GAN model checkpoint trained from the original
repo, given an appropriate config file. We also have a
[sample config](https://github.com/dan-wells/hifi-gan/blob/master/config/config_v1_16k_50Hz.json)
for 16 kHz audio with a 50 Hz frame shift, again to match audio preprocessing
when working with HuBERT models.

### Copy synthesis

To run copy synthesis through your chosen vocoder, load reference mel
spectrograms, pitch and duration values by adding corresponding fields in
`test_meta.tsv` pointing to `.pt` files on disk for each utterance, for example
as extracted using `prepare_dataset.py`. Paths can be relative if you also pass
e.g. `--dataset-path $DATA_ROOT` to `inference.py`.

A similar method can be used to generate time-aligned synthetic data for vocoder
fine-tuning, i.e. predicting ground-truth audio from errorful mel spectrograms
predicted by a FastPitch model. In that case, you should pass reference
durations and pitch contours during synthesis to limit variation from reference
audio. Use `--save-mels` to save predicted mel spectrograms to disk, either to
filepaths specified in the `{mel_,}output` field of `test_meta.tsv`, else to
sequential files like `$OUTPUT_DIR/mel_1.wav` if not specified.

### Audio transforms

There are several options for manipulating predicted audio, for example
adjusting the pace of output speech or transforming the predicted pitch
contours. See the original
[FastPitch v1.0 README](https://github.com/NVIDIA/DeepLearningExamples/tree/49e23b4597a6fca461321a085d8eb8abf1e215e2/PyTorch/SpeechSynthesis/FastPitch#inference-process)
for examples.
