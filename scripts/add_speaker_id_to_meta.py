#!/usr/bin/env python

import argparse
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add integer speaker IDs to utterance metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('meta_in', type=str,
        help="Input metadata file")
    parser.add_argument('meta_out', type=str,
        help="Output metadata file with added speaker information")
    parser.add_argument('--spkr-list', type=str, default='speaker_ids.txt',
        help="Output file listing speakers with integer IDs")
    parser.add_argument('--spkr-sep', type=str, default='_',
        help="Character separating speaker from rest of utterance ID")
    parser.add_argument('--n-spkrs', type=int, default=0,
        help="Expected number of speakers in output, used for validation (optional)")
    parser.add_argument('--shuf', action='store_true',
        help="Shuffle lines in output metadata")
    parser.add_argument('--seed', type=int, default=1337,
        help="Random seed for shuffling data")
    args = parser.parse_args()
    return args


def load_meta(meta_in, spkr_sep='_'):
    metadata = {}
    speakers = set()
    with open(meta_in) as inf:
        for line in inf:
            line = line.strip()
            mels, *_ = line.split('|')
            utt_id = os.path.splitext(os.path.basename(mels))[0]
            metadata[utt_id] = line
            spkr = utt_id.split(spkr_sep)[0]
            speakers.add(spkr)
    return metadata, speakers


def write_meta(meta_out, metadata, utts, speaker_ids, spkr_sep='_'):
    with open(meta_out, 'w') as outf:
        for utt in utts:
            spkr = utt.split(spkr_sep)[0]
            outf.write("{}|{}\n".format(metadata[utt], speaker_ids[spkr]))


def assign_speaker_ids(speakers, n_spkrs=0):
    if n_spkrs:
        assert len(speakers) == n_spkrs, \
            "Expected {} speakers, found {}".format(n_spkrs, len(speakers))
    else:
        print("Found {} speakers".format(len(speakers)))
    speaker_ids = {}
    for i, spkr in enumerate(sorted(speakers)):
        speaker_ids[spkr] = i
    if n_spkrs:
        assert max(speaker_ids.values()) == n_spkrs - 1
    return speaker_ids


def load_speaker_list(speakers_in, n_spkrs=0):
    speaker_ids = {}
    with open(speakers_in) as inf:
        for line in inf:
            spkr, spkr_id = line.strip().split()
            speaker_ids[spkr] = spkr_id
    if n_spkrs:
        assert len(speaker_ids) == n_spkrs, \
            "Expected {} speakers, found {}".format(n_spkrs, len(speaker_ids))
    else:
        print("Found {} speakers".format(len(speaker_ids)))
    return speaker_ids


def write_speaker_list(speakers_out, speaker_ids):
    with open(speakers_out, 'w') as outf:
        for spkr, spkr_id in speaker_ids.items():
            outf.write("{} {}\n".format(spkr, spkr_id))


if __name__ == '__main__':
    args = parse_args()

    metadata, speakers = load_meta(args.meta_in, args.spkr_sep)
    if os.path.isfile(args.spkr_list):
        # TODO: account for the case where spkr_list represents a subset
        # of speakers in meta_in, i.e. we maintain those we see again and
        # add new IDs for previously unseen speakers
        speaker_ids = load_speaker_list(args.spkr_list, args.n_spkrs)
    else:
        speaker_ids = assign_speaker_ids(speakers, args.n_spkrs)
        write_speaker_list(args.spkr_list, speaker_ids)

    utts = list(metadata.keys())
    if args.shuf:
        random.seed(args.seed)
        random.shuffle(utts)

    write_meta(args.meta_out, metadata, utts, speaker_ids, args.spkr_sep)
