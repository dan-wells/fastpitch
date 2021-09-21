#!/usr/bin/env python

import argparse
import panphon


# Will use XS.xs2ipa dict to map between X-SAMPA and IPA. This is not
# as smart as XS.convert(), so don't pass anything with X-SAMPA
# diacritics for now.
FT = panphon.FeatureTable()
XS = FT.xsampa

COMBILEX_TO_IPA = {
    '3': 'ɜ', '5': 'ɫ', '@': 'ə', 'A': 'ɑ', 'D': 'ð', 'E': 'ɛ', 'I': 'ɪ',
    'N': 'ŋ', 'O': 'ɔ', 'S': 'ʃ', 'T': 'θ', 'U': 'ʊ', 'V': 'ʌ', 'Z': 'ʒ',
    'a': 'a', 'b': 'b', 'd': 'd', 'dZ': 'd͡ʒ', 'e': 'e', 'e~': 'ẽ', 'f': 'f',
    'g': 'ɡ', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'l=': 'l̩',
    'm': 'm', 'm=': 'm̩', 'n': 'n', 'n=': 'n̩', 'o': 'o', 'o~': 'õ', 'p': 'p',
    'r': 'ɹ', 's': 's', 't': 't', 'tS': 't͡ʃ', 'u': 'u', 'v': 'v', 'w': 'w',
    'z': 'z'
}

COMBILEX_TO_XSAMPA = {
    '3': '3', '5': '5', '@': '@', 'A': 'A', 'D': 'D', 'E': 'E', 'I': 'I',
    'N': 'N', 'O': 'O', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'Z': 'Z',
    'a': 'a', 'b': 'b', 'd': 'd', 'dZ': 'dZ', 'e': 'e', 'e~': 'e~', 'f': 'f',
    'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'l=': 'l=',
    'm': 'm', 'm=': 'm=', 'n': 'n', 'n=': 'n=', 'o': 'o', 'o~': 'o~', 'p': 'p',
    'r': 'r\\', 's': 's', 't': 't', 'tS': 'tS', 'u': 'u', 'v': 'v', 'w': 'w',
    'z': 'z'
}

PHONE_MAPS = {
    'combilex': {
        'ipa': COMBILEX_TO_IPA,
        'xsampa': COMBILEX_TO_XSAMPA,
    },
    'xsampa': {
        'ipa': XS.xs2ipa,
        'combilex': {v: k for k, v in COMBILEX_TO_XSAMPA.items()},
    },
    'ipa': {
        'combilex': {v: k for k, v in COMBILEX_TO_IPA.items()},
        'xsampa': {v: k for k, v in XS.xs2ipa.items()},
    },
}

def convert_phones(text, src, tgt, diacritics=False):
    phone_map = PHONE_MAPS[src][tgt]
    converted_phones = []
    # assuming space-separated phone strings
    for phone in text.split(' '):
        if not diacritics:
            if src == 'combilex':
                phone = phone.strip('~=')
        if phone not in ['sp', 'spn', 'sil']:
            phone = phone_map[phone]
        converted_phones.append(phone)
    return ' '.join(converted_phones)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-phones', choices=['combilex', 'xsampa', 'ipa'], required=True)
    parser.add_argument('--tgt-phones', choices=['combilex', 'xsampa', 'ipa'], required=True)
    parser.add_argument('--src-meta', type=str, required=True)
    parser.add_argument('--tgt-meta', type=str, required=True)
    parser.add_argument('--diacritics', action='store_true')
    args = parser.parse_args()

    assert args.src_phones != args.tgt_phones, "Source and target phone set must be different."
    assert args.src_meta != args.tgt_meta, "Source and target metadata files must be different."

    if args.diacritics:
        raise NotImplementedError("Can't handle diacritics yet!")

    with open(args.src_meta) as inf, open(args.tgt_meta, 'w') as outf:
        for line in inf:
            converted_line = convert_phones(line.strip('\n'), args.src_phones, args.tgt_phones)
            outf.write('{}\n'.format(converted_line))
