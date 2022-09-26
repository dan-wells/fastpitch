""" adapted from https://github.com/keithito/tacotron """

import re

import panphon

from . import cleaners
from .symbols import get_symbols


# Mapping from Combilex phones to IPA
_combilex_to_ipa = {
    '3': 'ɜ', '5': 'ɫ', '@': 'ə', 'A': 'ɑ', 'D': 'ð', 'E': 'ɛ', 'I': 'ɪ',
    'N': 'ŋ', 'O': 'ɔ', 'S': 'ʃ', 'T': 'θ', 'U': 'ʊ', 'V': 'ʌ', 'Z': 'ʒ',
    'a': 'a', 'b': 'b', 'd': 'd', 'dZ': 'd͡ʒ', 'e': 'e', 'e~': 'ẽ', 'f': 'f',
    'g': 'ɡ', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'l=': 'l̩',
    'm': 'm', 'm=': 'm̩', 'n': 'n', 'n=': 'n̩', 'o': 'o', 'o~': 'õ', 'p': 'p',
    'r': 'ɹ', 's': 's', 't': 't', 'tS': 't͡ʃ', 'u': 'u', 'v': 'v', 'w': 'w',
    'z': 'z'
}


class PhoneProcessing(object):
    def __init__(self, symbol_set, symbol_type='phone'):
        self.symbol_set = symbol_set
        self.symbol_type = symbol_type
        self.symbols = get_symbols(symbol_set, symbol_type)

        if symbol_type == 'phone':
            self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
            self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        # Used if symbol_set == 'ipa' or to convert to phonological features
        self.ft = panphon.FeatureTable()
        self.sil_symbols = ['sil', 'sp', 'spn']
        self.sil_pf_vec = [1 if i == 'sil' else 0 for i in self.symbols]

    def phones_to_ids(self, text):
        # TODO: could also handle X-SAMPA this way if mapping through IPA
        # using panphon
        if self.symbol_set.startswith('ipa'):
            # Text can be either space-delimited phone strings or phonetized
            # words, e.g. 'sp ðə kæt sp'
            symbol_ids = []
            for word in text.split(' '):
                if word in self.sil_symbols:
                    symbol_ids.append(self.symbol_to_id[word])
                else:
                    for s in self.ft.ipa_segs(word):
                        symbol_ids.append(self.symbol_to_id[s])
            return symbol_ids
        else:
            # Assuming space-delimited phone strings, e.g. 'sp D @ k { t sp'
            return [self.symbol_to_id[s] for s in text.split(' ')]

    def phones_to_pfs(self, text):
        feats = []
        for s in text.split(' '):
            if s in self.sil_symbols:
                feats.append(self.sil_pf_vec)
            elif self.symbol_set == 'combilex':
                pf_vec = self.ft.fts(_combilex_to_ipa[s]).numeric()
                pf_vec += [0] # add unspecified silence feature
                feats.append(pf_vec)
            else:
                pf_vecs = self.ft.word_to_vector_list(
                    s, numeric=True, xsampa=self.symbol_set=='xsampa')
                pf_vecs = [i + [0] for i in pf_vecs]
                feats.extend(pf_vecs)
        return feats

    def ids_to_text(self, ids):
        return ' '.join(self.id_to_symbol[i] for i in ids)

    def encode_text(self, text):
        if self.symbol_type == 'pf':
            text_encoded = self.phones_to_pfs(text)
        else:
            text_encoded = self.phones_to_ids(text)
        return text_encoded


class UnitProcessing(object):
    def __init__(self, symbol_set, symbol_type):
        self.symbols = get_symbols(symbol_set, symbol_type)

    def ids_to_text(self, ids):
        return ' '.join(str(i) for i in ids)

    def encode_text(self, text):
        # embedding table indices should match 0-based unit IDs
        return [int(i) for i in text.split(' ')]


# TODO: work out best defaults and neatest interface to set handle_sil
# and add_spaces here
class TextProcessing(object):
    def __init__(self, symbol_set, cleaner_names, add_spaces=False, handle_sil=False):
        self.symbols = get_symbols(symbol_set)
        self.cleaner_names = cleaner_names
        self.add_spaces = add_spaces
        self.handle_sil = handle_sil
        self.sil_symbols = ['sil', 'sp', 'spn']

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_ids(self, text):
        # skip any unknown symbols -- more likely to be e.g. unanticipated
        # punctuation for text inputs compared to pre-defined phone sets
        # NOTE: this will not play nicely with durations from forced alignments
        # which do _not_ ignore unknown symbols, but should be fine for MAS
        return [self.symbol_to_id[i] for i in text if i in self.symbol_to_id]

    def ids_to_text(self, ids):
        symbols = [self.id_to_symbol[i] for i in ids]
        text = []
        for s1, s2 in zip(symbols, symbols[1:]):
            text.append(s1)
            if s1 in self.sil_symbols or s2 in self.sil_symbols:
                text.append(' ')
        text.append(s2)
        return ''.join(text)

    def split_with_sil(self, text):
        symbols = []
        text = text.split()
        for t1, t2 in zip(text, text[1:]):
            self._add_to_sym_list(symbols, t1, t2)
        self._add_to_sym_list(symbols, t2)
        return symbols

    def _add_to_sym_list(self, symbols, s, s_next=None):
        if s in self.sil_symbols:
            symbols.append(s)
        else:
            symbols.extend(s)  # split chars from word
            if s_next not in self.sil_symbols and s_next is not None:
                symbols.append(' ')  # restore spaces between words

    def clean_text(self, text):
        for name in self.cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            text = cleaner(text)
        return text

    def encode_text(self, text):
        text = self.clean_text(text)
        # handle silence phones from forced alignments as single tokens,
        # while splitting other words into character sequences
        # TODO: check if this actually lines up with forced alignment
        # durations -- might need to skip spaces here
        if self.handle_sil:
            text = self.split_with_sil(text)
        # add leading and trailing space to text transcript, e.g. to
        # represent silences if not referencing forced alignments
        if self.add_spaces:
            text = ' ' + text + ' '
        text = cleaners.collapse_whitespace(text)  # in case we add extra
        text_encoded = self.text_to_ids(text)
        return text_encoded
