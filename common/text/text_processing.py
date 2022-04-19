""" adapted from https://github.com/keithito/tacotron """
import re
import numpy as np
import panphon
from . import cleaners
from . import cmudict
from .numerical import _currency_re, _expand_currency
from .symbols import get_symbols


# Regular expression matching text enclosed in curly braces for encoding
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# Regular expression matching words and not words
_words_re = re.compile(r"([a-zA-ZÀ-ž]+['][a-zA-ZÀ-ž]{1,2}|[a-zA-ZÀ-ž]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž{}]+)")

# Regular expression separating words enclosed in curly braces for cleaning
_arpa_re = re.compile(r'{[^}]+}|\S+')


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
        self.sil_symbols = ['sp', 'spn', 'sil']
        self.sil_pf_vec = [1 if i == 'sil' else 0 for i in self.symbols]

    def encode_text(self, text):
        if self.symbol_type == 'pf':
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
        else:
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


# TODO: Handle simple text input with spaces between words, but expanding
# to individual characters to match input lengths and durations with those
# extracted from character-level TextGrid alignments, i.e.:
class TextProcessing(object):
    def __init__(self, symbol_set, cleaner_names, p_arpabet=0.0,
                 handle_arpabet='word', handle_arpabet_ambiguous='ignore',
                 expand_currency=True):
        self.symbols = get_symbols(symbol_set)
        self.cleaner_names = cleaner_names

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}
        self.expand_currency = expand_currency

        # cmudict
        self.p_arpabet = p_arpabet
        self.handle_arpabet = handle_arpabet
        self.handle_arpabet_ambiguous = handle_arpabet_ambiguous

    def text_to_sequence(self, text):
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self.symbols_to_sequence(text)
                break
            sequence += self.symbols_to_sequence(m.group(1))
            sequence += self.arpabet_to_sequence(m.group(2))
            text = m.group(3)

        return sequence

    def sequence_to_text(self, sequence):
        result = ''
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                result += s
        return result.replace('}{', ' ')

    def clean_text(self, text):
        for name in self.cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            text = cleaner(text)

        return text

    def symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if s in self.symbol_to_id]

    def arpabet_to_sequence(self, text):
        return self.symbols_to_sequence(['@' + s for s in text.split()])

    def get_arpabet(self, word):
        arpabet_suffix = ''

        if word.lower() in cmudict.heteronyms:
            return word

        if len(word) > 2 and word.endswith("'s"):
            arpabet = cmudict.lookup(word)
            if arpabet is None:
                arpabet = self.get_arpabet(word[:-2])
                arpabet_suffix = ' Z'
        elif len(word) > 1 and word.endswith("s"):
            arpabet = cmudict.lookup(word)
            if arpabet is None:
                arpabet = self.get_arpabet(word[:-1])
                arpabet_suffix = ' Z'
        else:
            arpabet = cmudict.lookup(word)

        if arpabet is None:
            return word
        elif arpabet[0] == '{':
            arpabet = [arpabet[1:-1]]

        # XXX arpabet might not be a list here
        if type(arpabet) is not list:
            return word

        if len(arpabet) > 1:
            if self.handle_arpabet_ambiguous == 'first':
                arpabet = arpabet[0]
            elif self.handle_arpabet_ambiguous == 'random':
                arpabet = np.random.choice(arpabet)
            elif self.handle_arpabet_ambiguous == 'ignore':
                return word
        else:
            arpabet = arpabet[0]

        arpabet = "{" + arpabet + arpabet_suffix + "}"

        return arpabet

    def encode_text(self, text, return_all=False):
        if self.expand_currency:
            text = re.sub(_currency_re, _expand_currency, text)
        text_clean = [self.clean_text(split) if split[0] != '{' else split
                      for split in _arpa_re.findall(text)]
        text_clean = ' '.join(text_clean)
        text_clean = cleaners.collapse_whitespace(text_clean)
        text = text_clean

        text_arpabet = ''
        if self.p_arpabet > 0:
            if self.handle_arpabet == 'sentence':
                if np.random.uniform() < self.p_arpabet:
                    words = _words_re.findall(text)
                    text_arpabet = [
                        self.get_arpabet(word[0])
                        if (word[0] != '') else word[1]
                        for word in words]
                    text_arpabet = ''.join(text_arpabet)
                    text = text_arpabet
            elif self.handle_arpabet == 'word':
                words = _words_re.findall(text)
                text_arpabet = [
                    word[1] if word[0] == '' else (
                        self.get_arpabet(word[0])
                        if np.random.uniform() < self.p_arpabet
                        else word[0])
                    for word in words]
                text_arpabet = ''.join(text_arpabet)
                text = text_arpabet
            elif self.handle_arpabet != '':
                raise Exception("{} handle_arpabet is not supported".format(
                    self.handle_arpabet))

        text_encoded = self.text_to_sequence(text)

        if return_all:
            return text_encoded, text_clean, text_arpabet

        return text_encoded

class UnitProcessing(object):
    def __init__(self, symbol_set, symbol_type):
        self.symbols = get_symbols(symbol_set, symbol_type)

    def encode_text(self, text):
        # embedding table indices should match 0-based unit IDs
        return [int(i) for i in text.split(' ')]
