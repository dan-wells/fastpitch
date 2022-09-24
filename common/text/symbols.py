""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text
that has been run through Unidecode. For other data, you can modify
_characters. See TRAINING_DATA.md for details.
'''

from panphon import FeatureTable
from panphon.permissive import PermissiveFeatureTable

from .cmudict import arpabet_symbols


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in arpabet_symbols]


def get_symbols(symbol_set='english_basic', symbol_type=None):
    if symbol_type == 'pf':
        ft = FeatureTable()
        _silence = ['sil']
        symbols = ft.names + _silence # 24 ~SPE features plus silence
    elif symbol_type == 'unit':
        _pad = ['_']
        # pass number of quantized units in symbol_set
        _units = list(range(int(symbol_set)))
        # embedding table indices will match 0-based unit IDs
        symbols = _units + _pad
    elif symbol_set == 'english_basic':
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_basic_lowercase':
        _pad = '_'
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_expanded':
        _pad = '_'
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = 'áçéêëñöøćž'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _punctuation + _math + _special + _accented + _letters) + _arpabet
    elif symbol_set == 'english_basic_sil':
        # this set is intended for use with character-level alignments,
        # where the 'pronunciation' of each word is represented by
        # space-separated characters in that word, e.g. 'cat' -> /c a t/.
        # Use with `--input-type phone`.
        _pad = '_'
        _silence = ['sil', 'sp']
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad) + _silence + list(_special + _punctuation + _letters)
    elif symbol_set == 'ipa':
        ft = PermissiveFeatureTable()
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        _phones = list(ft.bases.keys()) # 147 symbols, only base phones
        symbols = _pad + _silence + _oov + _phones
    elif symbol_set == 'ipa_all':
        ft = FeatureTable()
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        _phones = list(ft.seg_dict.keys()) # 6367 symbols (!) includes all diacritics
        symbols = _pad + _silence + _oov + _phones
    elif symbol_set == 'xsampa':
        ft = FeatureTable()
        xs = ft.xsampa
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        # TODO: some of these are only diacritics, doesn't make sense to have
        # entries in phone embedding table (and we shouldn't expect to see
        # them on their own in e.g. space-separated input files). To start,
        # let's assume X-SAMPA input only uses base symbols. In future, might
        # want to pass through IPA using xs.convert(), but then would need to
        # use the 'ipa_all' FeatureTable)
        _phones = list(xs.xs2ipa.keys()) # 169 symbols
        symbols = _pad + _silence + _oov + _phones
    elif symbol_set == 'combilex':
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        _phones = [
            '3', '5', '@', 'A', 'D', 'E', 'I', 'N', 'O', 'S', 'T', 'U', 'V', 'Z',
            'a', 'b', 'd', 'dZ', 'e', 'e~', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'l=',
            'm', 'm=', 'n', 'n=', 'o~', 'p', 'r', 's', 't', 'tS', 'u', 'v', 'w', 'z'
        ]
        symbols = _pad + _silence + _oov + _phones
    elif symbol_set == 'arpabet':
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        _phones = arpabet_symbols
        symbols = _pad + _silence + _oov + _phones
    elif symbol_set == 'globalphone':
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        _phones = [
            'AE', 'AX', 'A~', 'B', 'D', 'E', 'EU', 'E~', 'F', 'G', 'H', 'J',
            'K', 'L', 'M', 'N', 'NG', 'NJ', 'O', 'OE', 'OE~', 'P', 'R', 'S',
            'SH', 'T', 'V', 'W', 'Z', 'ZH', 'a', 'e', 'i', 'o', 'o~', 'u', 'y'
        ]
        symbols = _pad + _silence + _oov + _phones
    elif symbol_set == 'unisyn':
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        _phones = [
            '?', '@', 'a', 'aa', 'ae', 'aer', 'ai', 'ar', 'b', 'ch', 'd', 'dh',
            'e', 'eh', 'ei', 'eir', 'er', 'f', 'g', 'h', 'hw', 'i', 'i@', 'ii',
            'ir', 'iy', 'jh', 'k', 'l', 'l=', 'lw', 'm', 'm=', 'n', 'n=', 'ng',
            'o', 'oi', 'oo', 'or', 'ou', 'our', 'ow', 'owr', 'p', 'r', '@r',
            '@@r', 's', 'sh', 't', 't^', 'th', 'u', 'uh', 'ur', 'uu', 'uw',
            'v', 'w', 'x', 'y', 'z', 'zh',
        ]
        symbols = _pad + _silence + _oov + _phones
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))
    return symbols


def get_pad_idx(symbol_set='english_basic', symbol_type=None):
    if symbol_type == 'pf':
        return None  # no embedding table => no padding symbol (just zero vectors)
    try:
        return get_symbols(symbol_set, symbol_type).index('_')
    except ValueError:
        raise Exception("{} symbol set not used yet".format(symbol_set))
