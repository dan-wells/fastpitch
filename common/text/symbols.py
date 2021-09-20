""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from .cmudict import valid_symbols


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in valid_symbols]


def get_symbols(symbol_set='english_basic'):
    if symbol_set == 'english_basic':
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
    elif symbol_set == 'combilex':
        _pad = ['_']
        _silence = ['sil', 'sp']
        _oov = ['spn']
        _phones = [
            '3', '5', '@', 'A', 'D', 'E', 'I', 'N', 'O', 'S', 'T', 'U', 'V', 'Z',
            'a', 'b', 'd', 'dZ', 'e', 'e~', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'l=',
            'm', 'm=', 'n', 'n=', 'o~', 'p', 'r', 's', 't', 'tS', 'u', 'v', 'w', 'z'
        ]
        symbols = list(_pad + _silence + _oov + _phones)
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))

    return symbols


def get_pad_idx(symbol_set='english_basic'):
    try:
        return get_symbols(symbol_set).index('_')
    except ValueError:
        raise Exception("{} symbol set not used yet".format(symbol_set))
