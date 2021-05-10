import pandas as pd
import numpy as np
import spacy

from spellchecker import SpellChecker, WordFrequency
from symspellpy import SymSpell, Verbosity

### GENERAL UTILS

def strip_word(word, extra_symbols):
    '''strip commas, spaces from end of the string.'''
    
    unwanted_symbols = ' .,;"-/\()[]{}|«»<>_'
    unwanted_symbols += extra_symbols
    
    symbolsstring = ''.join(unwanted_symbols)
    
    word = word.strip(symbolsstring)
    
    return word


### PYSPELL UTILS

def is_word(string, speller):
    '''define if a string is a word according to pyspellchecker'''
    return (speller.unknown([string]) == set())


def get_unknown_words(text, speller):
    '''Input:string. Return: list of words not recognized by pyspellchecker'''
    
    newline = '\n'
    textlist = [word for word in text.split(' ')]
    
    for word in textlist:
        word = strip_word(word)
            
    wordcount = len(textlist)
    unknown_words = []
    hyphenated_words = []
    
    for word in textlist:
        if is_word(word, speller) == False:
            if newline in word:
                hyphenated_words.append(word)
            else:
                unknown_words.append(word)
                
    for word in hyphenated_words:
        if is_word(word.replace(newline, ''), speller) == False:
            firstword = word.split(newline)[0]
            secondword = word.split(newline)[1]
            appended_first = False
            
            if is_word(firstword, speller) == False:
                unknown_words.append(firstword)
                wordcount += 1
                appended_first = True
            if is_word(secondword, speller) == False:
                unknown_words.append(secondword)
                if not appended_first:
                    wordcount += 1

    print(f'The text has about {wordcount} words')
    print(f'{len(unknown_words)} words are not recognized by pyspellchecker')
    return unknown_words


def ocr_cleanliness_score(text):
    '''score from 0 to 1 to evaluate the relative readability of the text (on a word level)'''
    return len(get_unknown_words(text))/len(text.split(' '))


def generate_json_dict_for_pyspell(input_path, min_treshold):

    import json

    print('Opening dictionary')

    with open(input_path, encoding='utf8') as infile:
        lines = infile.readlines()

        freq_dict = {}

        freq = int(lines[0].split()[1])

        for line in lines:
            try:
                word = line.split()[0]
                word = strip_word(word)
                freq = int(line.split()[1])
            except IndexError:
                print(f'{line}, IndexError!')
                pass
            
            if freq > min_treshold:
                freq_dict[word] = freq

    print('Creating JSON...')

    output_path = input_path[:-4] + '.json'
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(freq_dict, f, ensure_ascii=False)

    print('Finished!')


### SYMSPELL UTILS

from climdist.ocr import dta_freq_generator


def spelling_correction(text, speller, edit_dist=2, ignorenonwords=True, transfercasing=True):
    
    correction = speller.lookup_compound(phrase=text,
                                         max_edit_distance=edit_dist,
                                         ignore_non_words=ignorenonwords,
                                         transfer_casing=transfercasing)
    
    return correction[0]._term


def compare_spellings(text, spellers, nlp_model=None, transfercasing=True, lower=False):
    
    if lower:
        text = text.lower()
        text = text.replace('\n', ' ')
    else:
        text = text.replace('\n', ' ')
    
    displacy_color_code = {'WEA': '#4cafd9',
                  'PER': '#ffb366',
                  'DAT': '#bf80ff',
                  'LOC': '#a88676',
                  'MISC': 'grey',
                  'MEA': '#85e085',
                  'ORG': '#5353c6'}

    displacy_options = {'ents': ['WEA', 'PER', 'DAT', 'LOC', 'MISC', 'MEA', 'ORG'], 'colors': displacy_color_code}
    
    if nlp_model == None:
        
        print('ORIGINAL')
        print(text)
        print('\n')
        print('\n')
        
        for speller in spellers:
            print(spelling_correction(text, speller, transfercasing=transfercasing))
            print('\n')
            print('\n')
        
    else:
        
        print('ORIGINAL')
        doc_original = nlp_model(text)
        spacy.displacy.render(doc_original, style='ent', jupyter=True, options=displacy_options)
        print('\n')
        print('\n')
        
        for speller in spellers:
            doc = nlp_model(spelling_correction(text, speller, transfercasing=transfercasing))
            spacy.displacy.render(doc, style='ent', jupyter=True, options=displacy_options)
            print('\n')
            print('\n')


### EXAMPLE SYMSPELL SPELLERS

if __name__ == "__main__":

    dictionaries_path = '../../pipeline/02_ocr/spell_dicts/'

    sym_default = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_default.load_dictionary((dictionaries_path + 'symspell_default_dict_de.txt'), 0, 1, encoding='utf8')

    sym_dta = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_dta.load_dictionary((dictionaries_path + 'symspell_default_dict_de.txt'), 0, 1, encoding='utf8')
    sym_dta.load_dictionary((dictionaries_path + 'dta_frequency_dict.txt'), 0, 1, encoding='utf8')

    sym_dta_only = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_dta_only.load_dictionary((dictionaries_path + 'dta_frequency_dict.txt'), 0, 1, encoding='utf8')


