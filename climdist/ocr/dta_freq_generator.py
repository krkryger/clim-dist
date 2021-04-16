import operator


def strip_word(word):
    '''strip commas, spaces from end of the string.'''
    while len(word) > 1 and word[-1] in [' ', '.', ',', "'", '"', '-']:
        word = word[0:-1]
    if word in [' ', '.', ',', "'", '"', '-']:
        return ''
    else:
        return word


def get_word_frequencies(file):
    '''Input: .txt file; Output: dictionary'''

    print(f'Opening {file}')
    f = open(file, 'r', encoding='utf8')
    text = f.read()
    f.close()
    words = text.split()
    print(f'{len(words)} words in total.')

    terms_freq = {}

    print('Counting words...')
    for entry in words:
        word = strip_word(entry).lower()
        if terms_freq.get(word) == None:
            terms_freq[word] = 1
        else:
            terms_freq[word] += 1

    print(f'{len(terms_freq)} unique words')

    print('Frequency dictionary finished')
    print('Closing file')

    return terms_freq


def generate(input_dir, output_path):

    import glob
    input_path = input_dir + '/*.txt'
    tda_corpus_texts = glob.glob(input_path)
    tda_freq_dictionary = {}

    for file in tda_corpus_texts:
        freq_in_file = get_word_frequencies(file)

        for key in freq_in_file.keys():
            if key not in tda_freq_dictionary.keys():
                tda_freq_dictionary[key] = freq_in_file[key]
            else:
                tda_freq_dictionary[key] += freq_in_file[key]


    tda_freq_dictionary = dict(sorted(tda_freq_dictionary.items(), key=operator.itemgetter(1), reverse=True))

    with open(output_path, 'w', encoding='utf8') as f:
        keys = list(tda_freq_dictionary.keys())
        values = list(tda_freq_dictionary.values())

        for entry in range(0, len(keys)):
            line = keys[entry] + ' ' + str(values[entry]) + '\n'
            f.write(line)

    print('Word frequencies generated from Deutsches Textarchiv 19th century materials!')


if __name__ == "__main__":

    input = '../../data/external/dta_19c_corpus/'
    output = '../../pipeline/02_ocr/spell_dicts/dta_frequency_dict.txt'
    
    generate(input, output)
