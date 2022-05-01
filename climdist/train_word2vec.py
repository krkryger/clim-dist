import pandas as pd
from climdist.data import load
import sys
import os
import json
import gensim.models
import time
import multiprocessing
from tqdm import tqdm


class MyCorpus:
    """An iterator that yields sentences from the corpus"""
    def __iter__(self):
        for line in tqdm(open(_sentences, 'r', encoding='utf8')):
            sentence = json.loads(line)
            for key, value in sentence.items():
                if int(key) in indexes: # checks if the line is in the timerange
                    yield value


class BigramIterator(): 
    """Iterator for retrieving bigrammed sentences"""
    def __iter__(self):
        for sentence in MyCorpus():
            yield bigram_model[sentence]


if __name__ == '__main__':

    _savepath = str(sys.argv[1])
    _sentences = str(sys.argv[2])
    _vector_size = int(sys.argv[3])  # 100
    _min_count = int(sys.argv[4])    # 10
    _epochs = int(sys.argv[5])       # 10
    _timerange = range(int(sys.argv[6]), int(sys.argv[7]))
    _bigram_path = str(sys.argv[8])

    if not os.path.exists('/'.join(_savepath.split('/')[:-1])):
        sys.exit(f'Savepath {_savepath} does not exist!')

    df = load('main', readability=True, heading2=False)
    indexes = df[(df.readability==True) & (df.year.isin(_timerange))].index

    bigram_model = gensim.models.phrases.Phrases.load(_bigram_path)

    print('Training Word2Vec')
    start = time.perf_counter()

    model = gensim.models.Word2Vec(sentences=BigramIterator(),
                                    vector_size=_vector_size,
                                    min_count=_min_count,
                                    workers=multiprocessing.cpu_count(),
                                    epochs=_epochs)

    stop = time.perf_counter()
    print(f'Finished in {round(stop-start)} seconds')

    print('Saving model')
    model.wv.save_word2vec_format(_savepath)

    print('Done!')
