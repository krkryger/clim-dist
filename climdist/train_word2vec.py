import sys
import json
import gensim.models
import time
import multiprocessing

_sentences = str(sys.argv[1])
_vector_size = sys.argv[2]
_min_count = sys.argv[3]
_epochs = sys.argv[4]

class MyCorpus:
    """An iterator that yields sentences from the corpus"""

    def __iter__(self):
        for line in open(_sentences, 'r', encoding='utf8'):
            yield json.loads(line)

print('Training Word2Vec')
start = time.perf_counter()

model = gensim.models.Word2Vec(sentences=MyCorpus(),
                                vector_size=_vector_size,
                                min_count=_min_count,
                                workers=multiprocessing.cpu_count(),
                                epochs=_epochs)

stop = time.perf_counter()
print(f'Finished in {round(stop-start)} seconds')

print('Saving model')
model.save('../data/models/word2vec_221021/w2v_model')

print('Done!')
