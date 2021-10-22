import json
import gensim.models
import time
import multiprocessing

class MyCorpus:
    """An iterator that yields sentences from the corpus"""

    def __iter__(self):
        for line in open('../pipeline/rz_sentences.jsonl', 'r', encoding='utf8'):
            yield json.loads(line)

print('Training Word2Vec')
start = time.perf_counter()

model = gensim.models.Word2Vec(vector_size=100,
                                min_count=10,
                                workers=multiprocessing.cpu_count)

stop = time.perf_counter()
print(f'Finished in {round(stop-start)} seconds')

print('Saving model')
model.save('../data/models/word2vec_221021/w2v_model')

print('Done!')
