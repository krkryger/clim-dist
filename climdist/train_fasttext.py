import sys
import json
import gensim.models
import time
import multiprocessing

_sentences = str(sys.argv[1])
_vector_size = int(sys.argv[2])
_min_count = int(sys.argv[3])
_epochs = int(sys.argv[4])

class MyCorpus:
    """An iterator that yields sentences from the corpus"""

    def __iter__(self):
        for line in open(_sentences, 'r', encoding='utf8'):
            yield json.loads(line)


start = time.perf_counter()

model = gensim.models.FastText(vector_size=_vector_size,
                                min_count=_min_count,
                                workers=multiprocessing.cpu_count())

print('Building vocab')
model.build_vocab(corpus_iterable=MyCorpus())
total_examples = model.corpus_count

print('Training FastText')
model.train(corpus_iterable=MyCorpus(),
            total_examples=total_examples,
            epochs=_epochs)

stop = time.perf_counter()
print(f'Finished in {round(stop-start)} seconds')

print('Saving model')
model.save('../data/models/fasttext_221021/ft_model')

print('Done!')
