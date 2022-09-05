from gensim.models.phrases import Phrases
from tqdm import tqdm
import json
import sys

de_connector_words = ['von', 'vom', 'den', 'des', 'der', 'die', 'das', 'zu', 'ein', 'eine', 'nach', 'ohne', 'in']

def sentences():
    with open('../data/processed/RZ_tokenized.jsonl', 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            sentence = json.loads(line)
            yield sentence['text']

if __name__ == '__main__':

    treshold = int(sys.argv[1]) # 20

    phrase_model20 = Phrases(sentences(), min_count=100, threshold=treshold, connector_words=de_connector_words)
    frozen_bigram_20 = phrase_model20.freeze()
    frozen_bigram_20.save('../data/models/bigram_models/bigrams_20.pkl')
