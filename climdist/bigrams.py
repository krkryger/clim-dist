from gensim.models.phrases import Phrases
de_connector_words = ['von', 'vom', 'den', 'des', 'der', 'die', 'das', 'zu', 'ein', 'eine', 'nach', 'ohne', 'in']
def sentences():
    with open('../data/processed/RZ_sentences.jsonl', 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            sentence = json.loads(line)
            for key, value in sentence.items():
                yield value
phrase_model20 = Phrases(sentences(), min_count=100, threshold=30, connector_words=de_connector_words)
frozen_bigram_20 = phrase_model20.freeze()
frozen_bigram_20.save('../data/models/bigram_models/bigrams_20.pkl')
