import json
import time
from tqdm import tqdm
import pandas as pd

def sentences():
    with open('../data/processed/RZ_sentences.jsonl', 'r', encoding='utf8') as f:
        for line in f.readlines():
            yield json.loads(line)


def detect_entities(patterns, sentences):
    
    entities = []
    
    for line in tqdm(sentences):
        index = line['id']
        words = line['text']
        ents_in_line = []
        
        for ix, word in enumerate(words):
            if word in patterns:
                ents_in_line.append((word, ix))
                
        entities.append({'id': index, 'ents': ents_in_line})
        
    return entities


if __name__ == '__main__':

    start = time.perf_counter()
    ruler = pd.read_excel('../pipeline/ner/ruler_patterns.xlsx')

    print('Starting entity detection')
    entities = detect_entities(ruler['key'].values, sentences())

    date = time.strftime('%d%m%y')

    with open(f'../data/processed/all_keywords_{date}.json', 'w', encoding='utf8') as f:
        json.dump(entities, f)

    stop = time.perf_counter()
    print(f'Finished in {(round(round(stop-start)/60))} min {round((stop-start)%60)} s')