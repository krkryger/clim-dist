import sys

path = 'C:\\Users\\krister\\Climdist'
if path not in sys.path:
    sys.path.append(path)

import pandas as pd
import spacy
import json
from tqdm import tqdm
import concurrent.futures
import time

print('Importing spacy model')
nlp = spacy.load('de_core_news_md')

print('Importing df')
df = pd.read_parquet('../data/processed/RZ_processed.parquet')
df['readability'] = pd.read_parquet('../data/processed/OCR_readability_column.parquet')
df = df[df.readability==1]

stops = nlp.Defaults.stop_words


def create_sentences_for_vec(df, fpath='../pipeline/RZ_sentences.json'):

    corpus = []

    for i in tqdm(df.index):
        text = df.loc[i, 'full_text']
        doc = nlp(text)

        for sent in doc.sents:
            sent_list = []
            for word in sent:
                if word.text not in stops:
                    sent_list.append(word.text)

        corpus.append(sent_list)

    #with open(fpath, 'w') as f:
    #    json.dump(corpus, f)

    print('Finished')
    return corpus


if __name__ == '__main__':

    start = time.perf_counter()

    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    ranges = 4*[df.sample(100)]
    #    results = executor.map(create_sentences_for_vec, ranges)

    create_sentences_for_vec(df.sample(100))

    stop = time.perf_counter()

    print(f'Operation time: {stop-start} seconds')






    

