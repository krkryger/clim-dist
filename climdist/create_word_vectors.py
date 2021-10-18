print('Importing modules')

import pandas as pd
import spacy
import time
import concurrent.futures
import multiprocessing
import json
from tqdm import tqdm

print('Loading spacy model')
nlp = spacy.load('de_core_news_md')


def create_sentences_for_vec(df):

    corpus = []

    for i in tqdm(df.index):
        text = df.loc[i, 'full_text']
        doc = nlp(text)

        for sent in doc.sents:
            sent_list = []
            for word in sent:
                if not word.is_stop:
                    sent_list.append(word.text)

        corpus.append(sent_list)

    return corpus


if __name__ == '__main__':
    
    print(f'Starting main')
    start = time.perf_counter()
    
    cpu_count = multiprocessing.cpu_count()
    
    all_sentences = []

    print('Importing df')
    df = pd.read_parquet('C:/Users/krister/Climdist/data/processed/RZ_sample.parquet')
    
    print('Processing')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        frames = cpu_count*[df.sample(100)]
        results = [executor.submit(create_sentences_for_vec, df) for df in frames]

        for f in concurrent.futures.as_completed(results):
            print(len(f.result()))
            print(type(f.result()))
            all_sentences += f.result()
            
    with open('C:/Users/krister/Climdist/temp/sentences_test.json', 'w', encoding='utf8') as f:
        json.dump(all_sentences, f)
            

    stop = time.perf_counter()

    print(f'Finished in {round(stop-start, 2)} seconds')



    

