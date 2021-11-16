import pandas as pd
import spacy
from climdist import data as cd
import time
import concurrent.futures
import multiprocessing
from itertools import repeat
from tqdm import tqdm
import numpy as np
import json


nlp = spacy.load('../data/models/spacy_ner_151121/model-best/')
#print('Importing df')
df = cd.load('main', readability=True, heading2=False)
df['entities'] = pd.NA


def split_df(a, n):
    #np.random.shuffle(a)
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def find_entities(ix_range, df=df, nlp=nlp):
   
    results = []

    for ix in tqdm(ix_range, mininterval=3):
        text = df.loc[ix, 'full_text']
        date = df.loc[ix, 'date']
        doc = nlp(text)
        entities = [(ent.text, ent.start_char, ent.end_char, ent.label_)
                    for ent in doc.ents]
        results.append({'id': ix, 'date': date, 'ents': entities}) 
        
    return results


if __name__ == '__main__':
    
    print(f'Starting main')
    start = time.perf_counter()
    cpu_count = multiprocessing.cpu_count()

    final_list = []
    
    print('Processing')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        slices = list(split_df(df.index, cpu_count))
        results = executor.map(find_entities, slices, repeat(df), repeat(nlp))

    for lst in results:
        final_list += lst
    

    with open('../data/processed/entities.json', 'w', encoding='utf8') as f:
        json.dump(final_list, f)

    #final_col = df.entities.to_frame()
    #final_col.to_csv('../temp/test.csv', header=False)
        
    stop = time.perf_counter()

    print(f'Finished in {round(stop-start, 2)} seconds')




