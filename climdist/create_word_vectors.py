#print('Importing modules')

import pandas as pd
import spacy
import time
import concurrent.futures
import multiprocessing
import json
from tqdm import tqdm
import string

#print('Loading spacy model')
nlp = spacy.load('de_core_news_md')


def token_is_garbage(token, treshold=0.3):
    
    allowed_symbols = string.ascii_letters + '0123456789.äüö'
    
    non_alphabetical = 0
    for symbol in token.text:
        if symbol not in allowed_symbols:
            non_alphabetical += 1
            
    if non_alphabetical/len(token.text) >= treshold:
        return True
    else:
        return False


def create_sentences_for_vec(df):

    articles_as_lists = []

    for i in tqdm(df.index):
        text = df.loc[i, 'full_text']
        doc = nlp(text)
        word_list = []

        for token in doc:
            if not token.is_stop and token.pos_ not in ['PUNCT', 'SPACE', 'NUM']:
                if token_is_garbage(token) == False:
                    word_list.append(token.text)

        articles_as_lists.append(word_list)

    return articles_as_lists


def divide_df(df, n):
    
    part_len = round(len(df)/n)
    
    for i in range(n):
        yield df.iloc[i*part_len:(i+1)*part_len] 


if __name__ == '__main__':
    
    print(f'Starting main')
    start = time.perf_counter()
    cpu_count = multiprocessing.cpu_count()
    all_articles = []

    #print('Importing df')
    df = pd.read_parquet('C:/Users/krister/clim-dist/data/processed/RZ_sample.parquet')
    
    print('Processing')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        frames = list(divide_df(df, cpu_count))
        results = [executor.submit(create_sentences_for_vec, df) for df in frames]

        for f in concurrent.futures.as_completed(results):
            all_articles += f.result()
            
    with open('C:/Users/krister/clim-dist/pipeline/RZ_sentences_test.json', 'w', encoding='utf8') as f:
        json.dump(all_articles, f)
            

    stop = time.perf_counter()

    print(f'Finished in {round(stop-start, 2)} seconds')



    

