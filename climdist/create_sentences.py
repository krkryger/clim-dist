#print('Importing modules')

from turtle import pu
import pandas as pd
import spacy
import json
from tqdm import tqdm
import string
from climdist.data import load
import sys
import time

def is_garbage(token, treshold=0.3):
    """Test if a token "is garbage", i.e. if it contains too many weird symbols"""
    
    allowed_symbols = string.ascii_letters + 'äüöß'

    if len(token) == 0:
        return True
    
    non_alphabetical = 0
    for symbol in token:
        if symbol not in allowed_symbols:
            non_alphabetical += 1
            
    if non_alphabetical/len(token) >= treshold:
        return True
    else:
        return False


def create_sentences_for_vec(df, stopwords, min_len=4):
    """Parse the main dataframe into lists if lowercase tokens,
    leaving out stopwords, garbage tokens and words with len < 4 (default value)"""

    articles_as_lists = []
    
    for ix, row in tqdm(df.iterrows()):
        tokens = row.full_text.split()
        cleaned = []

        for token in tokens:
            wordform = token.lower().strip(string.punctuation).lstrip(string.punctuation)
            if not is_garbage(wordform) and len(wordform) >= min_len and wordform not in stopwords:
                cleaned.append(wordform)

        line = {ix: cleaned}
        articles_as_lists.append(line)
        
    return articles_as_lists


# def divide_df(df, n):
#     """Split the df for multiprocessing"""
#     part_len = round(len(df)/n) 
#     for i in range(n):
#         yield df.iloc[i*part_len:(i+1)*part_len] 


if __name__ == '__main__':
    
    print(f'Starting main')

    savepath = sys.argv[1]  # must be .jsonl format
    timerange = range(int(sys.argv[2]), int(sys.argv[3])) # for diachronic embeddings

    start = time.perf_counter()

    #print('Importing df')
    df = load('main', heading2=False, readability=True)
    df = df[(df.readability==True) & (df.year.isin(timerange))]
    nlp = spacy.load('de_core_news_md')
    stopwords = nlp.Defaults.stop_words
    
    string.punctuation += '„”»«™'
    all_articles = create_sentences_for_vec(df, stopwords)
            
    with open(savepath, 'w', encoding='utf8') as f:
        for text in all_articles:
            json_string = json.dumps(text)
            f.write(json_string)
            f.write('\n')
        
    stop = time.perf_counter()

    print(f'Finished in {round(stop-start, 2)} seconds')



    

