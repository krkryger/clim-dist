#print('Importing modules')

#import pandas as pd
import spacy
#import gensim
import json
from tqdm import tqdm
import string
from climdist.utils import load_df
import sys
import os
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


def tokenize(df, stopwords, min_len=4):
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

        line = {'id': ix, 'text': cleaned}

        articles_as_lists.append(line)
        
    return articles_as_lists


# def divide_df(df, n):
#     """Split the df for multiprocessing"""
#     part_len = round(len(df)/n) 
#     for i in range(n):
#         yield df.iloc[i*part_len:(i+1)*part_len] 


if __name__ == '__main__':
    
    print(f'Starting main')

    savepath = sys.argv[1]          # '../data/processed/RZ_sentences.jsonl'
    if len(sys.argv) == 3:
        bigram_path = sys.argv[2]   # '../data/models/bigram_models/bigrams_20.pkl'
        bigram_model = gensim.models.phrases.Phrases.load(bigram_path)
    else:
        bigram_path = None

    if not os.path.exists('/'.join(savepath.split('/')[:-1])):
        sys.exit(f'Savepath {savepath} does not exist!')

    start = time.perf_counter()

    #print('Importing df')
    df = load_df('main')
    df = df[df.readability==True]
        

    nlp = spacy.load('de_core_news_md')
    stopwords = nlp.Defaults.stop_words
    stopwords.update({'find', 'fich', 'von', 'vom', 'den', 'des', 'der', 'die', 'das', 'zu', 'ein',
                      'eine', 'einem', 'eines', 'nach', 'ohne', 'in'})
    
    string.punctuation += '„”»«™'

    all_articles = tokenize(df, stopwords)
            
    with open(savepath, 'w', encoding='utf8') as f:
        for text in all_articles:
            json_string = json.dumps(text)
            f.write(json_string)
            f.write('\n')
        
    stop = time.perf_counter()

    print(f'Finished in {round(stop-start, 2)} seconds')



    

