import sys
import pandas as pd
from climdist.data import load as load_df
import json
import gensim
from gensim.test.utils import datapath
from tqdm import tqdm
import spacy
import multiprocessing

lim = int(sys.argv[1])

print('Loading df')
df = load_df('main', readability=True, heading2=False)

with open('../data/processed/entities_ruled.json', 'r', encoding='utf8') as f:
    ents_ruled = json.load(f)


def spans_from_text(ix, df, ents, window, distance, remove_ents=False):
    
    text = df.loc[ix].full_text
    wea = ents[ix]['ents']['WEA']
    spans = []
    
    if len(wea) == 1:
        start = wea[0][1]-window if wea[0][1] > window else 0
        stop = wea[0][2]+window if wea[0][2]+window < len(text) else len(text)
        spans.append((start, stop))
    
    elif len(wea) > 1:
        inside = False
        for i, ent in enumerate(wea):
            #print(i, ent)
            #print(inside)
            
            if i == len(wea)-1:
                #print('1')
                if inside == False:
                    start = wea[i][1]-window
                stop = wea[i][2]+window if wea[i][2]+window < len(text) else len(text)
                spans.append((start, stop))
                break
                
            if inside == False:
                #print('2')
                start = wea[i][1]-window if wea[i][1] > window else 0
                inside = True
                if wea[i+1][1]-wea[i][2] > distance:
                    #print('2.1')
                    stop = wea[i][2]+window
                    inside = False
                    spans.append((start,stop))
                    
            else:
                #print('3')
                if wea[i+1][1]-wea[i][2] > distance:
                    #print('2.1')
                    stop = wea[i][2]+window
                    inside = False
                    spans.append((start,stop))
    
    
    if remove_ents == True:
        
        wea_ranges = []
        for ent in wea:
            wea_ranges += list(range(ent[1],ent[2]))
            
        results = []
        for span in spans:
            span_text = ''
            for char in range(span[0], span[1]):
                if char not in wea_ranges:
                    span_text += (text[char])
            span_text = span_text.replace('  ', ' ')
            results.append(span_text)
        
        return results
                    
    else:
        return [text[span[0]:span[1]] for span in spans]


def preprocess(doc, stopwords):
    doc = gensim.utils.simple_preprocess(doc, deacc=True)
    doc = [word for word in doc if word not in stopwords]
    return doc


def create_docs(df, ents, stopwords):
    
    docs = []
    
    for entry in tqdm(ents):
        ix = entry['id']
        spans = spans_from_text(ix, df, ents, 100, 200, remove_ents=True)
        if len(spans) > 0:
            for span in spans:
                doc = preprocess(span, stopwords)
                docs.append(doc)
                
    return docs


print('Preparing stopwords')
with open('../pipeline/stopwords.json', 'r', encoding='utf8') as f:
    corpus_stopwords = json.load(f)
    
blank_de = spacy.blank('de')
default_stopwords = list(blank_de.Defaults.stop_words)

ey_words = []
for word in default_stopwords:
    if 'ei' in word:
        ey_words.append(word.replace('ei', 'ey'))

stopwords = default_stopwords + corpus_stopwords + ey_words



print('Building corpus')
texts = create_docs(df, ents_ruled[:1000], stopwords)
dictionary = gensim.corpora.Dictionary(texts[:lim])
corpus = [dictionary.doc2bow(text) for text in texts[:lim]]

if __name__ == '__main__':

    print('Building model with 20 topics')
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=20, 
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            workers=83

                                            #workers=multiprocessing.cpu_count()
                                            )


    print('Saving model')
    lda_model.save('../data/models/test_lda/lda.model')


    print('Finished')

