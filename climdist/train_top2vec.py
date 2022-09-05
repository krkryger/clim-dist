import sys
import json
from tqdm.notebook import tqdm
from top2vec import Top2Vec
import multiprocessing
from gensim.models.phrases import Phrases


# def get_sentences():
#     """Loads the tokenised articles"""
#     with open('../data/processed/RZ_sentences.jsonl', 'r', encoding='utf8') as f:
#         for ix, line in enumerate(f.readlines()):
#             yield json.loads(line)


class MyCorpus:
    """An iterator that yields sentences from the corpus"""
    def __iter__(self):
        for line in tqdm(open('../data/processed/RZ_tokenized.jsonl', 'r', encoding='utf8')):
            sentence = json.loads(line)
            yield sentence


class BigramIterator(): 
    """Iterator for retrieving bigrammed sentences"""
    def __iter__(self):
        for sentence in MyCorpus():
            yield bigram_model[sentence]


def build_spans(sentences, keywords, stretch, window, min_len):
    """Build concordance spans from tokens around the detected keywords.
    stretch (int): maximum distance (in words) of two keywords from each other before the span is broken into two
    window (int): nb of words included before first and after last keyword in span
    min_len (int): minimum nb of keywords that have to be in the text to start building spans"""

    print('Building spans for training')
    
    for entry, line in tqdm(zip(keywords, sentences)):
        
        spans = []
        line['text'] = bigram_model[line['text']]
        
        if len(entry['ents']) == 1 and min_len == 1:
            pos = entry['ents'][0][1]
            span_start = pos-window if pos >= window else 0
            span_end = pos+window if (len(line['text']) - pos) > window else len(line['text'])
            spans.append((span_start, span_end))
            
        elif len(entry['ents']) >= min_len:
            positions = [ent[1] for ent in entry['ents']]
            distances = [i-j for i, j in zip(positions[1:], positions[:-1])] + ['last']
            #print(positions, distances)
            
            inside = False
            span_len = 0
            for i, pos in enumerate(positions):
                #print(i)
                dist = distances[i] # distance to next position
                if inside == False:
                    #print(entry['ents'][i], 'outside')
                    span_start = positions[i]-window if positions[i] >= window else 0
                    span_len += 1
                    #print('start', span_start)
                    inside = True
                    if dist == 'last':
                        if span_len >= min_len:
                            span_end = positions[i]+window if (len(line['text']) - positions[i]) > window else len(line['text'])
                            #print('end', span_end)
                            spans.append((span_start, span_end))
                    elif dist > stretch:
                        if span_len >= min_len:
                            span_end = positions[i]+window
                            #print('end', span_end)
                            spans.append((span_start, span_end))
                            span_len = 0
                            inside = False 
                        else:
                            inside = False
                    else:
                        pass
                    
                elif inside == True:
                    span_len += 1
                    #print(entry['ents'][i], 'inside')
                    if dist == 'last':
                        if span_len >= min_len:
                            span_end = positions[i]+window if (len(line['text']) - positions[i]) > window else len(line['text'])
                            #print('end', span_end)
                            spans.append((span_start, span_end))
                    elif dist > stretch:
                        if span_len >= min_len:
                            span_end = positions[i]+window if (len(line['text']) - positions[i]) > window else len(line['text'])
                            #print(positions[i], len(line['text'])-positions[i])
                            #print('end', span_end)
                            spans.append((span_start, span_end))
                            inside = False
                        else:
                            span_len = 0
                            inside = False
                    else:
                        pass
                    
        else:
            pass
            

        if len(spans) > 0:
            line['spans'] = spans
            yield line                    
               
    #return [line for line in sentences if len(line['spans']) > 0]


def span_texts(entry):
    """Gets the tokens themselves from the span indices"""
    texts = []
    for span in entry['spans']:
        texts.append(entry['text'][span[0]:span[1]])
    return texts


def build_top2vec_corpus(spans):
    """Prepare top2vec corpus"""
    corpus = {}
    for entry in spans:
        for i, span in enumerate(span_texts(entry)):
            corpus[str(entry['id'])+'_'+str(i)] = ' '.join(span)
    return corpus


def train_top2vec(**kwargs):
    
    print('Building corpus')

    spans = []
    for line in build_spans(sentences, keywords, **kwargs):
        spans.append(line)

    t2v_corpus = build_top2vec_corpus(spans)
    print(f'corpus length: {len(t2v_corpus)}')

    print('Training...')
    t2v = Top2Vec(documents=list(t2v_corpus.values()),
              document_ids=list(t2v_corpus.keys()),
              min_count=20,
              speed='learn',
              workers=multiprocessing.cpu_count(),
              embedding_model_path='../data/models/word2vec_test/keyedvectors.txt')
    
    num_topics = t2v.get_num_topics()
    print(f'Found {num_topics} topics')
    
    return t2v


if __name__ == '__main__':

    span_min_len = int(sys.argv[1]) # 3
    save_path = str(sys.argv[2]) # .pkl
    bigram_path = str(sys.argv[3]) # '../data/models/bigram_models/bigrams_20.pkl'
    bigram_model = Phrases.load(bigram_path)

    print('Loading keyword occurrences')
    with open('../data/processed/keyword_occurrences.json', 'r', encoding='utf8') as f:
        keywords = json.load(f)
    
    print('Loading tokenized texts')
    sentences = MyCorpus()

    model = train_top2vec(stretch=100, window=20, min_len=span_min_len)
    model.save(save_path)






    