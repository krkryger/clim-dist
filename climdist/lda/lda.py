print('Importing modules')
import numpy as np
import pandas as pd
import json
import glob
from tqdm import tqdm
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
import spacy
from nltk.corpus import stopwords
import pyLDAvis.gensim_models
import warnings
warnings.filterwarnings('ignore')
print('Modules imported')


def lemmatize(texts, allowed_postags=['NOUN']):
    
    texts_out = []
    
    for text in tqdm(texts):
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags and token.text.lower() not in stops and token.lemma_ not in stops:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    
    return texts_out


def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True) #separates words, applies lowercase, removes accents
        final.append(new)
    return final


def get_optimal_n_topics(texts, no_above, start, end, step, alpha='auto', plot=True):

    if plot == True:
            from matplotlib import pyplot as plt

    id2word = corpora.Dictionary(texts)
    id2word.filter_extremes(no_above=no_above)
    corpus = [id2word.doc2bow(text) for text in texts]
    tfidf = TfidfModel(corpus, id2word=id2word)
    
    n_topics = []
    coherence_scores = []
    
    for n in range(start, end+1, step):
        print(f'Computing model with {n} topics')
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=n,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha=alpha)
        
        coherence_model = gensim.models.CoherenceModel(model=model, texts=data_bigrams_trigrams, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print('Coherence Score: ', coherence_score)
        
        n_topics.append(n)
        coherence_scores.append(coherence_score)

    if plot == True:    
        plt.plot(range(start, end+1, step), coherence_scores)
        plt.show()
    
    print(f'Optimal number of topics: {n_topics[coherence_scores.index(max(coherence_scores))]}')
    print('\n')
    print('\n')
        
    return {'alpha': alpha, 'treshold': no_above, 'scores': [(i,j) for i,j in zip(n_topics, coherence_scores)]}


def get_optimal_alpha_and_treshold(texts, treshold_range, alpha_range, output_path, start, end, step, plot=True):
    
    results = []

    for treshold in treshold_range:
        for alpha in alpha_range:
            print(f'\nTreshold {treshold}, alpha {alpha}\n')
            results.append(get_optimal_n_topics(texts,
                                                no_above=treshold,
                                                start=start,
                                                end=end,
                                                step=step,
                                                alpha=alpha,
                                                plot=plot))
            
    import json
    import codecs
            
    with codecs.open(output_path, 'w', encoding='utf8') as f:
        json_string = json.dumps(results)
        f.write(json_string)


if __name__=='__main__':

    print('Loading dataframe from LNB_processed.xlsx')
    df = pd.read_excel('../../data/processed/LNB_processed.xlsx')

    print('Loading spacy_model_250421')
    nlp = spacy.load('../../data/models/spacy_model_250421')

    print('Creating stopwords')
    stops = stopwords.words('german')
    common_ocr_errors = ['nnr', 'nnd', 'fich', 'find']
    other_noise = ['januar', 'februar', 'märz', 'april', 'mai', 'juni', 'juli', 'august', 'september', 'october', 'oktober', 'november', 'dezember', 'december']
    stops += common_ocr_errors + other_noise

    print('Sampling data')
    data = df[(df.w_count > 100) & (df.w_count < 1500)].sample(1000).full_text.values

    print('Lemmatizing')
    lemmatized_data = lemmatize(data)

    if len(lemmatized_data):
        print('Data lemmatized')

    print('Preprocessing an converting texts to lists')
    data_words = gen_words(lemmatized_data)

    print('Making bigrams and trigrams')
    bigram_phrases = gensim.models.Phrases(data_words, min_count=2)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=10)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = [bigram[doc] for doc in data_words]
    data_bigrams_trigrams = [trigram[bigram[doc]] for doc in data_words]

    print('Calculating optimal number of topics, alpha and treshold')
    optimal_scores = get_optimal_alpha_and_treshold(data_bigrams_trigrams,
                                                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                [0.1, 0.2, 0.3, 'auto'],
                                                '../../pipeline/04_lda/optimal_n_topics.json',
                                                start=3,
                                                end=24,
                                                step=3,
                                                plot=False)

    print('Finished')