import pandas as pd
import spacy
import time
import concurrent.futures
import multiprocessing

print('Importing spacy model')
nlp = spacy.load('de_core_news_md')


def create_sentences_for_vec(df):

    corpus = []

    for i in df.index:
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

    print('Importing df')
    df = pd.read_parquet('../data/processed/RZ_sample.parquet')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        frames = 4*[df.sample(100)]
        results = [executor.submit(create_sentences_for_vec, df) for df in frames]

        for f in concurrent.futures.as_completed(results):
            print(len(f.result()))
            print(type(f.result()))
            


    stop = time.perf_counter()

    print(f'Finished in {round(stop-start, 2)} seconds')






    

