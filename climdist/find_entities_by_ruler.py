import spacy
import pandas as pd
from tqdm import tqdm
from climdist.data import load as cd_load
import json
import time

print('Importing dataframe and NLP model')
df = cd_load('main', readability=True, heading2=False)

nlp = spacy.load('../data/models/spacy_ner_151121/model-best/', disable=['ner'])
nlp.add_pipe('entity_ruler', last=True).from_disk('../data/models/entity_ruler_171121/patterns.jsonl')


def find_entities(df=df, nlp=nlp):
   
    results = []

    for ix in tqdm(df.index, mininterval=3):
        text = df.loc[ix, 'full_text']
        date = df.loc[ix, 'date']
        doc = nlp(text)

        WEA = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents
                if ent.label_ == 'WEA']
        
        results.append({'id': ix,
                        'date': date,
                        'ents': {'WEA': WEA}}) 
        
    return results


start = time.perf_counter()
wea_entities = find_entities(df, nlp)

with open('../data/processed/entities_ruled.json', 'w', encoding='utf8') as f:
    json.dump(wea_entities, f)

stop = time.perf_counter()
duration = divmod(round(stop-start, 2), 60)
print(f'Finished in {duration[0]} minutes, {round(duration[1], 1)} seconds')