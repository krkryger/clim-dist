import pandas as pd
import spacy
from spacy import displacy
import numpy as np

from climdist.ocr.spellcorrection import spelling_correction

def generate_ocr_testbatch(input, batchsize, maxlen=None, maxwords=None):
    
    if type(input) == pd.core.frame.DataFrame:
        df = input
    else:
        try:
            print('Loading data from', input)
            df = pd.read_excel(input)
        except:
            print('Please provide a valid path or a dataframe object')
    
    if maxlen != None:
        data = df[df.text_len <= maxlen]
    elif maxwords != None:
        data = df[df.w_count <= maxwords]
    else:
        raise KeyError("Must provide either maxlen or maxwords")
    
    print('Randomizing...')
    result = data.iloc[np.random.randint(0, high=len(data), size=batchsize)]
    print('Finished!')
    return result

from climdist.ocr.spellcorrection import spelling_correction

def evaluate_spelling_for_ner(data, nlp_model, speller=None):
    
    displacy_color_code = {'WEA': '#4cafd9',
                  'PER': '#ffb366',
                  'DAT': '#bf80ff',
                  'LOC': '#a88676',
                  'MISC': 'grey',
                  'MEA': '#85e085',
                  'ORG': '#5353c6'}

    displacy_options = {'ents': ['WEA', 'PER', 'DAT', 'LOC', 'MISC', 'MEA', 'ORG'], 'colors': displacy_color_code}

    evaluation_results = {}
    ent_labels = nlp_model.get_pipe("ner").labels
    
    for entry in range(0, len(data)):
        
        text = data.iloc[entry].full_text.replace('\n', ' ')
        href = data.iloc[entry].href
        index = data.index[entry]

        if speller:
            text = spelling_correction(text,speller)
            
        print(f'Starting entry {index}: {data.iloc[entry].pub}, {data.iloc[entry].date}')
        print(href)
            
        doc = nlp_model(text)
        wordcount = len(doc)
        doc_labels = [ent.label_ for ent in doc.ents]
        entry_results = {}
            
        for label in ent_labels:

            displacy.render(doc, style='ent', jupyter=True, options=displacy_options)
            
            if label in set(doc_labels):

                label_positives = doc_labels.count(label)
        
                print(label)
                while True:
                    try:
                        TP = int(input('True Positives: '))
                    except ValueError:
                        print("Sorry, I didn't understand that.")
                        continue
                    else:         
                        break
                
                FP = label_positives - TP
                print(f'False Positives: {FP}')

            else:
                TP = 0
                FP = 0

            print(label)
            while True:
                try:
                    FN = int(input('False Negatives: '))
                except ValueError:
                    print("Sorry, I didn't understand that.")
                    continue
                else:
                    break
            
            TN = wordcount - FN - TP
            print(f'True Negatives: {TN}')

            label_results = [TP,FP,TN,FN]
            print(label, label_results)

            entry_results[label] = label_results
        print(entry, entry_results)
        
        evaluation_results[index] = entry_results     
            
    print(evaluation_results)
    return(evaluation_results)


def get_precision_recall(evaluation_results, labels=None):

    if labels == None:
        labels = ['WEA', 'LOC', 'DAT', 'PER', 'MISC', 'ORG', 'MEA']

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for entry in evaluation_results:
        results = evaluation_results[entry]
        for label in results.keys():
            if label in labels:
                scores = results[label]
                tp += scores[0]
                fp += scores[1]
                tn += scores[2]
                fn += scores[3]

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print(tp, 'true positives')
    print(fp, 'false positives')
    print(tn, 'true negatives')
    print(fn, 'false negatives')
    print('\n')
    
    print(f'From {len(evaluation_results)} entries, for labels {labels}: precision: {precision}, recall: {recall}')
    
    return (precision, recall)


def get_fscore(precisionrecall):

    precision = precisionrecall[0]
    recall = precisionrecall[1]

    f = 2 / ((recall**-1) + (precision)**-1)

    return f






