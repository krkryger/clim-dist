import sys
path = 'C:\\Users\\krister\\Climdist'
if path not in sys.path:
    sys.path.append(path)

import pickle
import string
import numpy as np

class OCR_predict:

    """Class for predicting the OCR quality of a given text from the RZ corpus. Uses default SVM model created on 15.09.21."""

    def __init__(self, nlp_model):

        self.nlp = nlp_model
        self.svm = self.load_svm_model()

        self.common_punctuation = ['.', ',', '?', '!', ';', '-', ':']
        self.common_symbols = set(string.ascii_lowercase).union(string.ascii_uppercase).union(self.common_punctuation)
        self.POS_tags = ['PRON', 'ADV', 'AUX', 'NUM', 'VERB', 'X', 'ADJ', 'SPACE', 'PUNCT']


    def load_svm_model(self, path='../data/models/ocr_quality_model_150921/ocr_quality_model.pkl'):

        with open(path, 'rb') as f:
            model = pickle.load(f)
        
            return model


    def get_text_parameters(self, text):
        
        doc = self.nlp(text)
        doc_properties = {}
        
        # proportions of token POS in doc
        doc_pos = [token.pos_ for token in doc]
        for cat in self.POS_tags:
            doc_properties[cat] = doc_pos.count(cat)/len(doc)
        
        # proportions of weird symbols
        uncommon_symbols = 0
        for symbol in text:
            if symbol not in self.common_symbols:
                uncommon_symbols += 1
                
        # character count to token proportion
        doc_properties['char_to_word'] = len(text)/len(doc)         
        doc_properties['uncommon_symb_proportion'] = uncommon_symbols/len(text)
        
        return doc_properties  


    def predict(self, text, print_text=False):

        """Predict text readability. Use print=True to show the text itself."""
        
        params = self.get_text_parameters(text)
        
        x = np.array(list(params.values()))
        x = x.reshape(1,-1)

        if print_text == True:
            print(text, '\n\n')

        result = self.svm.predict(x)

        return result


# if __name__ == '__main__':

#     import pandas as pd
#     import spacy

#     # load default spacy model
#     print('Loading spacy')
#     nlp = spacy.load('de_core_news_md')

#     print('Loading dataframe')
#     df = pd.read_parquet('../data/processed/RZ_processed.parquet')

#     ocr = OCR_predict(nlp)

#     ocr.create_ocr_column(df, '../data/processed/OCR_readability_column_beta.parquet')