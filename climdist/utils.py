import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm.notebook import tqdm
from wordcloud import WordCloud
import matplotlib.font_manager as fm


def load_df(which):

    """Load the Rigasche Zeitung data.
        which: str, 'main', 'sample'
        readability: bool, include the OCR readability column
        heading2: bool, include the heading2 column"""

    if which == 'main':
        df = pd.read_parquet('../data/processed/RZ_processed.parquet')
        df['readability'] = pd.read_parquet('../data/processed/RZ_readability.parquet')


        #if readability == True:
        #if heading2 == True:
            #df['heading2'] = pd.read_parquet('../data/processed/RZ_heading2.parquet')

    elif which == 'sample':
        df = pd.read_parquet('../data/processed/RZ_sample.parquet')

    return df


class Analyse():
    
    def __init__(self):
        
        print('Importing main dataframe')
        self.df = load_df('main')
        self.df = self.df[self.df.readability==True]
        
        print('Importing processed articles')
        self.sentences = list(self.load_sentences())
        
        print('Importing detected keywords')
        with open('../data/processed/all_keywords_100522.json', 'r', encoding='utf') as f:
            self.keywords = json.load(f)
            
        print('Importing auxiliaries')
        self.ruler = pd.read_excel('../pipeline/ruler_patterns_060522.xlsx', index_col='Unnamed: 0')
        self.wea_df = pd.read_csv('../data/processed/wea_df_010522.tsv', sep='\t', encoding='utf8', index_col=0)
        self.wordform_df = pd.read_csv('../data/processed/wordform_df_010522.tsv', sep='\t', encoding='utf8', index_col=0)
        self.phenomenon_df = pd.read_csv('../data/processed/phenomenon_df_110522.tsv', sep='\t', encoding='utf8', index_col=0)
        
        self.ix_to_year = self.df['year'].to_dict()
            
        self.font_path = '../references/cmunorm.ttf'
        fm.fontManager.addfont(self.font_path)

        print('Ready!')
        
    def wordform_to_key(self, wordform):
        return list(self.ruler.key[self.ruler.wordform == wordform].unique())
    
    def phenomenon_to_key(self, phenomenon):
        return list(self.ruler.key[self.ruler.phenomenon == phenomenon].unique())
    
    def phenomenon_to_wordform(self, phenomenon):
        return list(self.ruler.wordform[self.ruler.phenomenon == phenomenon].unique())
        
    def load_sentences(self):
        with open('../data/processed/RZ_sentences.jsonl', 'r', encoding='utf8') as f:
            for ix, line in tqdm(enumerate(f.readlines())):
                yield json.loads(line)
                
                
    def ids_by_kw(self, keys, timerange, index_filter=None): 
        """Returns indexes of all entites that contain a certain keyword"""
    
        if index_filter is None:
            index_filter = self.df.index
        keys=set(keys)
        results = []
        
        for sentence, entry in zip(self.sentences, self.keywords):
            if sentence['id'] in index_filter \
            and len(entry['ents']) > 0 \
            and self.df.loc[sentence['id'], 'year'] in timerange \
            and len(set([ent[0] for ent in entry['ents']]) & keys) > 0: 
                    results.append(sentence['id'])
                    
        return results
    
    
    def headings_by_kw(self, words, timerange=range(1802,1889), phenomenon=False): 
        """Returns a pd.Series.value_counts() object of headings of articles that contain a given keyword"""
    
        if phenomenon == True:
            keys = set(self.ruler.key[self.ruler.phenomenon.isin(words)])
        else:
            keys = set(self.ruler.key[self.ruler.wordform.isin(words)])    
        ids = self.ids_by_kw(keys, timerange)
        
        return self.df.heading2.loc[ids].value_counts()
    
    
    def word_context(self, word, window, index_filter=None, phenomenon=False):
        """Input: wordform. Returns a list of context words in a defined window"""
    
        if index_filter is None:
            index_filter = self.df.index
        if phenomenon == True:
            keys = self.phenomenon_to_key(word)
        else:
            keys = self.wordform_to_key(word)
            
        results = []
        
        for sentence, entry in tqdm(zip(self.sentences, self.keywords)):
            if sentence['id'] in index_filter:
                if len(entry['ents']) > 0:
                    for ent in entry['ents']:
                        if ent[0] in keys:
                            ix = ent[1]
                            context_before = sentence['text'][ix-window:ix]
                            context_after  = sentence['text'][ix+1:ix+window]
                            results.extend(context_before + context_after)
           
        return results
    
    
    def keyword_plot(self, keywords, cat='wordform', kind='line', wordform=True, relative=True, savepath=None, **kwargs):
    
        if cat == 'key':
            plot_df = self.wea_df
        elif cat == 'wordform':
            plot_df = self.wordform_df
        elif cat == 'phenomenon':
            plot_df = self.phenomenon_df
        else:
            return('Please select a valid CAT')
        
        matplotlib.rcParams['font.family'] = 'CMU Concrete'
        col = sns.color_palette("muted", len(keywords))
        plt.figure(figsize=(15,7))
        ticksrange = np.arange(1802,1890)
        xlabels = [num if num%5==0 else '' for num in np.arange(1802,1890)]
        
        ### lineplots
        if kind == 'line':
            if relative:
                for word in keywords:
                    plt.plot(plot_df[word]/plot_df.total, label=word)             
            else:
                for word in keywords:
                    plt.plot(plot_df[word], label=word)
                
        ### stackplots            
        if kind == 'stack':
            if relative == True:
                plt.stackplot(np.arange(1802,1889),
                              [plot_df[word]/plot_df.total for word in keywords],
                              labels=keywords,
                              colors=col)
            else:
                plt.stackplot(np.arange(1802,1889), [plot_df[word] for word in keywords], labels=keywords)    
        
        plt.grid(b=True, which='both')
        plt.yticks(fontsize=14)
        plt.xticks(ticks=ticksrange, labels=xlabels, fontsize=14)
        plt.tick_params(axis ='x', rotation = 45)
        plt.setp(plt.gca().spines.values(), color='black')
        plt.gca().set_frame_on(True)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], fontsize=12, **kwargs)
    
        if relative == True:
            plt.ylabel("part de tous les mots-clés", fontsize=18)
        else:
            plt.ylabel("nb. d'occurrences", fontsize=18)
            
        sns.set_style(style='whitegrid')
        
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
        
        plt.show()
        
        
    def context_wordcloud(self, context_words, savepath=None):
    
        wordcloud_data = ' '.join([word for word in context_words])
        
        print(f'Generated from {len(wordcloud_data.split())} context words')
        wc = WordCloud(background_color='white', width=4800, height=1600, font_path='../references/cmunrm.ttf')
        wc.generate(wordcloud_data)
        plt.figure(figsize=(24,8))
        plt.axis('off')
        
        plt.imshow(wc)
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
        plt.show()
                    
    