import pandas as pd
from tqdm import tqdm

def basic_preprocess(output_path):

    """Input: takes the two parquet files from data/raw, joins them, generates year, month date columns, checks for cuplicates.
        Output: main processed df for later use."""

    print('Importing raw data...')

    df1 = pd.read_parquet('../../data/external/riga_zeit_1802_1859.parquet')
    df2 = pd.read_parquet('../../data/external/riga_zeit_1859_1888.parquet')
    df = pd.concat([df1, df2])

    # remove duplicate index column
    #df.drop(df.columns[0], axis=1, inplace=True)

    # add separate colums for year, month, day
    df['year'] = df['date'].str.slice(0, 4).astype(int)
    df['month'] = df['date'].str.slice(5, 7).astype(int)
    df['day'] = df['date'].str.slice(8, 10).astype(int)
    df = df[['date', 'year', 'month', 'day', 'pub', 'head', 'full_text', 'href']]

    print(f'There are {len(df)} entries in the dataset with {df.duplicated().sum()} duplicate rows.')
    print('Removing duplicates...')
    df.drop_duplicates(keep='first', inplace=True)
    print(f'Duplicates removed. There are {len(df)} entries in the dataset after removing duplicate entries.')

    # add a column with length of full text
    print('Calculating text lenghts in characters...')
    df['text_len'] = df.full_text.str.len()

    # add a column with word counts
    #print('Counting words...')
    #df['w_count'] = df.full_text.str.split(' ')
    #df['w_count'] = df.w_count.apply(len)

    # normalize the most common headings
    #print('Normalizing headings...')
    #df = normalize_headings(df)

    df = df.rename(columns={'head':'heading'})

    if output_path:
        # save the dataframe after changes that are applied
        print('Saving the file...')
        df.to_parquet(output_path, index=False)

    return df


def normalize_headings(df):

    """Input: main df (after basic preprocess). Strips headings and uses a dict to manually converge some of the most widespread
        variations in heading names.
        Output: main df with normalized headings."""
    
    df['head'] = df['head'].str.strip('.')
    df['head'] = df['head'].str.strip(',')
    df['head'] = df['head'].str.strip(' ')
    
    df['head'] = df['head'].replace(to_replace=
                {
            'Witterungs-Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
            'Witterungs- Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
            'Meteorologische Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
            'Witterungs- Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
            'Witterungs -Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
            'Witterunsbeobachtungen in Riga': 'Witterungsbeobachtnogen in Riga',
            'Groszbritannien und Irland': 'Großbritannien und Irland',
            'Grossbritannien und Irland': 'Großbritannien und Irland',
            'Grosbritannien und Irland': 'Großbritannien und Irland',
            'Todes – Anzeige': 'Todes-Anzeige',
            'Todes– Anzeige': 'Todes-Anzeige',
            'Todes-Anzeigen': 'Todes-Anzeige',
            'Todes – Anzeigen': 'Todes-Anzeige',
            'Telegraphische Nachrichten': 'Telegramme',
            'Lokales': 'Locales',
            'Börsen- und Handels – Nachrichten': 'Börsen- und Handels-Nachrichten',
            'Börsen – und Handels – Nachrichten': 'Börsen- und Handels-Nachrichten',
            'Börsen und Handels-Nackrichten': 'Börsen- und Handels-Nachrichten',
            'Börsen – und Handels – Nachrichten': 'Börsen- und Handels-Nachrichten',
            'Telegraphische Depesche der „Rigaschen Zeitung"': 'Telegramme der „Rigaschen Zeitung"',
            'Witterungs-Telegramme': 'Telegraphische Witterungsberichte',
            'Vermischte Nachrichten': 'Vermischtes',
            'Tägliche Eisenbahuzüge': 'Eisenbahnzüge',
            'Tägliche – Eisenbahnzüge': 'Eisenbahnzüge',
            'Telegr. Berichte über den Barometerstand': 'Telegraphische Witterungsberichte',
            'Inländische Nachrichten': 'Inland',
            'Inlandische Nachrichten': 'Inland',
            'Telegr. der Rig. Telegraphen Agentur': 'Telegramme',
            'Folgende Personen sind gesonnen, von hier zu reise': 'Abreisende',
            'Nachstehende Personen zeigen ihre Abreise von hier': 'Abreisende',
            'Tägliche – Eisenbahnzüge': 'Eisenbahnzüge',
            'Deutsches «eich': 'Deutsches Reich',
            'Neueste Rachrichten': 'Neueste Nachrichten',
            'Neuste Nachrichten': 'Neueste Nachrichten',
                }
            )
    
    return df


def generate_alt_headings(df, ignorelocs, nlp,
                          output_path):

    """Input: main df with normalized headings. Use a trained NLP model for detecting LOC entities in headings.
        Output: column with alternate headings for simpler geographical analysis.
        Can take 1-2 hours."""
    
    df['heading2'] = df['heading']
    
    for i in tqdm(df.index):
        title = df.loc[i, 'heading2']
        doc = nlp(title)
        locs = [ent.text for ent in doc.ents if ent.label_ == 'LOC']
        if len(locs) == 1:
            if locs[0] not in ignorelocs:
                df.loc[i, 'heading2'] = locs[0]

    if output_path:
        df['heading2'].to_parquet(output_path, index=True)
    
    return df['heading2']


def generate_ocr_column(df, output_path):

    """Input: (preprocessed) main df.
        Output: column with a OCR readability prediction for each entry.
        Uses the default spacy model and the class OCR_predict to generate either 0 or 1 for each column. Can take 15-20 hours."""

    import spacy
    from climdist.ocr_quality import OCR_predict

    # load default spacy model
    print('Loading spacy')
    nlp = spacy.load('de_core_news_md')

    print('Loading dataframe')
    df = pd.read_parquet('../data/processed/RZ_processed.parquet')

    ocr = OCR_predict(nlp)
    
    readable = []

    for i in tqdm(df.index, mininterval=5, maxinterval=30, colour='green'):
        text = df.loc[i, 'full_text']
        readable.append(ocr.predict(text))
        
    column = pd.DataFrame(data=readable, index=df.index, columns=['readable'])

    if output_path:
        column.to_parquet(output_path, index=True)

    print('Finished')
    return column


def complete_data_preprocess():

    df = basic_preprocess(output_path='../../data/processed/RZ_processed.parquet')
    df = normalize_headings(df)
    
    import spacy
    
    nlp = spacy.load('../data/models/spacy_model_250421/')
    ignorelocs = ['Riga', 'Rigasche', 'Rigaschen', 'Rigische']
    generate_alt_headings(df, ignorelocs=ignorelocs, nlp=nlp, output_path='../data/processed/RZ_heading2.parquet')

    generate_ocr_column(df, output_path='../data/processed/OCR_readability_column.parquet')

    print('Data preprocessing finished')


#if __name__ == "__main__":
#
#   complete_data_preprocess()

    
