import pandas as pd
from tqdm import tqdm
import re
import json
from tqdm import tqdm
import numpy as np


def basic_preprocess():
    """Input: takes the two parquet files from data/raw, joins them, generates year, month date columns, checks for cuplicates.
        Output: main processed df for later use."""

    print('Importing raw data...')

    df1 = pd.read_parquet('../data/external/riga_zeit_1802_1859.parquet')
    df2 = pd.read_parquet('../data/external/riga_zeit_1859_1888.parquet')
    df = pd.concat([df1, df2])

    print('Generating columns: year, month, day')

    # remove duplicate index column
    #df.drop(df.columns[0], axis=1, inplace=True)

    # add separate colums for year, month, day
    df['year'] = df['date'].str.slice(0, 4).astype(int)
    df['month'] = df['date'].str.slice(5, 7).astype(int)
    df['day'] = df['date'].str.slice(8, 10).astype(int)
    df = df[['date', 'year', 'month', 'day', 'pub', 'head', 'full_text', 'href']]
    df.columns = ['date', 'year', 'month', 'day', 'pub', 'heading', 'full_text', 'href']

    print(f'There are {len(df)} entries in the dataset with {df.duplicated().sum()} duplicate rows.')
    print('Removing duplicates...')
    df.drop_duplicates(keep='first', inplace=True)
    print(f'Duplicates removed. There are {len(df)} entries in the dataset after removing duplicate entries.')

    # add a column with length of full text
    print('Calculating text lenghts in characters...')
    df['text_len'] = df.full_text.str.len()

    return df


def get_places_and_dates(heading):
    """Matches all the headings to a regex pattern that finds placenames
    and dates in a certain common heading format"""

    re_pattern = '(?:Aus |AuS |Vom |Aus dem |(Schreiben aus ))?(?P<placename>(St. )?[A-Z][a-züöä]+)(?:( im )|(, im ))?(?P<gouv>(([A-Za-z][a-züöä]+chen Gou[a-z]+)|(Gou[a-züöä]+ )[A-Z][a-züöä]+)?)?(?:,|, |., |.,)(?:(den )|(vom ))(?P<date>\d{1,2})(?:\.|ten|sten)?'

    match = re.search(re_pattern, heading)
    if match:
        return match.group('placename'), match.group('gouv'), match.group('date')
    else:
        return pd.NA


def apply_regex(df):
    """Generates columns for placename and the date of the origin of the info"""
    
    print('Applying regex to search for placenames and dates')
    patterns_in_headings = [get_places_and_dates(line) for line in df.heading]
    
    placenames = [entry[0] if type(entry) == tuple else entry for entry in patterns_in_headings]
    #gouvernements = [entry[1] if type(entry) == tuple else entry for entry in patterns_in_headings]
    dates = [entry[2] if type(entry) == tuple else entry for entry in patterns_in_headings]
    
    df['placename'] = placenames
    #df['gouvernement'] = gouvernements
    df['origin_date'] = dates


def find_grenze(df):
    """Handles an exception that is missed by the regex pattern"""

    for ix, name in df.placename[df.placename.notna()].iteritems():
        try:
            if 'Gränze' in name:
                heading_words = [word.strip('.,') for word in re.split('\s|,|-|.]', df.loc[ix, 'heading'])]
                grenze_ix = heading_words.index('Gränze') 
                new_placename = ' '.join(heading_words[grenze_ix-1:grenze_ix+1])
                #print(new_placename)
                df.loc[ix, 'placename'] = new_placename
            elif 'Grenze' in name:
                heading_words = [word.strip('.,') for word in re.split('\s|,|-|.]', df.loc[ix, 'heading'])]
                grenze_ix = heading_words.index('Grenze') 
                new_placename = ' '.join(heading_words[grenze_ix-1:grenze_ix+1])
                #print(new_placename)
                df.loc[ix, 'placename'] = new_placename
            else:
                pass
        except:
            pass
                
    
def remove_weekday_false_positives(df): 
    """Handles an exception that is missed by the regex pattern"""
    weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Sonntag', 'Sonnabend', 'Sonntag']
    for name in weekdays:
        df.placename.replace(name, pd.NA, inplace=True)


def remove_false_dates(df):
    """Handles an exception that is missed by the regex pattern"""
    df.origin_date.fillna(0, inplace=True) 
    df['origin_date'] = [date if int(date) in range(1,32) else pd.NA for date in df.origin_date]


def generate_heading2(df):

    apply_regex(df)
    find_grenze(df)
    remove_weekday_false_positives(df)
    remove_false_dates(df)

    with open('../pipeline/heading_replacement_dict.json', 'r', encoding='utf') as f:
        replacement_dict = json.load(f)

    df['heading2'] = df['heading'].copy()

    print('Generating heading2...')
    for key, value in tqdm(replacement_dict.items()):
        df.heading2.replace(key, value, inplace=True)

    df['heading2'] = df['heading2'].str.strip('., ')
    df['heading2'].loc[df.placename.notna()] = df.placename
    df.heading2.replace('Ist zu drucken erlaubt. Im Namen des General-Gouve', 'Ist zu drucken erlaubt...', inplace=True)
    df = df.fillna(value=np.nan)


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


def create_sample_df(df):

    sample_df = df.sample(10000)
    sample_df.to_parquet('../data/processed/RZ_sample.parquet')
    print('Created sample')


if __name__ == "__main__":

    df = basic_preprocess()
    generate_heading2(df)
    create_sample_df(df)
    df.to_parquet('../data/processed/RZ_processed.parquet')
    print('Finished')

    
