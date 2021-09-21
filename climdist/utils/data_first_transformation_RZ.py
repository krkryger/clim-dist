import pandas as pd

def normalize_headings(df):
    
    df['head'] = df['head'].str.strip('.')
    df['head'] = df['head'].str.strip(',')
    df['head'] = df['head'].str.strip(' ')
    
    df['head'] = df['head'].replace(to_replace=
                {
            'Witterungs-Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
            'Witterungs- Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
            'Meteorologische Beobachtungen in Riga': 'Witterungsbeobachtungen in Riga',
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

def preprocess_rz(output_path='../../data/processed/RZ_processed.parquet'):

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

    # save the dataframe after changes that are applied
    print('Saving the file...')
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":

    preprocess_rz()
