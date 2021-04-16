import pandas as pd
import numpy as np

input_path = '../../data/external/LNB_33K_full_text_results_1800_1900_Sturm_Hagel_Uberschwemmung.xlsx'
output_path = '../../data/processed/processed.xlsx'

print('Importing raw data...')
df = pd.read_excel(input_path)

# remove duplicate index column
df.drop(df.columns[0], axis=1, inplace=True)

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
print('Counting words...')
df['w_count'] = df.full_text.str.split(' ')
df['w_count'] = df.w_count.apply(len)


# create masks to define if text of an entry contains the keywords or their equivalents
# (alternative wordforms: the search mechanism of periodika.lv which was used to create the corpus uses stemming)
print('Counting storms...')
df['sturm'] = np.where((df.full_text.str.contains('Sturm')) |
                       (df.full_text.str.contains('Stürm')) |
                       (df.full_text.str.contains('sturm')) |
                       (df.full_text.str.contains('stürm')), True, False)

print('Counting hailstorms...')
df['hagel'] = np.where((df.full_text.str.contains('Hagel')) |
                       (df.full_text.str.contains('hagel')), True, False)

print('Counting floods...')
df['überschwemmung'] = np.where((df.full_text.str.contains('Überschwemmung')) |
                                (df.full_text.str.contains('überschwemmung')) |
                                (df.full_text.str.contains('berschwemm')), True, False)


# save the dataframe after changes that are applied
print('Saving the file...')
df.to_excel(output_path, index=False)
