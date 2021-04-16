import pandas as pd
import spacy

input_path = '../../data/processed/storms_for_annotation.xlsx'
output_dir = '../../pipeline/03_ner/01_doccano/'

# importer les données saisies à la main depuis un fichier Excel.
# il y a surtout des tempêtes, mais aussi un certain nombre des autres phénomènes
print('Reading Excel')
df = pd.read_excel(input_path)

print('Preparing data')
# supprimer les colonnes vides
df = df[['YEAR', 'MONTH', 'DATE_BEGIN', 'DATE_END', 'CAT_ID', 'COMP_ID','LOD_ID', 'EXC', 'COMMENT', 'LINK']]

# certaines entrées n'ont pas de l'extrait de la source - les supprimer
df = df.loc[pd.notna(df['EXC'])]

# créer une colonne qui mésure la longueur des extraits (len(str))
df['len'] = df['EXC'].apply(len)

# créer un nouvel dataframe qui ne contient que les entrées dont la longueur de l'extrait dépasse une longueur donnée
# (j'ai choisi 100 charactères pour donner un peu plus de contexte et éliminer les descriptions très laconiques d'un point de vue NLP)
# cependant, la limite est tout à faite arbitraire à ce moment
shortdf = df[df['len'] > 99]

# mettre en place un ordre chronologique pour le dataframe
shortdf = shortdf.sort_values(['YEAR', 'MONTH', 'DATE_BEGIN'])

# écrire tous les extraits sélectionnés dans un fichier .txt

output_path = output_dir + 'first_ner_annotation.txt'
print('Saving data as .txt file')
with open(output_path, 'w', encoding = 'utf8') as myfile:
    for txt in shortdf['EXC']:
        storm = txt + '\n\n'
        myfile.write(storm)
print(f'Wrote files to {output_path}')

# j'ai décidé d'experimenter avec Spacy, mais il reste à trouver si il y a des modèles pour l'allemand historique, p. ex. en NLTK
print('Loading Spacy model "de_core_news_md')
nlp = spacy.load("de_core_news_md")

# Import same .txt file to create Spacy object
with open (output_path, 'r', encoding='utf8') as myfile:
    data = myfile.read()

print('Creating Spacy doc')
doc = nlp(data)

import climdist.ner.doccano_transformations as dt

doccano_output_path = output_dir + 'storms_to_annotate.jsonl'
print('Preparing .jsonl for Doccano')
dt.spacy_to_doccano(doc, doccano_output_path)

print('Finished!')

