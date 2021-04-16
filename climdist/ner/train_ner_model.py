import climdist.ner.doccano_transformations as dt

input_path = '../../pipeline/03_ner/01_doccano/storms_annotated_19.03v2.jsonl'

print(f'Importing data from {input_path}')
train_data = dt.doccano_to_spacy(input_path)
print(f'Stripping spaces from entity positions')
train_data = dt.doccano_strip(train_data)

print('Loading Spacy')
import spacy
from spacy.util import minibatch, compounding 
from spacy.training import Example
nlp = spacy.load("de_core_news_md")
ner = nlp.get_pipe("ner")

for annotations in train_data:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])
        
print(f'The labels used in the annotation data are {ner.labels}')

iterations = int(input('Define the number of iterations to train Spacy NER pipe: '))

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(0, iterations):
        print("iteration: "+str(itn))
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            examples = []
            for ba in batch:
                examples.append(Example.from_dict(nlp.make_doc(ba["text"]), ba))
                nlp.update(examples)        
print("Training is finished!")

output_dir = '../../data/models/ner_test'
nlp.to_disk(output_dir)
print("Saved model to", output_dir)