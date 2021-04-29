import climdist.ner.doccano_transformations as dt

def prepare_data(input_path):

    print(f'Importing data from {input_path}')
    train_data = dt.doccano_to_spacy(input_path)
    print(f'Stripping spaces from entity positions')
    train_data = dt.doccano_strip(train_data)
    train_data = dt.doccano_strip(train_data)
    print('Fixing labels')
    train_data = dt.wea_to_nat(train_data)

    return train_data


def train_spacy(train_data, model_path, output_dir):

    print('Loading Spacy model')
    import spacy
    from spacy.util import minibatch, compounding 
    from spacy.training import Example
    nlp = spacy.load(model_path)
    ner = nlp.get_pipe("ner")

    for annotations in train_data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])
            
    print(f'The labels used in the annotation data are {ner.labels}')

    iterations = int(input('Define the number of iterations to train Spacy NER pipe: '))

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(0, iterations):
            print("iteration: " + str(itn))
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for ba in batch:
                    examples.append(Example.from_dict(nlp.make_doc(ba["text"]), ba))
                    nlp.update(examples)        
    print("Training is finished!")

    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)


if __name__ == '__main__':

    input_path = '../../pipeline/03_ner/01_doccano/storms_annotated_19.03v2.jsonl'
    output_dir = '../../data/models/test_ner_model/'

    train_spacy(input_path=input_path, output_dir=output_dir)