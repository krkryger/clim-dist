import spacy
import codecs
import json
from spacy.tokens import Span


def spacy_to_doccano(doc, output_path, sep='SPACE'):
    with codecs.open(output_path, 'w', encoding='utf8') as f:
        ct = 0
        marker = 0
        for token in doc:
            ct += 1
            if token.pos_ == sep:
                aspan = doc[marker:ct]  # créer un objet span qui correspond à une seule entrée (séparées par des espaces)
                marker = ct
                labels = []
                text = aspan.text
                if (len(aspan.ents) > 0):
                    for ent in aspan.ents:
                        labels.append(
                            [ent.start_char - aspan.start_char,
                             ent.end_char - aspan.start_char,
                             ent.label_])
                if (len(labels) > 0):
                    sentence = {"text": text, "labels": labels}
                else:
                    sentence = {"text": text}
                json_string = json.dumps(sentence, ensure_ascii=False)
                f.write(json_string)
                f.write("\n")
            else:
                pass

    f.close()
    print("File created :", output_path)


def pandas_to_doccano(df, nlp, output_path):
    
    with codecs.open(output_path, 'w', encoding='utf8') as f:
    
        for text in df.full_text:

            doc = nlp(text)
            labels = []
            if len(doc.ents) > 0:
                for ent in doc.ents:
                    labels.append([ent.start_char,
                                  ent.end_char,
                                  ent.label_])
            if len(labels) > 0:
                sentence = {'text': text, 'labels': labels}
            else:
                sentence = {'text': text}
            json_string = json.dumps(sentence, ensure_ascii=False)
            f.write(json_string)
            f.write('\n')
    print('File created!')


def create_ents_dico(doccano_data):
        
    ents_dico = {}
    
    for entry in doccano_data:
        entset = set([annotation['label'] for annotation in entry['annotations']])
        for entlabel in entset:
            if entlabel not in ents_dico.keys():
                for annotation in entry['annotations']:
                    if annotation['label'] == entlabel:
                        print(entry['text'][annotation['start_offset']:annotation['end_offset']])
                newlabel = input('label: ')
                ents_dico[entlabel] = newlabel
                print('\n')
                
    print(ents_dico)
    return ents_dico



def doccano_to_spacy(input_path, labels_dico):

    output_data = []

    labels_dico = labels_dico

    with codecs.open(input_path, "r", encoding='utf8') as f:
        lines = f.readlines()

        for line in lines:
            line = json.loads(line)
            entities = []

            if len(line['annotations']) > 0:
                for annotation in line['annotations']:
                    entity = (annotation["start_offset"],
                              annotation["end_offset"],
                              labels_dico[annotation["label"]])
                    entities.append(entity)

            output_data.append({"text": line['text'], "entities": entities})

        return output_data
        print('Doccano data imported')


def doccano_strip(data):

    for entry in data:
        counter = 0
        for ent in entry['entities']:
            entity_text = entry['text'][ent[0]:ent[1]]
            if entity_text.strip() != entity_text:
                newent = (ent[0], ent[1] - 1, ent[2])
                entry['entities'][counter] = newent
            counter += 1

    return data


def wea_to_nat(data):

    for entry in data:
        entity_list = entry['entities']
        index = 0
        for entity in entity_list:
            if entity[2] == 'WEA':
                entity_list[index] = (entity[0], entity[1], 'NAT')
            index += 1

    return data