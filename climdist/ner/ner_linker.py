import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import edit_distance


class NERlinker:
    def __init__(self, data, nlp, placename_files, ignorelocs=None):
        self.nlp = nlp
        self.data = data
        self.placename_files = placename_files
        self.ignorelocs = ignorelocs
        self.loc_data = self.create_loc_data()
        self.altnames = self.create_altnames()
        self.ed = edit_distance.edit_distance
        self.replace_dict = {"ß": "ss"}

    def create_loc_data(self):

        loc_data = []

        for file in self.placename_files:
            placenames = pd.read_csv(
                file, sep="\t", header=None, usecols=[0, 1, 3, 4, 5]
            )
            placenames.columns = ["id", "name", "altnames", "lat", "long"]
            loc_data.append(placenames)

        return loc_data

    def create_altnames(self):

        altnames = []

        for table in self.loc_data:
            namelist = []
            for lst in list(table.altnames.str.split(",")):
                if type(lst) == list:
                    for name in lst:
                        namelist.append(name)

            altnames.append(namelist)

        return altnames

    def create_docs(self):
        for text in self.data:
            yield self.nlp(text)

    def entity_positions(self, doc, entlabel="WEA"):

        positions = []
        for i in range(0, len(doc)):
            if doc[i].ent_iob == 3:
                positions.append(i)

        # leave out all other labels than entlabel
        for i in range(0, len(doc.ents) - 1):
            if doc.ents[i].label_ != entlabel:
                positions[i] = 0
        positions = [pos for pos in positions if pos != 0]

        return positions

    def build_concordances(
        self, doc, entdist, window_before, window_after, entlabel="WEA", render=False
    ):

        positions = self.entity_positions(doc)

        if positions == []:
            print("No WEA entity groups in entry")
            return None

        diff = (pos1 - pos0 for pos0, pos1 in zip(positions, positions[1:]))
        split_at = [i for i, dif in enumerate(diff, 1) if dif >= entdist]
        ent_groups = [positions[i:j] for i, j in zip([0] + split_at, split_at + [None])]
        ent_groups = [group for group in ent_groups if len(group) > 1]

        ct = 1

        for group in ent_groups:
            print(f"Building concordance {ct}")
            if window_before < group[1]:
                concordance_start = group[1] - window_before
            else:
                concordance_start = 0
            if window_after < (len(doc) - group[-1]):
                concordance_end = group[-1] + window_after
            else:
                concordance_end = len(doc)

            concordance = doc[concordance_start:concordance_end]

            if render == True:
                try:
                    self.render(concordance)
                except:
                    print("Displacy failed to render entities")

            ct += 1

            yield concordance

    def closest_name(self, string, namelist):

        if string in namelist:
            return (string, 0)

        else:
            string = string.lower()
            bestname = ""
            bestdist = len(string)

            for name in namelist:
                dist = self.ed(string, name.lower())[0]
                if dist < bestdist:
                    bestname = name
                    bestdist = dist
                    if dist == 1:
                        break

            for key in self.replace_dict.keys():
                if key in string:
                    string = string.replace(key, self.replace_dict[key])

                    for name in namelist:
                        dist = self.ed(string, name.lower())[0]
                        if dist < bestdist:
                            bestname = name
                            bestdist = dist
                            if dist == 0:
                                break

            return (bestname, bestdist)

    def get_loc_names(self, doc, maxdist=1, stripwords=False):

        if stripwords == True:
            from climdist.ocr.spellcorrection import strip_word

        entset = set([ent.text for ent in doc.ents if ent.label_ == "LOC"])

        loc_names = []

        if len(entset) > 0:
            print(f"LOC-s in concordance: {entset}")

            for entname in entset:
                if stripwords == True:
                    entname = strip_word(entname, extra_symbols="?!|«»<>_'")
                if entname in self.ignorelocs:
                    print(f'Passing {entname}', '\n')
                    continue
                print("Searching for", entname)
                candidates = {}
                for namelist in self.altnames:
                    location = self.closest_name(entname, namelist)
                    if location[1] <= maxdist:
                        candidates[location[1]] = location[0]

                if len(candidates) > 0:
                    loc_name = candidates[min(candidates.keys())]
                    print("Best match:", loc_name, "\n")
                    loc_names.append(loc_name)
                else:
                    print("No match found \n")

            return loc_names
        else:
            return None

    def get_location_id(self, loc_name):

        for file in self.loc_data:
            ix = file[file.altnames.str.contains(loc_name) == True]

            if len(ix) > 0:
                location_id = int(ix.id.values[0])
                location_lat = float(ix.lat.values[0])
                location_long = float(ix.long.values[0])

        return (location_id, location_lat, location_long)

    def link(self, entdist, window_before, window_after, output_path, stripwords=False, max_edit_dist=1, build_concordances=True, render=False):

        import codecs
        import json

        with open(output_path, "w", encoding="utf8") as f:

            output = {}

            for i in range(0, len(self.data)):
                
                entry_id = int(self.data.iloc[i].id)

                print("------------------------------")
                print(f"Starting entry {entry_id} ({i+1}/{len(self.data)})")
                print(self.data.iloc[i].pub, self.data.iloc[i].date,)
                print("------------------------------")

                # create doc
                print("Applying NLP")
                doc = self.nlp(self.data.iloc[i].full_text)

                entry_locations = {}

                if build_concordances == True:

                    # build concordance
                    for concordance in self.build_concordances(
                        doc,
                        entdist=entdist,
                        entlabel="WEA",
                        window_before=window_before,
                        window_after=window_after,
                        render=render,
                    ):

                        loc_names = self.get_loc_names(concordance, maxdist=max_edit_dist, stripwords=stripwords)

                        if loc_names:
                            for name in loc_names:
                                entry_locations[name] = self.get_location_id(name)

                else:

                        loc_names = self.get_loc_names(doc, maxdist=max_edit_dist, stripwords=stripwords)

                        if loc_names:
                            for name in loc_names:
                                entry_locations[name] = self.get_location_id(name)


                entry_data = {entry_id: entry_locations}
                print(entry_data, "\n\n")

                output[entry_id] = entry_locations

            json.dump(output, f, ensure_ascii=False, separators=(", ", ": "), indent=4)

    def render(self, doc):
        from spacy import displacy

        displacy_color_code = {
            "WEA": "#4cafd9",
            "PER": "#ffb366",
            "DAT": "#bf80ff",
            "LOC": "#a88676",
            "MISC": "grey",
            "MEA": "#85e085",
            "ORG": "#5353c6",
        }

        displacy_options = {
            "ents": ["WEA", "PER", "DAT", "LOC", "MISC", "MEA", "ORG"],
            "colors": displacy_color_code,
        }

        displacy.render(doc, style="ent", jupyter=True, options=displacy_options)
