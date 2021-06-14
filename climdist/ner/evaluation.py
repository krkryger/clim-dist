def evaluate_ner(pred, gold):
    
    '''Inputs are a Spacy NLP object and a gold standard annotation.
    Returns two a tuple of two lists of booleans, first for gold standards, second for predictions.
    Lists can later be compared with Scikit-learn.confusion_matrix to get TP, FP, TN, FN scores.'''
    
    # dico for converting NE types from integer to string (ex. 385 to 'LOC')
    types_labels = {0:0}
    for ent in pred.ents:
        types_labels[ent.label] = ent.label_
        
    # the spacy prediction is tokenised, unlike the gold standard annotation.
    pred_tokens = []
    pred_entities = []
    pred_positions = []
    
    # loop over these tokens to get their positions in the doc and their entity types
    for token in pred:
        pred_tokens.append(token)
        pred_entities.append(types_labels[token.ent_type])
        pred_positions.append((token.idx,token.idx + len(token)))
        
    #print(list(zip(pred_tokens,pred_entities,pred_positions)))

    # slice the gold standard text using the token positions from the prediction
    gold_tokens = []  
    for pos in pred_positions:
        gold_tokens.append(gold['text'][pos[0]:pos[1]])
    
    # create a set of all unique entity categories in the gold standard
    entity_set = []
    for ent in gold['entities']:
        if ent[2] not in entity_set:
            entity_set.append(ent[2])
     
    # this creates a dico from the gold standard that has all the character positions
    # that contain a given entity type. It basically maps the areas of the
    # gold standard text where each entity is present
    gold_entity_ranges = {}
    
    for ent in entity_set:
        entpos = []
        for entity in gold['entities']:
            if entity[2] == ent:
                entpos += (list(range(entity[0], entity[1])))
        gold_entity_ranges[ent] = entpos
            
            
    # this creates a list for all of the tokens in the prediction. if the token is not in the range of
    # any entity (cf last variable, gold_entity_ranges), the loop appends the label of the token,
    # otherwise, it appends 0
    gold_entities = []
    
    for pos in pred_positions:
        isentity = False
        for label in entity_set:
            if set(range(pos[0], pos[1])) & set(gold_entity_ranges[label]):
                isentity = True
                gold_entities.append(label)
                break
        if not isentity:
            gold_entities.append(0)
            
    #print(gold_entities)
        
    # finally we can create two boolean lists that describe the gold standards and the predictions
    # on a token level for all the entities. for each label, there will be a boolean list "label_gold"
    # that has the length of tokens in the doc. for each token, the list has 1 if the entity in question is
    # present and 0 otherwise.
    # the second list 'label_pred' does the same, but this time with the prediciton. the point is to make
    # the predictions comparable for each label: if the two lists, label_gold and label_pred are identical
    # for a label, your model got all the entities right in that category. if label_gold has more, your model
    # missed some, and vice-versa
    
    results = {}
    
    for label in entity_set:
        label_gold = [1 if ent==label else 0 for ent in gold_entities]
        label_pred = [1 if ent==label else 0 for ent in pred_entities]
        
        results[label] = (label_gold, label_pred)
        
    return results


def evaluation_results(predictions, gold_data, allowed_labels, output_dir=None, cmap='viridis'): 
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    
    total_gold = []
    total_pred = []
    
    for prediction, annotation in zip(predictions, gold_data): # get all your stuff into to lists!
        
        eval_scores = evaluate_ner(prediction, annotation)
        
        for label in list(eval_scores.keys()):
            if label in allowed_labels:
                total_gold += eval_scores[label][0]
                total_pred += eval_scores[label][1]
            
    matrix = confusion_matrix(total_gold, total_pred)
    print(matrix)
    
    tn, fp, fn, tp = matrix.ravel()
    #print(tn, fp, fn, tp)
    
    print(allowed_labels)
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_score = 2 / ((recall**-1) + (precision)**-1)
    
    print(f'precision: {precision}, recall: {recall}, f-score: {f_score}')
    print('\n')
    
    cmplot = ConfusionMatrixDisplay(matrix)
    cmplot.plot()
    cmplot.ax_.set(title='  '.join([label for label in allowed_labels]))
    cmplot.im_.set_cmap(cmap)
    
    if output_dir:
        plotname = '_'.join([label for label in allowed_labels])
        cmplot.figure_.savefig(output_dir + 'cm_' + plotname + '.png', bbox_inches='tight')