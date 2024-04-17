"""
Dataset of prompts based on random relations (same relations as above though)
    - Should we get paraphrases by doing auto-prompt training with the objective to output similar probabilities
        (i) (X, Answer from Random Prompt but constrained to Y)
        (ii) (X, Shuffle Y) shuffled within a relation  P1
             Note to self: Xs are coherent, Ys are coherent, X and Y related, (X,Y)s are not coherent
        (iii) (X, Y’) shuffled, with X coming from one relation, Y’ coming from another relation P2
              Note to self: Xs are coherent, Ys are coherent, X and Y not related, (X,Y)s are not coherent
        (iv) (X, Y’) sample X and Y’ from all the X’s and Y’s from all relations P3
             Note to self: Xs are not coherent, Ys are not coherent, X and Y not related, (X,Y)s are not coherent)
        (v) (X,Y) sample X and Y from all possible tokens

Currently doing: (ii), (iii) & (iv)

Questions:
    1. Is Autoprompt able to map any X to any Y?
    2. If yes which Knowledge Neuron is accessed while doing so?

Expected template of Autoprompt:

{"index":509,
 "sub_uri":"Q3300448",
 "obj_uri":"Q96",
 "obj_label":"Mexico",
 "sub_label":"Ozumba",
 "lineid":510,
 "uuid":"5a2d6d86-4086-4d6f-b64c-e9fe3afdd355"}

BUT: "index" optionnal, "lineid" optionnal
AND: call, "sub_uri" and "obj_uri" useless too.

One line per uuid (or training example): at least two files, train.jsonl & dev.jsonl

"""

import argparse
import json
import jsonlines
import numpy as np
import os
import random
from typing import List, Dict


TRAIN_DEV_SPLIT = 0.8

def shuffle2(rela_dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
        Xs and Ys same relation BUT Ys is shuffled.
    """
    Ys = []
    for elem in rela_dataset:
       Ys.append(elem['obj_label'])
    random.shuffle(Ys)
    
    for k in range(len(rela_dataset)):
        rela_dataset[k]['obj_label'] = Ys[k]

    return rela_dataset

def shuffle3(dataset: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    """
        Xs from one relation, Ys from another.
    """
    # Association of LAMA relations based on their subjects
    rela2rela = [['P17', 'P31'],
                ['P19', 'P39'],
                ['P20', 'P101'], 
                ['P27', 'P106'],
                ['P30', 'P138'],
                ['P36', 'P136'],
                ['P37', 'P176'],
                ['P47', 'P140'],
                ['P103', 'P178'],
                ['P108', 'P131'],
                ['P127', 'P1303'],
                ['P159', 'P527'],
                ['P190', 'P413'],
                ['P264', 'P364'],
                ['P276', 'P279'],
                ['P361', 'P407'],
                ['P449', 'P495'],
                ['P463', 'P740'],
                ['P530', 'P1001'], # WTF did I do that Country vs County??
                ['P937', 'P1412']]
    excluded = 'P1376' # because odd number of relations
    
    rela2Ysrela = {}
    for rela1, rela2 in rela2rela:
        rela2Ysrela[rela1] = rela2
        rela2Ysrela[rela2] = rela1
        
    new_dataset = {}
    for predicate_id in rela2Ysrela.keys():
        predicate_id_lst = []
        Ys_predicate_id = rela2Ysrela[predicate_id]
        length = min(len(dataset[predicate_id]), len(dataset[Ys_predicate_id])) 
        perm = np.arange(len(dataset[Ys_predicate_id])) # Shuffle Ys  
        np.random.shuffle(perm)
        for i in range(length):
            elem = dataset[predicate_id][i].copy()
            elem['obj_label'] = dataset[Ys_predicate_id][perm[i]]['obj_label']
            predicate_id_lst.append(elem)
        new_dataset[predicate_id] = predicate_id_lst
    
    new_dataset[excluded] = None

    return new_dataset


def get_uuid_for_trex():
    """
        So for some reasons we need the T-REX Dataset from Autoprompt
        to use the dev and train set. But this dataset doesn't use uuid.
        
        So in this function we'll associate uuid to this dataset using another
        LAMA dataset... 
        
        T-REX is available here:
        https://drive.google.com/drive/folders/1EBVEGSQI2nf41hG2T66i17lQPDQroRAN 
    
    """
    ### GET T-REX WITH UUID ###
    # Get predicate ID
    predicate_ids = [n[:-6] for n in os.listdir(os.path.join('data', 'pararel', 'trex_lms_vocab'))]
    
    # Load vocab
    uuid_dataset = {}
    for predicate_id in predicate_ids:
        rela_dataset = {}
        with open(os.path.join('data', 'pararel', 'trex_lms_vocab', f'{predicate_id}.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                rela_dataset[(data['sub_label'],data['obj_label'])] = data['uuid']
        uuid_dataset[predicate_id] = rela_dataset
        
    ### WRITE UUID IN T-REX ###
    key_errors, num_keys = 0, 0
    splits = ['train', 'dev', 'test']
    for predicate_id in predicate_ids:
        for split in splits:
            with jsonlines.open(os.path.join('data', 'trex', 'vanilla', f'{predicate_id}' , f'{split}.jsonl'), 'r') as reader, jsonlines.open(os.path.join('data', 'trex', 'vanilla', f'{predicate_id}' , f'{split}_test.jsonl'), 'w') as writer:
                for obj in reader:
                    # Assuming 'uuid_lst' has enough values for each JSON object
                    try:
                        uuid_value = uuid_dataset[predicate_id][(obj['sub_label'],obj['obj_label'])]
                    except:
                        key_errors += 1
                        num_keys += 1
                        continue
                        
                    num_keys += 1
                    # Adding 'uuid' key to the JSON object
                    obj['uuid'] = uuid_value
                    
                    # Writing the updated JSON object to the output file
                    writer.write(obj)
    print(f"Total of {key_errors}/{num_keys} KeyError.")


if __name__ == '__main__':
    
    ### Get Params ###
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffling_method', 
                        type=int, 
                        default=2)
    args = parser.parse_args()
    
    ### Open T-Rex vocab ###
    
    # Get predicate ID
    predicate_ids = [n for n in os.listdir(os.path.join('data', 'trex', 'vanilla'))]
    
    # splits
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        # Load vocab
        dataset = {}
        for predicate_id in predicate_ids:
            rela_dataset = []
            with open(os.path.join('data', 'trex', 'vanilla', f'{predicate_id}', f'{split}.jsonl'), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    rela_dataset.append({'sub_label': data['sub_label'], 
                                         'obj_label': data['obj_label']})
            dataset[predicate_id] = rela_dataset            
            
        ### Write Dataset ###
        if args.shuffling_method == 3:
            dataset = shuffle3(dataset)
        
        for predicate_id in predicate_ids:
            # Create directory
            rela_path = os.path.join(
                            'data',
                            'trex', 
                            'shuffle',
                            str(args.shuffling_method), 
                            predicate_id
                            )
            os.makedirs(rela_path, exist_ok=True)
            
            # Shuffle
            if args.shuffling_method == 2:
                rela_dataset = shuffle2(dataset[predicate_id])
            else:
                rela_dataset = dataset[predicate_id]
                if rela_dataset is None:
                    continue
            
            # Save file
            with open(os.path.join(rela_path, f'{split}.jsonl'), "w") as jsonl_file:
                for item in rela_dataset:
                    jsonl_file.write(json.dumps(item) + "\n")
            
        