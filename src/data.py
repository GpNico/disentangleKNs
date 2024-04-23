

import json
import os
import torch
from pathlib import Path
import pandas as pd
from transformers import PreTrainedTokenizer
from typing import Dict, List, Union

from config import Config

    
def load_trex_by_uuid(
                    config: Config,
                    model_name: str,
                    tokenizer: PreTrainedTokenizer,
                    autoprompt: bool = False,
                    lower: bool = False,
                    split: str = 'test',
                    autoregressive: bool = False,
                    multilingual: bool = False,
                    add_prefix: bool = False
                    ) -> Union[Dict[str, Dict[str, Union[torch.Tensor, str, int]]], Dict[str, Dict[str, Dict[str, Union[torch.Tensor, str, int]]]]]:
    
    
    #
    if autoprompt:
        print("Loading Autoprompt...")
    else:
        print("Loading ParaRel...")
    
    # Deal With MultiLingual
    if multilingual:
        print("Loading and Tokenizing mLAMA...")
        assert split in ['dev', 'train']
        langs = config.LANGS
        triplets_path = config.PATH_TO_MLAMA
    else:
        print("Loading and Tokenizing TREx...")
        assert split in ['dev', 'train', 'test']
        langs = [''] # Little Trick, it won't impact path
        triplets_path = config.PATH_TO_TREX
    
    # Trick to get special tokens
    toks = tokenizer('', add_special_tokens=True).input_ids
    if len(toks) == 2:
        bos_tok_id, eos_tok_id = toks[0], toks[1] # e.g. BERT
    elif len(toks) == 1:
        bos_tok_id, eos_tok_id = toks[0], None # e.g. OPT
    else:
        bos_tok_id, eos_tok_id = None, None # e.g. GPT2
    
    # Get relation predicates ids
    predicate_ids = list(
                        set(
                            [n for n in os.listdir(os.path.join(config.PATH_TO_TREX))]
                            ).intersection(
                                    set(
                                        [n[:-6] for n in os.listdir(os.path.join(config.PATH_TO_AUTOREGRESSIVE_PARAREL))]
                                        )
                                    )
                        )

    # Load Data
    full_dataset = {}
    for lang in langs:
        
        if multilingual:
            print(f'Loading {lang}...')
        
        dataset = {}
        warning_flag = False # If we couldn't filter prompts
        
        for predicate_id in predicate_ids:
            
            # Load Prompts
            if autoprompt:
                # /!\ Autoprompt need for the sake of simplicity, to be tokenized here to avoid possible mistake /!\
                autoprompts_tok = []
                with open(os.path.join(config.PATH_TO_SAVED_PROMPTS, model_name, lang, f'{predicate_id}.jsonl'), 'r') as f:
                    for k, line in enumerate(f):
                        data = json.loads(line)
                        if 'P@1' in data.keys():
                            if data['P@1'] < config.PROMPT_MIN_PRECISION:
                                continue
                        else:
                            print(f"\tWARNING: {predicate_id} doesn't have P@1 scores written.")
                            warning_flag = True
                        autoprompts_tok.append(data['ids'])
                        assert data['seed'] == k
                        ### IF WE ARE TO REMOVE PROMPTS BASED ON TREX score DO IT HERE!
                num_prompts = len(autoprompts_tok)
            else:
                prompts = []
                if multilingual:
                    prompts_path = os.path.join(config.PATH_TO_MPARAREL, lang)
                else:
                    prompts_path = config.PATH_TO_AUTOREGRESSIVE_PARAREL
                with open(os.path.join(prompts_path, f'{predicate_id}.jsonl'), 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        if f'{model_name}_P@1' in data.keys():
                            if data[f'{model_name}_P@1'] < config.PROMPT_MIN_PRECISION:
                                continue
                        else:
                            print(f"\tWARNING: {predicate_id} doesn't have P@1 scores written.")
                            warning_flag = True
                        _prompt = data['pattern']
                        if not(multilingual):
                            if add_prefix:
                                _prompt = "Answer in only one word: " + _prompt
                        else:
                            raise Exception("Adding Prefix is not supported for multilingual.")
                        prompts.append(_prompt)
                num_prompts = len(prompts)

            # Load Xs, Ys
            trex_vocab = []
            with open(os.path.join(triplets_path, lang,  f'{predicate_id}', f'{split}.jsonl'), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if split == 'test' or (split == 'dev' and multilingual):
                        trex_vocab.append((data['sub_label'], data['obj_label'], data['uuid']))
                    else:
                        trex_vocab.append((data['sub_label'], data['obj_label'], None))
                        
            # Oraganize and Process Data
            if num_prompts < config.MIN_N_PROMPTS:
                print(f"Relation {predicate_id} skipped. Not enough prompts.")
                continue
            
            dataset_by_uuid = {}
            
            for X, Y, uuid in trex_vocab:
                
                if config.SINGLE_TOKEN_TARGET:
                    Y_ids = tokenizer(Y, 
                                      add_special_tokens=False).input_ids
                    if len(Y_ids)>1:
                        continue
                    
                if lower:
                    X = X.lower()
                    Y = Y.lower()
                    
                dataset_by_uuid[uuid] = {'Y': Y,
                                        'X': X,
                                        'sentences_tok': [], 
                                        'num_prompts': num_prompts}
                for i in range(num_prompts):
                    if autoprompt:
                        X_ids = tokenizer(X, 
                                        add_special_tokens=False).input_ids
                        _autoprompt_tok = X_ids + autoprompts_tok[i]
                        
                        if not(autoregressive):
                            _autoprompt_tok += [tokenizer.mask_token_id]
                            if eos_tok_id:
                                _autoprompt_tok += [eos_tok_id]
                        
                        if bos_tok_id:
                            _autoprompt_tok = [bos_tok_id] + _autoprompt_tok
                            
                        # /!\ No dot here I removed them from the Autoprompt Template /!\
                            
                        dataset_by_uuid[uuid]['sentences_tok'].append(torch.tensor(_autoprompt_tok))
                        
                    else: 
                        _prompt = prompts[i]
                        
                        if autoregressive:
                            assert _prompt[-5:] == '[Y] .' or _prompt[-4:] == '[Y].' or  _prompt[-3:] == '[Y]' or _prompt[-4:] == '[Y]?' # Keeps only the prompt that end with [Y] for autoregressive models 
                            if _prompt[-5:] == '[Y] .':
                                _prompt = _prompt[:-6] # get rid of the last space /!\
                            elif _prompt[-4:] == '[Y].' or _prompt[-4:] == '[Y]?':
                                _prompt = _prompt[:-5] # get rid of the last space /!\
                            elif _prompt[-3:] == '[Y]':
                                _prompt = _prompt[:-4] # get rid of the last space /!\
                            
                        if lower:
                            sentence = _prompt.replace('[X]', X).lower()
                            if not(autoregressive):
                                sentence = sentence.replace('[y]', '[MASK]') # /!\
                        else:
                            sentence = _prompt.replace('[X]', X)
                            if not(autoregressive):
                                sentence = sentence.replace('[Y]', '[MASK]')
                                
                        _prompt_tok = tokenizer(sentence, add_special_tokens=True).input_ids
                        
                        dataset_by_uuid[uuid]['sentences_tok'].append(torch.tensor(_prompt_tok))
                      
            # Store Data
            print(f"\t{predicate_id}: {len(dataset_by_uuid)} uuids; {num_prompts} prompts.")
            dataset[predicate_id] = dataset_by_uuid
            
        # Warning
        if warning_flag:
            print("WARNING: Impossible to filter some prompts based on P@1. Prompts without a score have been loaded.")
            
        # kwargs  
        if multilingual: 
            if autoprompt:
                dataset['type'] = f'm_autoprompt_{lang}'
            else:
                dataset['type'] = f'm_pararel_{lang}'
        else:
            if autoprompt:
                dataset['type'] = 'autoprompt'
            else:
                dataset['type'] = 'pararel'
            
            return dataset # If not multilingual we stop here!
        
        # If multilingual store the lang dataset and pursue
        full_dataset[lang] = dataset

    return full_dataset
                
