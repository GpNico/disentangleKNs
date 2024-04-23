"""
    Compute TREx score for each prompt and store it.
    So when loading prompts we can filter it by score.
"""
import argparse
from transformers import PreTrainedTokenizer
from typing import Dict, Union, List, Tuple
import tqdm
import json
import numpy as np
import torch
import os

from src.models import ModelWrapper
from src.utils import pad_input_ids
from config import Config

def compute_trex_scores(
                    model: ModelWrapper, 
                    tokenizer: PreTrainedTokenizer, 
                    dataset: Union[Dict[str, Dict[str, Union[torch.Tensor, str, int]]], Dict[str, Dict[str, Dict[str, Union[torch.Tensor, str, int]]]]],
                    config: Config
                    ) -> Union[Tuple[ Dict[str, float], Dict[str, Dict[str, float]] ], Dict[str, Tuple[ Dict[str, float], Dict[str, Dict[str, float]]] ] ]:
    
    if 'type' in dataset.keys():
        # Then we're not doing multilingual
        return _compute_trex_scores(
                        model = model, 
                        tokenizer = tokenizer, 
                        dataset = dataset, 
                        config = config
                        )
    else:
        # We're doing multilingual
        scores = {}
        for lang in dataset.keys():
            
            lang_avg_scores, lang_predicate_ids_to_scores = _compute_trex_scores(
                                                                            model = model, 
                                                                            tokenizer = tokenizer, 
                                                                            dataset = dataset[lang], 
                                                                            config = config
                                                                            )
            scores[lang] = lang_avg_scores, lang_predicate_ids_to_scores
        
        return scores
    
def _compute_trex_scores(
                        model: ModelWrapper, 
                        tokenizer: PreTrainedTokenizer, 
                        dataset: Dict[str, Dict[str, Union[torch.Tensor, str, int]]],
                        config: Config
                        ) -> Tuple[ Dict[str, float], Dict[str, Dict[str, float]] ]:
    max_k = max(config.ACCURACY_RANKS)
    
    scores = {}
    for predicate_id in tqdm.tqdm(dataset.keys(), total = len(dataset)):
        if predicate_id == 'type':
            continue
        
        rela_scores = {f'P@{k}': np.zeros( next(iter(dataset[predicate_id].values()))['num_prompts'] ) for k in config.ACCURACY_RANKS}
        
        for uuid in dataset[predicate_id].keys():
            # Get template Tokens & Y 
            sentences_tok = dataset[predicate_id][uuid]['sentences_tok']    
            Y = dataset[predicate_id][uuid]['Y']
            
            # Target Tokenization
            target = tokenizer([Y]*len(sentences_tok), 
                                return_tensors = 'pt',
                                add_special_tokens=False)['input_ids'].to(model.device) # No need to pas
            
            # Pad & Create Attention Mask
            input_ids, attention_mask = pad_input_ids(sentences_tok) # The device is set in the get_prediction_logits method
            
            # Get Logits
            prediction_logits = model.get_prediction_logits(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    tokenizer= tokenizer
                                    ) # Shape [batch_size, Voc_size]
            
            
            # Compute P@k
            _, ids = torch.topk(prediction_logits, k = max_k)
            
            ids = ids.cpu()
            target = target.cpu()
            
            #print(ids)
            #print(target)
            #print(Y)
            #exit(0)
            for k in config.ACCURACY_RANKS:
                _p_k = (target[:] == ids[:,:k]).any(axis = 1).numpy().astype(float)
                rela_scores[f'P@{k}'] += _p_k
        
        # Normalize Scores        
        rela_scores = {k: v/len(dataset[predicate_id]) for k,v in rela_scores.items()}
        scores[predicate_id] = rela_scores
        
    # Write the scores
    write_trex_scores(scores, 
                      config, 
                      dataset_type = dataset['type'],
                      model_name = model.model_name)
    
    
    # Compute Final Scores
    predicate_ids_to_scores = {predicate_id: {k: v.mean() for k, v in _rela_score.items()} for predicate_id, _rela_score in scores.items()}
    
    # Avg Scores
    avg_scores = {f'P@{k}': 0 for k in config.ACCURACY_RANKS}
    for _scores in predicate_ids_to_scores.values():
        for k in avg_scores.keys():
            avg_scores[k] += _scores[k]
    avg_scores = {k: v/len(predicate_ids_to_scores) for k,v in avg_scores.items()}
        
    return avg_scores, predicate_ids_to_scores
                
def write_trex_scores(scores: Dict[str, Dict[str, np.ndarray]], 
                      config: Config,
                      dataset_type: str, 
                      model_name: str) -> None:
    """
        Write each scores in the respective prompt file.
    """
    
    if 'pararel' in dataset_type:
        # Check if multilingual or not
        if dataset_type[:2] == 'm_':
            lang = dataset_type.split('_')[-1]
            path_to_dataset = os.path.join(config.PATH_TO_MPARAREL, lang)
        else:
            path_to_dataset = config.PATH_TO_AUTOREGRESSIVE_PARAREL
        # Save P@1    
        for predicate_id, rela_scores in scores.items():
            P1 = rela_scores['P@1']
            # Open & Write file
            with open(os.path.join(path_to_dataset, f'{predicate_id}.jsonl'), 'r') as f:
                new_lines = []
                k = 0
                for line in f:
                    data = json.loads(line)
                    if f'{model_name}_P@1' in data.keys(): # This whole trick is useful think about it before deleting it!
                        if data[f'{model_name}_P@1'] < config.PROMPT_MIN_PRECISION: # Means the prompt is not in the current dataset
                            new_lines.append(data) # Store the line as is
                            continue # move to next line
                        else:
                            # Here the score should be the same
                            try:
                                assert abs(data[f'{model_name}_P@1'] - P1[k]) < 0.01
                            except:
                                print(f"Error while writing prompts scores ({predicate_id}). Scores doesn't match!\n\tOld P@1 = {np.round(data[f'{model_name}_P@1'], 3)} - New P@1 = {np.round(P1[k], 3)}")
                                raise
                    data[f'{model_name}_P@1'] = P1[k]
                    new_lines.append(data)
                    k += 1
            with open(os.path.join(path_to_dataset, f'{predicate_id}.jsonl'), 'w') as f:
                for new_line in new_lines:
                    json.dump(new_line, f)  # Write JSON data
                    f.write('\n')
    
    elif 'autoprompt' in dataset_type:
        # Check if multilingual or not
        if dataset_type[:2] == 'm_':
            lang = dataset_type.split('_')[-1]
            path_to_dataset = os.path.join(config.PATH_TO_SAVED_PROMPTS, model_name, lang)
        else:
            path_to_dataset = os.path.join(config.PATH_TO_SAVED_PROMPTS, model_name)
        # Save P@1
        for predicate_id, rela_scores in scores.items():
            P1 = rela_scores['P@1']
            # Open & Write file
            with open(os.path.join(path_to_dataset, f'{predicate_id}.jsonl'), 'r') as f:
                new_lines = []
                k = 0
                for line in f:
                    data = json.loads(line)
                    if f'P@1' in data.keys(): # This whole trick is useful think about it before deleting it!
                        if data[f'P@1'] < config.PROMPT_MIN_PRECISION: # Means the prompt is not in the current dataset
                            new_lines.append(data) # Store the line as is
                            continue # move to next line
                        else:
                            # Here the score should be the same
                            assert abs(data[f'P@1'] - P1[k]) < 0.01
                    data['P@1'] = P1[k]
                    new_lines.append(data)
                    k += 1
            with open(os.path.join(path_to_dataset, f'{predicate_id}.jsonl'), 'w') as f:
                for new_line in new_lines:
                    json.dump(new_line, f)  # Write JSON data
                    f.write('\n')
                    

def delete_model_scores(model_name: str, config: Config, multilingual: bool, autoprompt: bool) -> None:
    
    if multilingual:
        if autoprompt:
            path_to_dataset = os.path.join(config.PATH_TO_SAVED_PROMPTS, model_name)
        else:
            path_to_dataset = config.PATH_TO_MPARAREL
        langs = os.listdir(path_to_dataset) # could use config.LANGS but here we are sure
    else:
        if autoprompt:
            path_to_dataset = os.path.join(config.PATH_TO_SAVED_PROMPTS, model_name)
        else:
            path_to_dataset = config.PATH_TO_AUTOREGRESSIVE_PARAREL
        langs = ['']
    
    
    for lang in langs:
        _path_to_dataset = os.path.join(path_to_dataset, lang)
        print(f'Deleting P@1 in {_path_to_dataset} for model {model_name}.')
        
        # Predicate ids
        predicate_ids = [n[:-6] for n in os.listdir(_path_to_dataset)]
        
        # Delete P@1    
        for predicate_id in tqdm.tqdm(predicate_ids, total=len(predicate_ids)):
            # Open & Write file
            with open(os.path.join(_path_to_dataset, f'{predicate_id}.jsonl'), 'r') as f:
                new_lines = []
                for line in f:
                    data = json.loads(line)
                    if f'{model_name}_P@1' in data.keys(): # This whole trick is useful think about it before deleting it!
                        del data[f'{model_name}_P@1']
                    new_lines.append(data)
            with open(os.path.join(_path_to_dataset, f'{predicate_id}.jsonl'), 'w') as f:
                for new_line in new_lines:
                    json.dump(new_line, f)  # Write JSON data
                    f.write('\n')