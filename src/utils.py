
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from transformers import AutoTokenizer

from config import Config
    
def load_config(debug: bool) -> Config:
    config = Config(debug = debug)
    return config

def load_tokenizer(model_name: str):
    
    if 'opt' in model_name:
        return AutoTokenizer.from_pretrained(f'facebook/{model_name}',
                                             add_prefix_space = True,) # I think I should) # Not sure why
    else:
        return AutoTokenizer.from_pretrained(model_name)

def should_lower(model_name: str) -> bool:
    """
    In fact very important function: for cased-sensitive models like OPT "Tokyo" is 1 token but "tokyo" is 3!
    """
    if model_name in ['bert-base-uncased',
                      'bert-large-uncased']:
        return True
    elif model_name in ['opt-350m', 'opt-6.7b']:
        return False
    else:
        raise Exception("Don't forget to put your model in the should_lower function :'(")

def is_autoregressive(model_name):
    if 'bert' in model_name:
        return False
    if 'gpt' in model_name:
        return True
    if 'opt' in model_name:
        return True
    
def get_model_intermediate_layer(model: nn.Module, 
                                 model_name: str,
                                 layer_num: int) -> nn.Module:
    if 'bert' in model_name:
        return model.bert.encoder.layer[layer_num].intermediate
    elif 'opt' in model_name:
        return model.model.decoder.layers[layer_num].fc1
    else:
        raise Exception("Don't forget to put your model in the get_model_intermediate_layer function :'(")
    
def get_intermediate_dim(model: nn.Module,
                         model_name: str) -> int:
    if 'bert' in model_name:
        return model.bert.encoder.layer[0].intermediate.dense.out_features
    elif 'opt' in model_name:
        return model.model.decoder.layers[0].fc1.out_features
    else:
        raise Exception("Don't forget to put your model in the get_model_intermediate_layer function :'(")
    
def pad_input_ids(tokens_lst: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create attention 
    attention_lst = []
    for toks in tokens_lst:
        attention_lst.append(torch.tensor([1]*len(toks)))
    # pad
    input_ids = torch.nn.utils.rnn.pad_sequence(
                                    tokens_lst, 
                                    batch_first = True
                                    ).to(torch.int32)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
                                    attention_lst, 
                                    batch_first = True
                                    ).to(torch.int8)
    
    return input_ids, attention_mask
    

def compute_variation(
                probs_before: list, 
                probs_after: list
                ) -> float:
  # Scores
  prob_changes = []
  failed_count = 0
  for uuid in probs_before.keys():
    fail_flag = False
    _prob_changes_uuid = []
    for k in range(len(probs_before[uuid])):
        try:
            prob_change = (probs_after[uuid][k] - probs_before[uuid][k]) / probs_before[uuid][k]
            _prob_changes_uuid.append(prob_change)
        except:
            fail_flag = True
            failed_count += 1
            break
    if fail_flag:
        continue
    prob_changes.append( sum(_prob_changes_uuid)/len(_prob_changes_uuid) )

  print(f'\tFailed Counts: {failed_count}/{len(probs_before.keys())}')
  
  return sum(prob_changes)/len(prob_changes)

def filter_tensor_by_another(original_tensor: torch.Tensor, filter_tensor: torch.Tensor) -> torch.Tensor:
    # Create a dictionary to store the indices of elements in the filter tensor
    filter_indices = {value: idx for idx, value in enumerate(filter_tensor.tolist())}

    # Use a list comprehension to filter elements from the original tensor
    filtered_tensor = [value for value in original_tensor.tolist() if value in filter_indices]

    # Convert the filtered list back to a tensor
    result_tensor = torch.tensor(filtered_tensor, dtype=original_tensor.dtype)

    return result_tensor

def tensors_intersection_size(tensor1: torch.Tensor, tensor2: torch.Tensor) -> int:
    """
        Shape of tensor1 and tensor2 are expected to be [n] & [m]
    
    """
    return len(set(tensor1.tolist()).intersection(set(tensor2.tolist())))

def find_closest_elem(lst: List[float], elem: float) -> float:
    min_d = np.infty
    for e in lst:
        d = abs(e - elem)
        if d < min_d:
            argmin = e
            min_d = d
    return argmin

def get_run_name(args) -> str:
    """
        Get the name of wandb run name
        based on argparse.
    """
    
    if args.filter_prompts:
        if args.autoprompt:
            return 'TREx Eval - Autoprompt'
        else:
            return 'TREX Eval'
    if args.kns_compute:
        if args.autoprompt:
            return 'KNs Computation - Autoprompt'
        else:
            return 'KNs Computation'
    if args.kns_exps:
        if args.autoprompt:
            if args.equal:
                return 'KNs Experiments - Autoprompt (equal)'
            else:
                return 'KNs Experiments - Autoprompt (all)'
        else:
            if args.equal:
                return 'KNs Experiments (equal)'
            else:
                return 'KNs Experiments (all)'
    if args.kns_eval:
        if args.autoprompt:
            return 'KNs Evaluation - Autoprompt'
        else:
            return 'KNs Evaluation'
    if args.kns_eval:
        return 'KNs Analysis'
    
    