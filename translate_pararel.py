
import argparse
import os
import json
import time 
from typing import Dict, List, Tuple
import wandb

import torch
from transformers import AutoProcessor, SeamlessM4TModel

SUPPORTED_LANGS = {
                 'af': 'afr',
                 'ar': 'arb',
                 'az': 'azj',
                 'be': 'bel',
                 'bg': 'bul',
                 'bn': 'ben',
                 'ca': 'cat',
                 'ceb': 'ceb',
                 'cs': 'ces',
                 'cy': 'cym',
                 'da': 'dan',
                 'de': 'deu',
                 'el': 'ell',
                 'en': 'eng',
                 'es': 'spa',
                 'et': 'est',
                 'eu': 'eus',
                 'fa': 'pes',
                 'fi': 'fin',
                 'fr': 'fra',
                 'ga': 'gle',
                 'gl': 'glg',
                 'he': 'heb',
                 'hi': 'hin',
                 'hr': 'hrv',
                 'hu': 'hun',
                 'hy': 'hye',
                 'id': 'ind',
                 'it': 'ita',
                 'ja': 'jpn',
                 'ka': 'kat',
                 'ko': 'kor',
                 'la': 'lao', # Not sure but don't care
                 'lt': 'lit', 
                 'lv': 'lvs',
                 'ms': 'mlt', # Not sure but don't care
                 'nl': 'nld',
                 'pl': 'pol',
                 'pt': 'por',
                 'ro': 'ron',
                 'ru': 'rus',
                 'sk': 'slk',
                 'sl': 'slv',
                 'sq': None,
                 'sr': 'srp',
                 'sv': 'swe',
                 'ta': 'tam',
                 'th': 'tha',
                 'tr': 'tur',
                 'uk': 'ukr',
                 'ur': 'urd',
                 'vi': 'vie',
                 'zh': 'cmn'
                }

def get_vocab(lang: str):
    mlama_path = os.path.join('data', 'mlama1.1')
    predicate_ids = [n for n in os.listdir(os.path.join(mlama_path, lang)) if n[0] == 'P' and len(n) < 6]
    ### TRAIN ###
    vocab = {}
    for predicate_id in predicate_ids:
        vocab[predicate_id] = {}
        with open(os.path.join(mlama_path, lang, predicate_id, f'train.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                X, Y, uuid = data['sub_label'], data['obj_label'], data['uuid']
                vocab[predicate_id][uuid] = (X,Y)

    return vocab 


def find_common_uuids(lang_vocab: Dict[str, Tuple[str, str]], 
                     en_vocab: Dict[str, Tuple[str, Tuple[str, str]]]) -> str:
    uuids = []
    for uuid in lang_vocab.keys():
        if uuid in en_vocab.keys():
            uuids.append(uuid)
    return uuids

def update_wandb_table(data: dict, lang: str):
    columns = ['predicate id', 'EN number of prompts' , 'number of prompts', 'translated prompts']
    table = wandb.Table(columns=columns)
    for k, v in data.items():
        table.add_data(v[0], v[1], v[2], v[3])
    wandb.log({f'Results {lang.upper()}': table})

def translate_prompts(lang: str,
                      processor,
                      model) -> None:
    auto_pararel_path = os.path.join('data', 'autoregressive_pararel')
    predicate_ids = [n[:-6] for n in os.listdir(os.path.join(auto_pararel_path))]
    mpararel_path = os.path.join('data', 'mpararel', lang)
    os.makedirs(mpararel_path, exist_ok=True)
    
    # Load vocabs
    en_vocab = get_vocab(lang = 'en')
    lang_vocab = get_vocab(lang = lang)
    
    # Wandb storing (because it sucks!)
    wandb_table = {}
    
    # Translation
    full_lang_prompts = {}
    time_beg = time.time()
    for k, predicate_id in enumerate(predicate_ids):
        # Log
        wandb.log({f'Progress (on {len(predicate_ids)})': k})
        
        print(f"Translating {predicate_id}... ({k+1}/{len(predicate_ids)})")
        # Load en prompts
        prompts = []
        with open(os.path.join(auto_pararel_path,  f'{predicate_id}.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['pattern'])
        
        # Get X & Y
        uuids = find_common_uuids(
                            lang_vocab[predicate_id], 
                            en_vocab[predicate_id])
        if len(uuids) == 0:
            raise Exception(f"\tNo common uuid found for {predicate_id}.")
        
        # Translate prompts
        MAX_VOTING_UUIDS = 30
        lang_prompts = []
        for prompt in prompts:
            _prompts_candidates = {}
            i = 0
            for uuid in uuids:
                X_en, Y_en = en_vocab[predicate_id][uuid]
                X_lang, Y_lang = lang_vocab[predicate_id][uuid]
                lang_prompt, trans_flag = translate_one_prompt(
                                            tgt_lang = lang,
                                            prompt = prompt,
                                            X_en = X_en,
                                            Y_en = Y_en,
                                            X_lang = X_lang,
                                            Y_lang = Y_lang,
                                            processor = processor,
                                            model = model
                                            )
                if trans_flag:
                    if lang_prompt in _prompts_candidates.keys():
                        _prompts_candidates[lang_prompt] += 1
                    else:
                        _prompts_candidates[lang_prompt] = 1
                    i += 1
                    display_message = f"\tDone {i}/{MAX_VOTING_UUIDS}."+40*" "
                    print(display_message, end='\r', flush=True)
                    if i >= MAX_VOTING_UUIDS:
                        break
                else:
                    print('\tuuid failed. Trying next one.', end='\r', flush=True)
            
            if len(_prompts_candidates) > 0:
                # Need to vote
                _prompts_candidates = {k: v for k, v in sorted(_prompts_candidates.items(), key=lambda item: -item[1])}
                sorted_prompts = list(_prompts_candidates.keys())
                # Here we would need to:
                #    (i) Penalize prompts that have already been found    
                #    (ii) Get rid of non autoregressive prompts
                idx = 0
                prompt_saved_flag = False
                while idx < min(4, len(sorted_prompts)-1): # We only go up to 4th one
                    if (sorted_prompts[idx] in lang_prompts) \
                        or not(sorted_prompts[idx][-5:] == '[Y] .' or sorted_prompts[idx][-4:] == '[Y].' or  sorted_prompts[idx][-3:] == '[Y]' or sorted_prompts[idx][-5:] == '[Y] ?' or sorted_prompts[idx][-4:] == '[Y]?'):    
                        idx += 1
                    else:
                        prompt_saved_flag = True # we saved the prompt!
                        # Keep best prompt
                        lang_prompts.append(sorted_prompts[idx])
                        print(f"\tEN: {prompt} - {lang.upper()}: {sorted_prompts[idx]}")
                        # storing for wandb table
                        if predicate_id in wandb_table.keys():
                            v = wandb_table[predicate_id]
                            wandb_table[predicate_id] = (v[0], v[1], v[2]+1, v[3] + '\n' + sorted_prompts[idx])
                        else:
                            wandb_table[predicate_id] = (predicate_id, len(prompts), 1, sorted_prompts[idx])
                        update_wandb_table(data = wandb_table, lang = lang)
                        break
                    
                if not(prompt_saved_flag):
                    print("\tFail to save the prompt as it was either not autoregressive compatible either already saved.")
            else:
                raise Exception(f"\tNo valid uuid found for prompt {prompt}.")
        
        full_lang_prompts[predicate_id] = lang_prompts
        
        # Save it
        with open(os.path.join(mpararel_path, f'{predicate_id}.jsonl'), 'w') as f:
            for prompt in lang_prompts:
                json.dump({'pattern': prompt}, f)
                f.write('\n')
                
        # Time estimate
        time_tot = time.time() - time_beg
        _avg_time = time_tot/(k+1)
        _remaining_time = (len(predicate_ids) - (k+1)) * _avg_time
        print(f"Time ellapsed: {int(time_tot//3600)}h {int((time_tot%3600)//60)}m - Estimated Remaining Time: {int(_remaining_time//3600)}h {int((_remaining_time%3600)//60)}m")


def translate_one_prompt(tgt_lang: str, 
                         prompt: str, 
                         X_en: str,
                         Y_en: str,
                         X_lang: str,
                         Y_lang: str,
                         processor, 
                         model) -> str:
    
    
    # Create src prompt
    sentence = prompt.replace('[X]', X_en).replace('[Y]', Y_en)
    # Tokenize
    text_inputs = processor(
                        text = sentence, 
                        src_lang="eng", 
                        return_tensors="pt"
                        ).to(device)
    # Translate
    output_tokens = model.generate(
                        **text_inputs, 
                        tgt_lang=SUPPORTED_LANGS[tgt_lang],
                        generate_speech=False
                        )
    translated_text = processor.decode(
                        output_tokens[0].tolist()[0], 
                        skip_special_tokens=True
                        )
    
    # Replace X & Y
    if X_lang in translated_text and Y_lang in translated_text:
        translated_prompt = translated_text.replace(X_lang, '[X]').replace(Y_lang, '[Y]')
        if '[X]' in translated_prompt and '[Y]' in translated_prompt:
            return translated_prompt, True
        else:
            return None, False
    else:
        return None, False 
    
    
if __name__ == '__main__':
    
    # Argparse
    parser = argparse.ArgumentParser()
    
    # Model & Data
    parser.add_argument('--lang', 
                        type=str, 
                        help="lang to transflate (mLAMA format).")
    args = parser.parse_args()
    
    # Init Model
    model_name = 'hf-seamless-m4t-large'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Found device: {device}')
    processor = AutoProcessor.from_pretrained(f"facebook/{model_name}", use_fast = False)
    model = SeamlessM4TModel.from_pretrained(f"facebook/{model_name}", torch_dtype = torch.float16,).to(device)
    
    # logging
    wandb.init(
            project="mParaRel", 
            name=args.lang, 
            mode="offline"
            )
    
    translate_prompts(lang=args.lang, processor = processor, model = model)