import os
from config import Config
from typing import List
import json
import wandb

def run_autoprompt(
            config: Config, 
            model_name: str,
            dataset: str
            ) -> None:
    
    
    # Create Autoprompt Template based on model
    if 'bert' in model_name: # It include mBERT
        template = '[CLS] {sub_label} ' + '[T] '*config.N_TRIGGER_TOKENS + '[P] [SEP]'
    elif 'gpt2' in model_name:
        template = '{sub_label} ' + '[T] '*config.N_TRIGGER_TOKENS + '[P]'
    elif 'opt' in model_name:
        template = '</s> {sub_label} ' + '[T] '*config.N_TRIGGER_TOKENS + '[P]'
    elif 'xlm' in model_name:
        template = '<s> {sub_label} ' + '[T] '*config.N_TRIGGER_TOKENS + '[P] </s>'
    elif 'Llama' in model_name:
        template = '<s> {sub_label}' + '[T]'*config.N_TRIGGER_TOKENS + '[P]'
    
    
    # Create csv folder
    jsonl_path = os.path.join(
                    config.PATH_TO_SAVED_PROMPTS, 
                    model_name
                    )
    
    
    # Langs to compute
    if dataset == 'mlama':
        langs = config.LANGS
    else:
        langs = ['en']
    
    for lang in langs:
        # Data path
        if dataset == 'trex':
            data_path = config.PATH_TO_TREX
            lang_specific_path = ''
        elif dataset == 'mlama':
            data_path = os.path.join(config.PATH_TO_MLAMA, lang)
            lang_specific_path = lang
        
        # Get TREx relation names
        predicate_ids = [n for n in os.listdir(data_path) if n[0] == 'P' and len(n) < 6]
        
        # Get current last seed
        if os.path.exists(os.path.join(jsonl_path, lang_specific_path, f'{predicate_ids[0]}.jsonl')):
            with open(os.path.join(jsonl_path, lang_specific_path, f'{predicate_ids[0]}.jsonl'), 'r') as f:
                for line in f:
                    data = json.loads(line)
                last_seed = data['seed']+1
            print(f"Found existing prompts. Starting from seed {last_seed}.")
        else:
            last_seed = 0
            
        os.makedirs(os.path.join(jsonl_path, lang_specific_path), exist_ok=True)
        
        # Compute Autoprompt for each seeds
        for seed in range(last_seed, last_seed + config.N_SEEDS):
            
            if config.WANDB:
                ### WANDB ###
                wandb.init(
                        project=model_name, 
                        name=f"Autoprompt - Seed {seed}", 
                        mode="offline" # /!\
                        )
                wandb_flag = '--wandb'
            else:
                wandb_flag = ''
            
            # Run Autoprompt
            for idx, predicate_id in enumerate(predicate_ids):
                # Create seed model name
                _jsonl_path = os.path.join(
                                jsonl_path, 
                                lang_specific_path,
                                f'{predicate_id}.jsonl'
                                )
                if config.WANDB:
                    wandb.log({f"Predicate Ids Completion (max. {len(predicate_ids)})": idx})
                    
                print(os.path.join(data_path, predicate_id))
                print(predicate_ids)
                exit(0)
                # Command Line    
                command_line = f'python -m autoprompt.autoprompt.create_trigger \
                                --train {os.path.join(data_path, predicate_id)}/train.jsonl \
                                --dev {os.path.join(data_path, predicate_id)}/dev.jsonl \
                                --template "{template}" \
                                --num-cand 10 \
                                --accumulation-steps 1 \
                                --model-name {model_name} \
                                --seed {seed} \
                                --bsz {config.AUTOPROMPT_BATCH_SIZE} \
                                --eval-size 56 \
                                --iters {config.AUTOPROMPT_ITERS} \
                                --label-field "obj_label" \
                                --tokenize-labels \
                                --filter \
                                --print-lama \
                                --save_best_tokens \
                                --path_to_saved_prompts {_jsonl_path} \
                                --data_id {predicate_id} \
                                --seed {seed} \
                                {wandb_flag}'

                os.system(command_line)
                
            if config.WANDB:
                wandb.finish()

    

    