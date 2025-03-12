
import argparse
import torch

import src.utils as utils
from src.models import ModelWrapper
from src.knowledge_neurons import KnowledgeNeurons
from src.data import load_trex_by_uuid

if __name__ ==  '__main__':
    ### Argparse ###
    
    parser = argparse.ArgumentParser()
    
    # Model & Data
    parser.add_argument('--model_name', 
                        type=str, 
                        default='bert-base-uncased',
                        help="HuggingFace model's name.")
    parser.add_argument('--dataset', 
                        type=str, 
                        default='trex',
                        help="For now only trex and mlama are supported.")
    parser.add_argument('--autoprompt',
                        action="store_true",
                        help="Load the autoprompt version of the datasets.")
    parser.add_argument('--predicate_id',
                        type=str)
    parser.add_argument('--lang', 
                        type=str)
    
    parser.add_argument('--add_prefix',
                        action="store_true",
                        help="Add a prefix 'Answer in one word:' to ParaRel prompts.")

    args = parser.parse_args()
    
    ### LOAD EVERYTHING ###
    
    if args.dataset == 'trex':
        multilingual = False
        split = 'test'
    elif args.dataset == 'mlama':
        multilingual = True
        split = 'dev'
    else:
        raise Exception(f'Dataset {args.dataset} is unsupported.')
    
    config = utils.load_config(debug = False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokenizer = utils.load_tokenizer(args.model_name)
    
    model = ModelWrapper(
                    model_name = args.model_name,
                    device = device
                    )
    
    dataset = load_trex_by_uuid(
                                config = config,
                                model_name = args.model_name,
                                tokenizer = tokenizer,
                                autoprompt = args.autoprompt,
                                lower = utils.should_lower(args.model_name),
                                split = split,
                                autoregressive = utils.is_autoregressive(args.model_name),
                                multilingual = multilingual,
                                add_prefix = args.add_prefix
                                )
    
    ### CREATE KNs OBJECT ###
    if args.lang:
        dataset = dataset[args.lang]
    dataset_type = dataset['type']
        
    kn = KnowledgeNeurons(
                model = model.model,
                tokenizer = tokenizer,
                data = dataset,
                dataset_type = dataset_type,
                model_name = args.model_name,
                device = device,
                config = config
            )
    
    ### COMPUTE ###
    
    kn._compute_kns_one_predicate_id(predicate_id=args.predicate_id)