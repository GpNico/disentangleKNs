
import argparse
import os
import pickle
import torch
import tqdm

from src.data import load_trex_by_uuid
from src.models import ModelWrapper
from src.knowledge_neurons import KnowledgeNeurons
import src.utils as utils
import src.plots as plots
from src.filter_prompt import compute_trex_scores, delete_model_scores
from autoprompt.compute_trex_prompts import run_autoprompt

if __name__ == '__main__':
    
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
    
    # TREx & mLAMA
    parser.add_argument('--filter_prompts',
                        action="store_true",
                        help="Compute TREx P@k for each template & filter them based on PROMPT_MIN_PRECISION.")
    parser.add_argument('--delete',
                        action="store_true",
                        help = 'Delete scores written in tha datasets.')
    
    # ToBeTested
    parser.add_argument('--add_prefix',
                        action="store_true",
                        help="Add a prefix 'Answer in one word:' to ParaRel prompts.")
    
    # Knowledge Neurons
    parser.add_argument('--kns_compute',
                        action="store_true",
                        help="Compute knowledge neurons.")
    parser.add_argument('--kns_eval',
                        action="store_true",
                        help="Compute knowledge neurons surgery.")
    parser.add_argument('--kns_unmatch',
                        action="store_true",
                        help="If set to True use KNs computed on the other dataset for kns_eval.")
    parser.add_argument('--kns_overlap',
                        action="store_true",
                        help="Compute knowledge neurons overlap between ParaRel & Autoprompt.")
    parser.add_argument('--kns_analysis',
                        action="store_true",
                        help="Compute knowledge neurons analysis.")
    parser.add_argument('--kns_exps',
                        action="store_true",
                        help="Compute knowledge neurons experiments.")
    parser.add_argument('--equal',
                        action="store_true",
                        help="When computing knowledge neurons experiments, use the 'equal' parameters. Otherwise 'all'.")
    
    # Autoprompt
    parser.add_argument('--run_autoprompt',
                        action="store_true",
                        help="Run Autoprompt.")

    
    # For fast run
    parser.add_argument('--debug',
                        action="store_true")
    args = parser.parse_args()
    
    
    ### LOAD CONFIG ###
    
    config = utils.load_config(debug = args.debug)
    
    
    
    ### RUN AUTOPROMPT ###
    
    if args.run_autoprompt:
        run_autoprompt(
            config = config,
            model_name = args.model_name,
            dataset = args.dataset
        )
        # For now end here
        exit(0)
    
    
    ### DEVICE, MODEL & TOKENIZER ###
    
    if (args.filter_prompts and not(args.delete)) or args.kns_compute or args.kns_eval or args.kns_exps:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        tokenizer = utils.load_tokenizer(args.model_name)
        
        model = ModelWrapper(
                        model_name = args.model_name,
                        device = device
                        )
    
    ### Load Dataset(s) ###
    
    if args.dataset == 'trex':
        multilingual = False
        split = 'test'
    elif args.dataset == 'mlama':
        multilingual = True
        split = 'dev'
    else:
        raise Exception(f'Dataset {args.dataset} is unsupported.')
    
    if args.filter_prompts or args.kns_compute or args.kns_eval or args.kns_exps:
        if args.delete:
            delete_model_scores(
                model_name = args.model_name, 
                config = config,
                multilingual = multilingual,
                autoprompt = args.autoprompt
                )
            exit(0)
        else:
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
    else:
        dataset = None
    
    
    
    ### WANDB ###
    if config.WANDB:
        import wandb
        wandb.init(
                project=args.model_name, 
                name=utils.get_run_name(args), 
                mode="offline" # /!\
                )
    
    ### Filter Prompts Based on TREx P@1 ###
    
    if args.filter_prompts:
        trex_scores = compute_trex_scores(
                        model = model,
                        tokenizer = tokenizer,
                        dataset = dataset,
                        config = config
                    )

        if multilingual:
            dataset_type = dataset[config.LANGS[0]]['type']
        else:
            dataset_type = dataset['type']
        
        plots.plot_trex_scores(scores = trex_scores,
                               model_name=args.model_name,
                               dataset_type=dataset_type,
                               wandb_flag = config.WANDB)
        
        if config.WANDB:
            wandb.finish()
    
    
    ### Knowledge Neurons ###
    
    # /!\ WHEN ADDING A MODEL /!\
        # check:
        # In config.py: MODELS_TO_PLOT and MULTILINGUAL_MODELS_TO_PLOT
        # In models.py: __init__ and get_prediction_logits
        # In autoprompt: 
        #       compute_trex_prompts.py, 
        #       create_trigger.py 
        #           PredictWrapper, 
        #           load_pretrained, 
        #           get_embeddings, 
        #           isupper, 
        #           initial trigger part (~l.290),
        #           print lama part (~l.430, 470, 515, 527)
        #       utils.py
        #           TriggerTemplatizer
        #           add_task_specific_tokens
        # In utils.py:
        #       load_tokenizer
        #       should_lower
        #       is_autoregressive
        #       get_model_intermediate_layer
        #       get_intermediate_dim
        # In knowledge_neurons: model_layers_num
    
    ### NEED TO ADD MULTILINGUAL MODEL ###
    # mBERT (Done!)
    # XLM-R
    # mT5
    # XLM
    # mBART
    # mT0
    
    ### We NEED big models ###
    # LLAMA-2-7B
    # OPT-6.7b (13b?)
    # GPT-J
    
    if multilingual:
        
        if args.kns_compute or args.kns_eval:
            for lang in dataset.keys():
                print(f"###### COMPUTING {lang.upper()} ######\n")
                
                # Initialize KnowledgeNeurons Object
                kn = KnowledgeNeurons(
                    model = model.model,
                    tokenizer = tokenizer,
                    data = dataset[lang],
                    dataset_type = dataset[lang]['type'],
                    model_name = args.model_name,
                    device = device,
                    config = config
                )
                    
                if args.kns_compute:
                    kn.compute_knowledge_neurons()
                
                if args.kns_eval:
                    scores, relative_probs = kn.knowledge_neurons_surgery(kns_match = not(args.kns_unmatch))
                    
                    plots.plot_kns_surgery(
                                        scores, 
                                        relative_probs, 
                                        os.path.join(kn.kns_path, kn.dataset_type), 
                                        kns_match = not(args.kns_unmatch),
                                        lang = lang
                                        )
                
        if args.kns_analysis:   
            # Here analysis is a bit different. We will just focus on KNs (all, syn, sem, know) 
            # intersection accross languages
            
            kn = KnowledgeNeurons(
                    model = None,
                    tokenizer = None,
                    data = None,
                    dataset_type = None,
                    model_name = args.model_name,
                    device = None,
                    config = config
                )
            
            analysis_res = kn.compute_kns_multilingual_analysis()
            
            plots.plot_multilingual_analysis(
                                            analysis_res,
                                            kns_path = kn.kns_path,
                                            dataset_type = 'pararel', # /!\
                                            config = config,
                                            )
            exit(0)
    else:
        
        if args.kns_compute or args.kns_eval or args.kns_exps:
            if dataset:
                dataset_type = dataset['type']
            else:
                if args.autoprompt:
                    dataset_type = 'autoprompt'
                else:
                    dataset_type = 'pararel'
            
            # Initialize KnowledgeNeurons Object
            kn = KnowledgeNeurons(
                model = model.model,
                tokenizer = tokenizer,
                data = dataset,
                dataset_type = dataset_type,
                model_name = args.model_name,
                device = device,
                config = config
            )
                
        if args.kns_compute:
            kn.compute_knowledge_neurons()
        
        if args.kns_eval:
            scores, relative_probs = kn.knowledge_neurons_surgery(kns_match = not(args.kns_unmatch))
            
            plots.plot_kns_surgery(
                                scores, 
                                relative_probs, 
                                os.path.join(kn.kns_path, kn.dataset_type), 
                                kns_match = not(args.kns_unmatch))
            
            if config.WANDB:
                wandb.finish()
            exit(0)
            
    
        if args.kns_exps:
            
            kns_mode = 'equal' if args.equal else 'all'
            
            scores = kn.compute_experiments(
                                        thresh = config.ACCROSS_UUIDS_THRESHOLD, 
                                        kns_mode=kns_mode, 
                                        exps = [1, 2], 
                                        db_fact=config.TRIVIAL_PROMPT_ACTIVATION_FACT)
            
            plots.plot_kns_exps(scores = scores, 
                                kns_path=kn.kns_path,
                                config = config, 
                                dataset_name = kn.dataset_type)
            
            if config.WANDB:
                wandb.finish()
            exit(0)
            
        if args.kns_analysis:
            
            models_analysis = {}
            for model_name in config.MODELS_TO_PLOT:
                if os.path.exists(os.path.join(config.PATH_TO_KNS_DIR, model_name)): # Allow to avoid an error as we run models in //
                    kn = KnowledgeNeurons(
                                model = None,
                                tokenizer = None,
                                data = None,
                                dataset_type = None,
                                model_name = model_name,
                                device = None,
                                config = config
                            )
                    
                
                    analysis_res = kn.compute_kns_analysis()
                    models_analysis[model_name] = analysis_res
                else:
                    print(f"Do not have the KNs results for {model_name} yet! Skipping analysis.")
                
            
            
            ### New Definitive Figs ###
            
            plots.plot_KNs_types_all_models(
                                        models_analysis,
                                        kns_path = config.PATH_TO_KNS_DIR,
                                        plot_error_bars = config.PLOT_ERROR_BARS,
                                        wandb_flag=config.WANDB
                                        ) 
            plots.plot_sem_syn_know_layer_distribution(
                                        models_analysis,
                                        threshold = config.ACCROSS_UUIDS_THRESHOLD,
                                        kns_path=config.PATH_TO_KNS_DIR,
                                        wandb_flag=config.WANDB
                                        )
            
            if config.WANDB:
                wandb.finish()
            exit(0)
            
            
            ### Below Depreciated ###
            if config.PLOT_OLD_FIGS:
                # KNs categories (fig with layers & all KNs types)
                plots.plot_KNs_categories(
                                    res = analysis_res['category'],
                                    thresh = config.ACCROSS_UUIDS_THRESHOLD,
                                    kns_path = kn.kns_path
                                    )
                
                # Big figure with lama rela subject & num associated sem KNs
                plots.plot_sem_kns_to_rela_category(
                                    sem_kns2rela=analysis_res['sem_kns2rela'],
                                    kns_path = kn.kns_path
                                    )
                
                # Proportion Sem, Syn, Know KNs by threhsold
                plots.plot_sem_syn_know_dist(
                                    res_array = analysis_res['pararel_sem_syn_know_dist'],
                                    file_name_suffix = 'pararel',
                                    kns_path = kn.kns_path
                                    )
                plots.plot_sem_syn_know_dist(
                                    res_array = analysis_res['autoprompt_sem_syn_know_dist'],
                                    file_name_suffix = 'autoprompt',
                                    kns_path = kn.kns_path
                                    )
                
                # Number of Syntax & Semnatics KNs
                plots.plot_KNs_num_by_threshold(kns = analysis_res['sem_syn_num_avg_pararel'],
                                                title_suffix = '- Semantics & Syntax (Pararel)',
                                                file_name_suffix = 'sem_pararel',
                                                kns_path = kn.kns_path)
                
                plots.plot_KNs_num_by_threshold(kns = analysis_res['sem_syn_num_avg_autoprompt'],
                                                title_suffix = '- Semantics & Syntax (Autoprompt)',
                                                file_name_suffix = 'sem_autoprompt',
                                                kns_path = kn.kns_path)
            
        if args.kns_overlap:
            # I think a bit outdated
            layer_kns_pararel, layer_kns_autoprompt, layer_overlap_kns = kn.compute_overlap()
            
            plots.plot_KNs_layer_distribution(
                                layer_kns_pararel, 
                                num_layers = model.model.config.num_hidden_layers,
                                dataset = 'pararel',
                                overlap = False,
                                kns_path = kn.kns_path
                                )
            plots.plot_KNs_layer_distribution(
                                layer_kns_autoprompt, 
                                num_layers = model.model.config.num_hidden_layers,
                                dataset = 'autoprompt',
                                overlap = False,
                                kns_path = kn.kns_path
                                )
            
            plots.plot_KNs_layer_distribution(
                                layer_overlap_kns, 
                                num_layers = model.model.config.num_hidden_layers,
                                overlap = True,
                                kns_path = kn.kns_path
                                )
        
    
    