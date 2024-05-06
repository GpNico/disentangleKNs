
from captum.attr import IntegratedGradients, LayerActivation
import json
import copy
import numpy as np
import os
from pathlib import Path
import torch
import tqdm
import wandb
from typing import Dict, List, Tuple, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils import compute_variation, tensors_intersection_size, find_closest_elem, pad_input_ids, is_autoregressive, get_model_intermediate_layer, get_intermediate_dim
from config import Config


TREX_CATEGORIES = {
                'P17': 'Country',
                'P19': 'Country-City',
                'P20': 'Country-City',
                'P27': 'Country',
                'P30': 'Continent',
                'P36': 'Country',
                'P31': 'Thing', 
                'P37': 'Language',
                'P39': 'Profession',
                'P47': 'Country-City',
                'P101': 'Profession-Field',
                'P103': 'Language',
                'P106': 'Profession',
                'P108': 'Company',
                'P127': 'Country-City-Company-Person',
                'P131': 'Region-City',
                'P136': 'Music',
                'P138': 'Thing',
                'P140': 'Religion',
                'P159': 'Country-City',
                'P176': 'Company',
                'P178': 'Company',
                'P190': 'City',
                'P264': 'Music_Label',
                'P276': 'City',
                'P279': 'Thing',
                'P361': 'Thing',
                'P364': 'Language',
                'P407': 'Language',
                'P413': 'Sport_Position',
                'P449': 'Radio-TV',
                'P463': 'Organism',
                'P495': 'Country',
                'P527': 'Thing',
                'P530': 'Country',
                'P740': 'City',
                'P937': 'City',
                'P1001': 'Country',
                'P1303': 'Music_Instrument',
                'P1376': 'Country',
                'P1412': 'Language',
                }

class KnowledgeNeurons:
    """
        Compute Knowledge Neurons (KNs) of HuggingFace Models
        based on this paper:
         https://arxiv.org/abs/2104.08696
         
    """
    
    # Write here supported models layers count
    model_layers_num = {
                "bert-base-uncased": 12,
                "bert-large-uncased": 24,
                "bert-base-multilingual-uncased": 12,
                "opt-350m": 24,
                "opt-6.7b": 32,
                'Llama-2-7b-hf': 32,
                'flan-t5-xl': 24
                }
    
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 data: Dict[str, Dict[str, Union[List[str], str]]],
                 dataset_type: str,
                 model_name: str,
                 device: str,
                 config: Config,
                 p_thresh: float = None) -> None:
        
        #self.model = torch.compile(model) # I am Speed doesn't work on windows but will work on Jean Zay
        self.model = model
        self.model_name = model_name
        self.is_autoregressive = is_autoregressive(model_name)
        self.tokenizer = tokenizer
        self.data = data
        self.kns_path = os.path.join(config.PATH_TO_KNS_DIR, model_name)
        
        if dataset_type:
            if dataset_type[:2] == 'm_':
                self.multilingual = True
                _, self.dataset_type, self.lang = dataset_type.split('_')
            else:
                self.multilingual = False
                self.dataset_type = dataset_type
                self.lang = '' # Trick for paths
            # Create Dir
            os.makedirs(os.path.join(self.kns_path, self.dataset_type, self.lang), exist_ok=True)
        
        self.device = device
        self.config = config

        if p_thresh:
            self.p_thresh = p_thresh
        else:
            self.p_thresh = self.config.P_THRESH
        print('threshold p = ', self.p_thresh )
        
        
        
    def _compute_kns_one_predicate_id(self, predicate_id: str):
        """
            This function was origally contained in the 
            compute_knowledge_neurons. But as we need speed
            we need to parallelize, hence we need to access
            the computation of a single predicate_id. 
        
        """
        if self.config.WANDB:
            run_name = f"KNs {predicate_id} "
            if self.multilingual:
                run_name += self.dataset_type
                run_name += f' ({self.lang.upper()})'
            else:
                run_name += self.dataset_type
            wandb.init(
                    project=self.model_name, 
                    name=run_name, 
                    mode="offline" # /!\
                    )
        
        
            
        already_computed_kns = {}
        seen_uuids = {}
        for p in self.config.P_THRESHS:
            if os.path.exists(os.path.join(self.kns_path, self.dataset_type, self.lang, f'kns_p_{p}', predicate_id + '.json')):
                with open(os.path.join(self.kns_path, self.dataset_type, self.lang, f'kns_p_{p}', predicate_id + '.json'), 'r') as f:
                    p_already_computed_kns = json.load(f)
                p_seen_uuids = list(p_already_computed_kns.keys())
            else:
                os.makedirs(os.path.join(self.kns_path, self.dataset_type, self.lang, f'kns_p_{p}'), exist_ok=True)
                p_already_computed_kns = {}
                p_seen_uuids = []
            already_computed_kns[p] = p_already_computed_kns
            seen_uuids[p] = p_seen_uuids

        kns_rela = self.compute_knowledge_neurons_by_uuid(
                            predicate_id=predicate_id,
                            seen_uuids = seen_uuids,
                            t_thresh=self.config.T_THRESH,
                            p_threshs=self.config.P_THRESHS
                            )
        if kns_rela:
            for p in self.config.P_THRESHS:
                if len(already_computed_kns[p]) > 0:
                    kns_rela[p].update(already_computed_kns[p])
                
                with open(os.path.join(self.kns_path, self.dataset_type, self.lang, f'kns_p_{p}', f'{predicate_id}.json'), 'w') as f:
                    json.dump(kns_rela[p], f)
                    
        if self.config.WANDB:
            wandb.finish()
        
    def compute_knowledge_neurons(self) -> None:
        
        for k, predicate_id in enumerate(self.data.keys()):
            self._compute_kns_one_predicate_id(predicate_id=predicate_id)
              
                    
                    
    def knowledge_neurons_surgery(self,
                                  kns_match: bool = True,
                                  correct_category: bool = True) -> Dict[str, Dict[str, float]]:
        """
        
            If kns_match = True the KNs corresponding to self.data are loaded.
            Otherwise the other KNs are load (for example, self.data is autoprompt, KNs ParaRel). 
        
        """
        if not(correct_category):
            scores = {
                    'vanilla': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0},
                    'wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0},
                    'db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0}
                    }
        else:
            scores = {
                    'vanilla': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                    'wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                    'db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0}
                    }
        num_eval = 0
        relative_probs = {
                        'wo_kns': {},
                        'db_kns': {}
                        }
        
        # KNs path
        if kns_match:
            kns_path = os.path.join(self.kns_path, self.dataset_type, self.lang, f'kns_p_{self.p_thresh}')
            rela_names = [n[:-5] for n in os.listdir(kns_path) if '.json' in n]
        else:
            # So rela were ignored by ParaRel
            rela_names = list(
                        set(
                            [n[:-5] for n in os.listdir(os.path.join(self.kns_path, 'autoprompt', self.lang, f'kns_p_{self.p_thresh}')) if '.json' in n]
                            ).intersection(
                                set(
                                    [n[:-5] for n in os.listdir(os.path.join(self.kns_path, 'pararel', self.lang, f'kns_p_{self.p_thresh}')) if '.json' in n]
                                    )
                            )
                        )
            if self.dataset_type == 'autoprompt':
                kns_path = os.path.join(self.kns_path, 'pararel', self.lang, f'kns_p_{self.p_thresh}')
            elif self.dataset_type == 'pararel':
                kns_path = os.path.join(self.kns_path, 'autoprompt', self.lang, f'kns_p_{self.p_thresh}')
            else:
                raise
        
        for i, rela in enumerate(rela_names):
            if rela not in self.data.keys():
                continue
            try:
                 # Getting KNs
                with open(os.path.join(kns_path, rela + '.json'), 'r') as f:
                    rela_kns = json.load(f)
            except:
                print(f"Error with {rela}. Skipped.")
                continue
            print("Compuing Rela ", rela, f" ({i+1}/{len(rela_names)})")
            
            # Vanilla
            vanilla_raw_res = self.eval_one_rela_by_uuid(
                    predicate_id = rela,
                    mode = None,
                    rela_kns = None,
                    correct_category = correct_category
                    )
            if vanilla_raw_res:
                # Storing P@k
                num_eval += len(vanilla_raw_res[0])
                for uuid in vanilla_raw_res[0]:
                    for _s in scores['vanilla'].keys():
                        scores['vanilla'][_s] += vanilla_raw_res[0][uuid][_s]
                        
            # Erase
            wo_raw_res = self.eval_one_rela_by_uuid(
                    predicate_id = rela,
                    mode = 'wo',
                    rela_kns = rela_kns,
                    correct_category = correct_category
                    )
            if wo_raw_res:
                # Storing P@k
                for uuid in vanilla_raw_res[0]:
                    for _s in scores['wo_kns'].keys():
                        scores['wo_kns'][_s] += wo_raw_res[0][uuid][_s]
                # Storing Relative Probs
                relative_probs['wo_kns'][rela] = compute_variation(vanilla_raw_res[1], wo_raw_res[1])
                
            # Enhance
            db_raw_res = self.eval_one_rela_by_uuid(
                    predicate_id = rela,
                    mode = 'db',
                    rela_kns = rela_kns,
                    correct_category = correct_category
                    )
            if db_raw_res:
                # Storing P@k
                for uuid in vanilla_raw_res[0]:
                    for _s in scores['db_kns'].keys():
                        scores['db_kns'][_s] += db_raw_res[0][uuid][_s]
                # Storing Relative Probs
                relative_probs['db_kns'][rela] = compute_variation(vanilla_raw_res[1], db_raw_res[1])
                
        scores['vanilla'] = {k: v/num_eval for k, v in scores['vanilla'].items()}
        scores['db_kns'] = {k: v/num_eval for k, v in scores['db_kns'].items()}
        scores['wo_kns'] = {k: v/num_eval for k, v in scores['wo_kns'].items()}
                    
        return scores, relative_probs
    
    def compute_overlap(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        
        # Result path
        #res_path = Path(self.kns_path).parent.absolute()
        # ParaRel path
        pararel_path = os.path.join(self.kns_path, 'pararel')
        # Autoprompt path
        autoprompt_path = os.path.join(self.kns_path, 'autoprompt')
        
        # Get relations names
        rela_names = list(
                        set(
                            [n[:-5] for n in os.listdir(pararel_path) if '.json' in n]
                            ).intersection(
                                set(
                                    [n[:-5] for n in os.listdir(autoprompt_path) if '.json' in n]
                                    )
                            )
                        )
        
        # scores
        overlap_by_rela = {}
        
        # Used for global overlap
        autoprompt_kns = {}
        pararel_kns = {}
        print("Intra Relation Overlap:")
        for i, rela in enumerate(rela_names):
            # Getting KNs by rela for rela overlap
            with open(os.path.join(autoprompt_path, rela + '.json'), 'r') as f:
                autoprompt_rela_kns = json.load(f)
            autoprompt_kns.update(autoprompt_rela_kns)
            with open(os.path.join(pararel_path, rela + '.json'), 'r') as f:
                pararel_rela_kns = json.load(f)
            pararel_kns.update(pararel_rela_kns)
            
            # Compute rela overlap
            overlap_by_rela[rela] = KnowledgeNeurons._overlap_metrics(pararel_rela_kns, 
                                                                      autoprompt_rela_kns)
                
            # Print Overlap
            print(f"\t{rela}: {np.round(overlap_by_rela[rela]['overlap'], 2)}  (on {np.round(overlap_by_rela[rela]['num_kns_1'] ,2)} ParaRel KNs & {np.round(overlap_by_rela[rela]['num_kns_2'] ,2)} Autoprompt KNs)")
            
        # Global overlap
        global_overlap = KnowledgeNeurons._overlap_metrics(pararel_kns, 
                                                           autoprompt_kns)
        
        print(f"Total Overlap: {np.round(global_overlap['overlap'], 2)}  (on {np.round(global_overlap['num_kns_1'] ,2)} ParaRel KNs & {np.round(global_overlap['num_kns_2'] ,2)} Autoprompt KNs)")
        
        return global_overlap['layer_kns_1'], global_overlap['layer_kns_2'], global_overlap['layer_overlap_kns']
    
    
    
    def compute_experiments(self, 
                            thresh: float,
                            kns_mode: str = 'all',
                            exps: List[int] = [1,2],
                            db_fact: float = 2.) -> None:
        """
        
            kns_mode (str) 'all' : using all kns to compute boost
                           'equal' : using the same amount of KNs to compute boost 
                                     so equal to min( |sem_kns|, |syn_kns|)
            db_fact (float) only used in exp 2 for now. It replaces the x2 in the boost experiment
        
        
        """
        
        scores = {}
        
        assert kns_mode in ['all', 'equal']
        
        analysis_dict = self.compute_kns_analysis()
        
        if 1 in exps:
            ### EXP 1 ###
            # We measure correct category proportion by boosting
            # respectively: Semantics, Syntax & Knowledge Neurons
            # Expected Results: Only semantics should boost correct category
            #
            # Remark: as the number of semantics, syntax & knowledge KNs is
            #         not the same we can test using all & the same amounts of KNs
            
            scores_exp1 = {
                        'vanilla': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'sem_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'sem_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'syn_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'syn_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'only_know_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'only_know_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'shared_know_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'shared_know_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0}
                        }
            num_eval = {k: 0 for k in scores_exp1.keys()}
            
            for k, rela in enumerate(analysis_dict['rela_names']):
                
                if self.config.DEBUG:
                    if k == 3:
                        break
                
                thresholds = list(analysis_dict[rela]['sem_kns'].keys())
                _thresh = find_closest_elem(thresholds, thresh)
                
                _sem_kns = analysis_dict[rela]['sem_kns'][_thresh]
                
                if self.dataset_type == 'pararel':
                    _syn_kns = analysis_dict[rela]['pararel_syn_kns'][_thresh]
                    _shared_know_kns = analysis_dict[rela]['shared_know_kns']
                    _only_know_kns = analysis_dict[rela]['pararel_only_know_kns']
                elif self.dataset_type == 'autoprompt':
                    _syn_kns = analysis_dict[rela]['autoprompt_syn_kns'][_thresh]
                    _shared_know_kns = analysis_dict[rela]['shared_know_kns']
                    _only_know_kns = analysis_dict[rela]['autoprompt_only_know_kns']
                    
                
                if len(_sem_kns) == 0 or len(_syn_kns) == 0:
                    print(f"Skipping {rela} as Semantics and/or Syntax KNs are empty.")
                    print("\tNum. Semantics KNs: ", len(_sem_kns))
                    print("\tNum. Syntax KNs: ", len(_syn_kns))
                    continue
                else:
                    print(f"EXP1: Computing {rela}... ({k+1}/{len(analysis_dict['rela_names'])})")
                    print("\tNum. Semantics KNs: ", len(_sem_kns))
                    print("\tNum. Syntax KNs: ", len(_syn_kns))
                    
                if kns_mode == 'equal':
                    m = min(len(_sem_kns), len(_syn_kns))
                    _sem_kns = _sem_kns[:m]
                    _syn_kns = _syn_kns[:m]
                    _only_know_kns = {k: v[:m] for k,v in _only_know_kns.items()}
                    _shared_know_kns = {k: v[:m] for k,v in _shared_know_kns.items()}
                
                
                ### Vanilla ###
                
                vanilla_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = None,
                        rela_kns = None,
                        correct_category = True
                        )
                
                if vanilla_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['vanilla'] += len(vanilla_raw_res[0])
                    for uuid in vanilla_raw_res[0]:
                        for _s in scores_exp1['vanilla'].keys():
                            scores_exp1['vanilla'][_s] += vanilla_raw_res[0][uuid][_s]
                
                ### Semantics Eval ###
                
                sem_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _sem_kns,
                        correct_category = True
                        )
                if sem_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['sem_wo_kns'] += len(sem_wo_raw_res[0])
                    for uuid in sem_wo_raw_res[0]:
                        for _s in scores_exp1['sem_wo_kns'].keys():
                            scores_exp1['sem_wo_kns'][_s] += sem_wo_raw_res[0][uuid][_s]
                            
                sem_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _sem_kns,
                        correct_category = True
                        )
                if sem_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['sem_db_kns'] += len(sem_db_raw_res[0])
                    for uuid in sem_db_raw_res[0]:
                        for _s in scores_exp1['sem_db_kns'].keys():
                            scores_exp1['sem_db_kns'][_s] += sem_db_raw_res[0][uuid][_s]
                            
                ### Syntax Eval ###
                
                syn_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _syn_kns,
                        correct_category = True
                        )
                if syn_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['syn_wo_kns'] += len(syn_wo_raw_res[0])
                    for uuid in syn_wo_raw_res[0]:
                        for _s in scores_exp1['syn_wo_kns'].keys():
                            scores_exp1['syn_wo_kns'][_s] += syn_wo_raw_res[0][uuid][_s]
                            
                syn_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _syn_kns,
                        correct_category = True
                        )
                if syn_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['syn_db_kns'] += len(syn_db_raw_res[0])
                    for uuid in syn_db_raw_res[0]:
                        for _s in scores_exp1['syn_db_kns'].keys():
                            scores_exp1['syn_db_kns'][_s] += syn_db_raw_res[0][uuid][_s]
                            
                ### Knowledge Eval ###
                
                only_know_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _only_know_kns,
                        correct_category = True,
                        )
                if only_know_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['only_know_wo_kns'] += len(only_know_wo_raw_res[0])
                    for uuid in only_know_wo_raw_res[0]:
                        for _s in scores_exp1['only_know_wo_kns'].keys():
                            scores_exp1['only_know_wo_kns'][_s] += only_know_wo_raw_res[0][uuid][_s]
                            
                only_know_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _only_know_kns,
                        correct_category = True,
                        )
                if only_know_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['only_know_db_kns'] += len(only_know_db_raw_res[0])
                    for uuid in only_know_db_raw_res[0]:
                        for _s in scores_exp1['only_know_db_kns'].keys():
                            scores_exp1['only_know_db_kns'][_s] += only_know_db_raw_res[0][uuid][_s]
                            
                shared_know_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _shared_know_kns,
                        correct_category = True,
                        )
                if shared_know_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['shared_know_wo_kns'] += len(shared_know_wo_raw_res[0])
                    for uuid in shared_know_wo_raw_res[0]:
                        for _s in scores_exp1['shared_know_wo_kns'].keys():
                            scores_exp1['shared_know_wo_kns'][_s] += shared_know_wo_raw_res[0][uuid][_s]
                            
                shared_know_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _shared_know_kns,
                        correct_category = True,
                        )
                if shared_know_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['shared_know_db_kns'] += len(shared_know_db_raw_res[0])
                    for uuid in shared_know_db_raw_res[0]:
                        for _s in scores_exp1['shared_know_db_kns'].keys():
                            scores_exp1['shared_know_db_kns'][_s] += shared_know_db_raw_res[0][uuid][_s]
                            
                
                            
            
            scores_exp1 = {k1: {k2: v/num_eval[k1] for k2, v in scores_exp1[k1].items()} for k1 in scores_exp1.keys()}
            
            # Storing mode
            scores_exp1['kns_mode'] = kns_mode
            scores_exp1['threshold'] = thresh
            
            # Storing scores
            scores[1] = scores_exp1
            
        
        if 2 in exps:
            ### EXP 2 ###
            # Promptless experiment: X [MASK].
            # KNs are supposed to help even without prompts.
            # X [MASK] boost with sem & know should help
            # [MASK] boost with know should help but not with sem 
            
            
            scores_exp2 = {
                        'vanilla': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'sem_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'sem_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'syn_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'syn_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'only_know_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'only_know_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'shared_know_wo_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0},
                        'shared_know_db_kns': {'P@1': 0, 'P@5': 0, 'P@20': 0, 'P@100': 0, 'ccp@1': 0, 'ccp@5': 0, 'ccp@20': 0, 'ccp@100': 0}
                        }
            num_eval = {k: 0 for k in scores_exp1.keys()}
            
            for k, rela in enumerate(analysis_dict['rela_names']):
                
                if self.config.DEBUG:
                    if k == 3:
                        break
                
                thresholds = list(analysis_dict[rela]['sem_kns'].keys())
                _thresh = find_closest_elem(thresholds, thresh)
                
                _sem_kns = analysis_dict[rela]['sem_kns'][_thresh]
                
                if self.dataset_type == 'pararel':
                    _syn_kns = analysis_dict[rela]['pararel_syn_kns'][_thresh]
                    _shared_know_kns = analysis_dict[rela]['shared_know_kns']
                    _only_know_kns = analysis_dict[rela]['pararel_only_know_kns']
                elif self.dataset_type == 'autoprompt':
                    _syn_kns = analysis_dict[rela]['autoprompt_syn_kns'][_thresh]
                    _shared_know_kns = analysis_dict[rela]['shared_know_kns']
                    _only_know_kns = analysis_dict[rela]['autoprompt_only_know_kns']
                
                if len(_sem_kns) == 0 or len(_syn_kns) == 0:
                    print(f"Skipping {rela} as Semantics and/or Syntax KNs are empty.")
                    print("\tNum. Semantics KNs: ", len(_sem_kns))
                    print("\tNum. Syntax KNs: ", len(_syn_kns))
                    continue
                else:
                    print(f"EXP2: Computing {rela}... ({k+1}/{len(analysis_dict['rela_names'])})")
                    print("\tNum. Semantics KNs: ", len(_sem_kns))
                    print("\tNum. Syntax KNs: ", len(_syn_kns))
                    
                if kns_mode == 'equal':
                    m = min(len(_sem_kns), len(_syn_kns)) # Asserting there will always be more Knowledge KNs
                    _sem_kns = _sem_kns[:m]
                    _syn_kns = _syn_kns[:m]
                    _only_know_kns = {k: v[:m] for k,v in _only_know_kns.items()}
                    _shared_know_kns = {k: v[:m] for k,v in _shared_know_kns.items()}
                    
                ### Creating Custom Prompts ###
                
                custom_dataset = {}
                _autoregressive = is_autoregressive(model_name=self.model_name)
                for uuid in self.data[rela]:
                    X, Y, num_prompts = self.data[rela][uuid]['X'], self.data[rela][uuid]['Y'], self.data[rela][uuid]['num_prompts']
                    
                    if _autoregressive:
                        _prompt_tok = self.tokenizer(X, add_special_tokens=True).input_ids
                    else:
                        _prompt_tok = self.tokenizer(f'{X} {self.tokenizer.mask_token} .', add_special_tokens=True).input_ids
                    
                    custom_dataset[uuid] = {'sentences_tok': [torch.tensor(_prompt_tok)],
                                            'X': X,
                                            'Y': Y,
                                            'num_prompts': num_prompts}
                
                ### Vanilla ###
                
                vanilla_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = None,
                        rela_kns = None,
                        correct_category = True,
                        custom_dataset = custom_dataset
                        )
                
                if vanilla_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['vanilla'] += len(vanilla_raw_res[0])
                    for uuid in vanilla_raw_res[0]:
                        for _s in scores_exp2['vanilla'].keys():
                            scores_exp2['vanilla'][_s] += vanilla_raw_res[0][uuid][_s]
                
                ### Semantics Eval ###
                
                sem_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _sem_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset,
                        db_fact = db_fact
                        )
                if sem_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['sem_wo_kns'] += len(sem_wo_raw_res[0])
                    for uuid in sem_wo_raw_res[0]:
                        for _s in scores_exp2['sem_wo_kns'].keys():
                            scores_exp2['sem_wo_kns'][_s] += sem_wo_raw_res[0][uuid][_s]
                            
                sem_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _sem_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset,
                        db_fact = db_fact
                        )
                if sem_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['sem_db_kns'] += len(sem_db_raw_res[0])
                    for uuid in sem_db_raw_res[0]:
                        for _s in scores_exp2['sem_db_kns'].keys():
                            scores_exp2['sem_db_kns'][_s] += sem_db_raw_res[0][uuid][_s]
                            
                ### Syntax Eval ###
                
                syn_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _syn_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset
                        )
                if syn_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['syn_wo_kns'] += len(syn_wo_raw_res[0])
                    for uuid in syn_wo_raw_res[0]:
                        for _s in scores_exp2['syn_wo_kns'].keys():
                            scores_exp2['syn_wo_kns'][_s] += syn_wo_raw_res[0][uuid][_s]
                            
                syn_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _syn_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset
                        )
                if syn_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['syn_db_kns'] += len(syn_db_raw_res[0])
                    for uuid in syn_db_raw_res[0]:
                        for _s in scores_exp2['syn_db_kns'].keys():
                            scores_exp2['syn_db_kns'][_s] += syn_db_raw_res[0][uuid][_s]
                            
                ### Knowledge Eval ###
                            
                only_know_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _only_know_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset,
                        db_fact = db_fact
                        )
                if only_know_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['only_know_wo_kns'] += len(only_know_wo_raw_res[0])
                    for uuid in only_know_wo_raw_res[0]:
                        for _s in scores_exp2['only_know_wo_kns'].keys():
                            scores_exp2['only_know_wo_kns'][_s] += only_know_wo_raw_res[0][uuid][_s]
                            
                only_know_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _only_know_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset,
                        db_fact = db_fact
                        )
                if only_know_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['only_know_db_kns'] += len(only_know_db_raw_res[0])
                    for uuid in only_know_db_raw_res[0]:
                        for _s in scores_exp2['only_know_db_kns'].keys():
                            scores_exp2['only_know_db_kns'][_s] += only_know_db_raw_res[0][uuid][_s]
                            
                shared_know_wo_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'wo',
                        rela_kns = _shared_know_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset,
                        db_fact = db_fact
                        )
                if shared_know_wo_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['shared_know_wo_kns'] += len(shared_know_wo_raw_res[0])
                    for uuid in shared_know_wo_raw_res[0]:
                        for _s in scores_exp2['shared_know_wo_kns'].keys():
                            scores_exp2['shared_know_wo_kns'][_s] += shared_know_wo_raw_res[0][uuid][_s]
                            
                shared_know_db_raw_res = self.eval_one_rela_by_uuid(
                        predicate_id = rela,
                        mode = 'db',
                        rela_kns = _shared_know_kns,
                        correct_category = True,
                        custom_dataset = custom_dataset,
                        db_fact = db_fact
                        )
                if shared_know_db_raw_res:
                    # Storing P@k & CCP@k
                    num_eval['shared_know_db_kns'] += len(shared_know_db_raw_res[0])
                    for uuid in shared_know_db_raw_res[0]:
                        for _s in scores_exp2['shared_know_db_kns'].keys():
                            scores_exp2['shared_know_db_kns'][_s] += shared_know_db_raw_res[0][uuid][_s]
                            
                
            scores_exp2 = {k1: {k2: v/num_eval[k1] for k2, v in scores_exp2[k1].items()} for k1 in scores_exp2.keys()}
            
            # Storing params
            scores_exp2['kns_mode'] = kns_mode
            scores_exp2['threshold'] = thresh
            scores_exp2['db_fact'] = db_fact
            
            # Storing scores
            scores[2] = scores_exp2
            
            ### EXP 3 ###
            # Syntax KNs should increase the probability of the correct POS overall?
            # When doing multilingual correlation shared syntax tokens wrt hamming distance?
        
        return scores 
    
    
    
    
    def compute_kns_analysis(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        
        # ParaRel path
        pararel_path = os.path.join(self.kns_path, 'pararel', f'kns_p_{self.p_thresh}')
        # Autoprompt path
        autoprompt_path = os.path.join(self.kns_path, 'autoprompt', f'kns_p_{self.p_thresh}')
        
        # Get relations names
        rela_names = list(
                        set(
                            [n[:-5] for n in os.listdir(pararel_path) if '.json' in n]
                            ).intersection(
                                set(
                                    [n[:-5] for n in os.listdir(autoprompt_path) if '.json' in n]
                                    )
                            )
                        )
        
        # Used for global overlap
        res_dict = {}
        autoprompt_kns, pararel_kns = {}, {}
        for i, rela in enumerate(rela_names):
            # Getting KNs by rela for rela overlap
            with open(os.path.join(autoprompt_path, rela + '.json'), 'r') as f:
                autoprompt_rela_kns = json.load(f)
            with open(os.path.join(pararel_path, rela + '.json'), 'r') as f:
                pararel_rela_kns = json.load(f)
            
            autoprompt_kns[rela] = autoprompt_rela_kns
            pararel_kns[rela] = pararel_rela_kns

            ### ANALYSIS ###
            rela_analysis = self._kns_analysis(pararel_rela_kns, autoprompt_rela_kns)
            res_dict[rela] = rela_analysis
            
            ### SEMANTICS, SYNTAX & KNOWLEDGE KNs SUM ###
            if 'sem_kns' not in res_dict.keys():
                res_dict['sem_kns'] = {thresh: set(kns) for thresh, kns in rela_analysis['sem_kns'].items()}
                
                res_dict['pararel_syn_kns'] = {thresh: set(kns) for thresh, kns in rela_analysis['pararel_syn_kns'].items()}
                res_dict['autoprompt_syn_kns'] = {thresh: set(kns) for thresh, kns in rela_analysis['autoprompt_syn_kns'].items()}
                
                res_dict['shared_know_kns_set'] = {kn for kn in rela_analysis['shared_know_kns_set']}
                res_dict['pararel_only_know_kns_set'] = {kn for kn in rela_analysis['pararel_only_know_kns_set']}
                res_dict['autoprompt_only_know_kns_set'] = {kn for kn in rela_analysis['autoprompt_only_know_kns_set']}
                
                res_dict['category'] = copy.copy(rela_analysis['category'])
            else:
                res_dict['shared_know_kns_set'].update(rela_analysis['shared_know_kns_set'])
                res_dict['pararel_only_know_kns_set'].update(rela_analysis['pararel_only_know_kns_set'])
                res_dict['autoprompt_only_know_kns_set'].update(rela_analysis['autoprompt_only_know_kns_set'])
                
                for thresh in res_dict['sem_kns'].keys():
                    res_dict['sem_kns'][thresh].update(rela_analysis['sem_kns'][thresh])
                    res_dict['pararel_syn_kns'][thresh].update(rela_analysis['pararel_syn_kns'][thresh])
                    res_dict['autoprompt_syn_kns'][thresh].update(rela_analysis['autoprompt_syn_kns'][thresh])
                    
                    for k in res_dict['category'][thresh].keys():
                        if 'count' in k:
                            res_dict['category'][thresh][k] += rela_analysis['category'][thresh][k]
                        else:
                            for l in res_dict['category'][thresh][k].keys():
                                res_dict['category'][thresh][k][l] += rela_analysis['category'][thresh][k][l]
                     
            
            ### SEMANTICS & SYNTAX KNs PROP AVG ###
            if 'sem_syn_num_avg_pararel' not in res_dict.keys():
                res_dict['sem_syn_num_avg_pararel'] = {thresh: len(kns)/len(rela_analysis['pararel_sem_syn_kns'][0.]) for thresh, kns in rela_analysis['pararel_sem_syn_kns'].items()}
                res_dict['sem_syn_num_avg_autoprompt'] = {thresh: len(kns)/len(rela_analysis['autoprompt_sem_syn_kns'][0.]) for thresh, kns in rela_analysis['autoprompt_sem_syn_kns'].items()}
            else:
                for thresh in res_dict['sem_syn_num_avg_pararel'].keys():
                    res_dict['sem_syn_num_avg_pararel'][thresh] += len(rela_analysis['pararel_sem_syn_kns'][thresh])/len(rela_analysis['pararel_sem_syn_kns'][0.])
                    res_dict['sem_syn_num_avg_autoprompt'][thresh] += len(rela_analysis['autoprompt_sem_syn_kns'][thresh])/len(rela_analysis['autoprompt_sem_syn_kns'][0.])
                    
        
        res_dict['sem_syn_num_avg_pararel'] = {k: v/len(rela_names) for k,v in res_dict['sem_syn_num_avg_pararel'].items()}
        res_dict['sem_syn_num_avg_autoprompt'] = {k: v/len(rela_names) for k,v in res_dict['sem_syn_num_avg_autoprompt'].items()}
        

        ## SEM, SYN, KOWN  PROPOPRTION ###
        
        sem_syn_know_dist_tuple = self._sem_syn_know_dist( # pararel res, autoprompt res, pararel res by rela, autoprompt res by rela
                                            res_dict,
                                            pararel_kns,
                                            autoprompt_kns,
                                            thresholds = list(res_dict['sem_kns'].keys()),
                                            rela_names = rela_names
                                            )
        res_dict['pararel_sem_syn_know_dist'] = sem_syn_know_dist_tuple[0]
        res_dict['autoprompt_sem_syn_know_dist'] = sem_syn_know_dist_tuple[1]
        res_dict['pararel_sem_syn_know_dist_se'] = sem_syn_know_dist_tuple[2]
        res_dict['autoprompt_sem_syn_know_dist_se'] = sem_syn_know_dist_tuple[3]
        
        ### CATEGORY ###
        
        for thresh in res_dict['sem_kns'].keys():
            for k in res_dict['category'][thresh].keys():
                if 'count' in k:
                    continue
                if 'syn' in k:
                    count = res_dict['category'][thresh]['syn_count']
                elif 'sem' in k:
                    count = res_dict['category'][thresh]['sem_count']
                elif 'know' in k:
                    count = res_dict['category'][thresh]['know_count']
                else:
                    raise Exception()
                
                if count == 0:
                    continue
                
                for l in res_dict['category'][thresh][k].keys():
                    res_dict['category'][thresh][k][l] /= count
                    
        # Sanity Check
        """
        for thresh in res_dict['category'].keys():
            sem_count, syn_count, know_count = 0, 0, 0
            sem_count = sum(list(res_dict['category'][thresh]['sem'].values()))
            syn_count += sum(list(res_dict['category'][thresh]['pararel_syn'].values()))
            syn_count += sum(list(res_dict['category'][thresh]['autoprompt_syn'].values()))
            know_count += sum(list(res_dict['category'][thresh]['pararel_know'].values()))
            know_count += sum(list(res_dict['category'][thresh]['autoprompt_know'].values()))
            know_count += sum(list(res_dict['category'][thresh]['pararel_autoprompt_know'].values()))
            assert abs(sem_count - 1.) < 0.001 or thresh == 1.
            assert abs(syn_count - 1.) < 0.001 or thresh == 1.
            assert abs(know_count - 1.) < 0.001 or thresh == 0.0
        """
        ### SEM KNS TO RELA ###
        
        res_dict['sem_kns2rela'] = self._sem_kns2rela(
                                        res_dict,
                                        thresholds = list(res_dict['sem_kns'].keys()),
                                        rela_names = rela_names
                                        )
        res_dict['rela_names'] = rela_names
        
        return res_dict
    
    
    def compute_kns_multilingual_analysis(self,):
        """
        
        
        """
        ### LOAD KNS ###
        
        # ParaRel path
        pararel_path = os.path.join(self.kns_path, 'pararel')
        # Autoprompt path
        autoprompt_path = os.path.join(self.kns_path, 'autoprompt')
        
        full_kns = {}
        for lang in self.config.LANGS:
            # Get relations names of THIS lang


            predicate_ids = list(
                            set(
                                [n[:-5] for n in os.listdir(os.path.join(pararel_path, lang, f'kns_p_{self.p_thresh}')) if '.json' in n]
                                ).intersection( # add instead of intersection /!\
                                    set(
                                        [n[:-5] for n in os.listdir(os.path.join(autoprompt_path, lang, f'kns_p_{self.p_thresh}')) if '.json' in n]
                                        )
                                )
                            )
            
            # Store all KNs
            autoprompt_kns, pararel_kns = {}, {}
            for predicate_id in predicate_ids:
                # Getting KNs by rela for rela overlap
                if os.path.exists(os.path.join(autoprompt_path, lang, f'kns_p_{self.p_thresh}', predicate_id + '.json')):
                    with open(os.path.join(autoprompt_path, lang, f'kns_p_{self.p_thresh}', predicate_id + '.json'), 'r') as f:
                        autoprompt_kns[predicate_id] = json.load(f)
                if os.path.exists(os.path.join(pararel_path, lang, f'kns_p_{self.p_thresh}', predicate_id + '.json')):
                    with open(os.path.join(pararel_path, lang, f'kns_p_{self.p_thresh}', predicate_id + '.json'), 'r') as f:
                        pararel_kns[predicate_id] = json.load(f)
                    
            full_kns[lang] = {'autoprompt': autoprompt_kns,
                              'pararel': pararel_kns}
            
        
        ### COMPUTE SEM, SYN & KNOW ###
        
        full_analysis = {}
        _thresh = None
        for lang in full_kns.keys():
            
            lang_analysis = {}
            
            for predicate_id in full_kns[lang]['pararel'].keys():
                lang_analysis[predicate_id] = self._kns_analysis(
                                            pararel_kns_dict = full_kns[lang]['pararel'][predicate_id],
                                            autoprompt_kns_dict = full_kns[lang]['autoprompt'][predicate_id]
                                            )
                if _thresh is None:
                    _thresh = find_closest_elem(list(lang_analysis[predicate_id]['sem_kns'].keys()), 
                                                self.config.ACCROSS_UUIDS_THRESHOLD)
             
            full_analysis[lang] = lang_analysis
        
        print(full_analysis.keys())
            
        ### COMPUTE BELONGINGS ###
        
        STUDIED_DATASET = 'pararel'
        
        # For now we'll do all KNs for all predicate_ids
        inter_predicate_ids = set()
        for lang in full_kns.keys():
            if len(inter_predicate_ids) == 0:
                inter_predicate_ids = inter_predicate_ids.union(set(full_kns[lang][STUDIED_DATASET].keys()))
            else:
                inter_predicate_ids = inter_predicate_ids.intersection(set(full_kns[lang][STUDIED_DATASET].keys()))
        
        print(f'Found: {len(inter_predicate_ids)} common predicate_ids.')
        
        kns_2_lang = {}
        sem_kns_2_lang = {}
        syn_kns_2_lang = {}
        know_kns_2_lang = {}
        
        for lang in full_kns.keys():
            
            for predicate_id in inter_predicate_ids:
                
                # KNS
                for uuid in full_kns[lang][STUDIED_DATASET][predicate_id]:
                    _kns = full_kns[lang][STUDIED_DATASET][predicate_id][uuid]
                    for kn in _kns:
                        kn = (kn[0], kn[1]) # Hashable
                        if kn in kns_2_lang.keys():
                            kns_2_lang[kn][lang] = True
                        else:
                            _res = {l: False for l in self.config.LANGS}
                            _res[lang] = True
                            kns_2_lang[kn] = _res
                            
                # SEM
                _sem_kns = full_analysis[lang][predicate_id]['sem_kns'][_thresh]
                for kn in _sem_kns:
                    if kn in sem_kns_2_lang.keys():
                        sem_kns_2_lang[kn][lang] = True
                    else:
                        _res = {l: False for l in self.config.LANGS}
                        _res[lang] = True
                        sem_kns_2_lang[kn] = _res
                        
                # SYN
                _syn_kns = full_analysis[lang][predicate_id][f'{STUDIED_DATASET}_syn_kns'][_thresh]
                for kn in _syn_kns:
                    if kn in syn_kns_2_lang.keys():
                        syn_kns_2_lang[kn][lang] = True
                    else:
                        _res = {l: False for l in self.config.LANGS}
                        _res[lang] = True
                        syn_kns_2_lang[kn] = _res
                        
                # KNOW
                for uuid in full_analysis[lang][predicate_id][f'{STUDIED_DATASET}_know_kns']:
                    _know_kns = full_analysis[lang][predicate_id][f'{STUDIED_DATASET}_know_kns'][uuid][_thresh]
                    for kn in _know_kns:
                        if kn in know_kns_2_lang.keys():
                            know_kns_2_lang[kn][lang] = True
                        else:
                            _res = {l: False for l in self.config.LANGS}
                            _res[lang] = True
                            know_kns_2_lang[kn] = _res
                            

                            
                            
        ### LAYER ANALYSIS ###
        
        
        # KNs
        layers_count = {l+1: {k: 0 for k in range(self.model_layers_num[self.model_name])} for l in range(len(self.config.LANGS))}
        
        for kn in kns_2_lang.keys():
            res = list(kns_2_lang[kn].values())
            layer, _ = kn
            num_langs = sum(res)
            layers_count[num_langs][layer] += 1
            
        # SEM
        sem_layers_count = {l+1: {k: 0 for k in range(self.model_layers_num[self.model_name])} for l in range(len(self.config.LANGS))}
        
        for kn in sem_kns_2_lang.keys():
            res = list(sem_kns_2_lang[kn].values())
            layer, _ = kn
            num_langs = sum(res)
            sem_layers_count[num_langs][layer] += 1
            
        # SYN
        syn_layers_count = {l+1: {k: 0 for k in range(self.model_layers_num[self.model_name])} for l in range(len(self.config.LANGS))}
        
        for kn in syn_kns_2_lang.keys():
            res = list(syn_kns_2_lang[kn].values())
            layer, _ = kn
            num_langs = sum(res)
            syn_layers_count[num_langs][layer] += 1
            
        # KNOW
        know_layers_count = {l+1: {k: 0 for k in range(self.model_layers_num[self.model_name])} for l in range(len(self.config.LANGS))}
        
        for kn in know_kns_2_lang.keys():
            res = list(know_kns_2_lang[kn].values())
            layer, _ = kn
            num_langs = sum(res)
            know_layers_count[num_langs][layer] += 1
            
        # Heatmap 
        
        heatmap = np.zeros((len(self.config.LANGS), len(self.config.LANGS)))
        for kn in kns_2_lang.keys():
            for idx1, (lang1, b1) in enumerate(kns_2_lang[kn].items()):
                assert self.config.LANGS[idx1] == lang1 # Never too sure
                for idx2, (lang2, b2) in enumerate(kns_2_lang[kn].items()):
                    assert self.config.LANGS[idx2] == lang2 # Never too sure
                    if b1 and b2: # kn appears in both lang
                        heatmap[idx1, idx2] += 1     
                        
        sem_heatmap = np.zeros((len(self.config.LANGS), len(self.config.LANGS)))
        for kn in sem_kns_2_lang.keys():
            for idx1, (lang1, b1) in enumerate(sem_kns_2_lang[kn].items()):
                assert self.config.LANGS[idx1] == lang1 # Never too sure
                for idx2, (lang2, b2) in enumerate(sem_kns_2_lang[kn].items()):
                    assert self.config.LANGS[idx2] == lang2 # Never too sure
                    if b1 and b2: # kn appears in both lang
                        sem_heatmap[idx1, idx2] += 1    
                        
        syn_heatmap = np.zeros((len(self.config.LANGS), len(self.config.LANGS)))
        for kn in syn_kns_2_lang.keys():
            for idx1, (lang1, b1) in enumerate(syn_kns_2_lang[kn].items()):
                assert self.config.LANGS[idx1] == lang1 # Never too sure
                for idx2, (lang2, b2) in enumerate(syn_kns_2_lang[kn].items()):
                    assert self.config.LANGS[idx2] == lang2 # Never too sure
                    if b1 and b2: # kn appears in both lang
                        syn_heatmap[idx1, idx2] += 1 
                        
        know_heatmap = np.zeros((len(self.config.LANGS), len(self.config.LANGS)))
        for kn in know_kns_2_lang.keys():
            for idx1, (lang1, b1) in enumerate(know_kns_2_lang[kn].items()):
                assert self.config.LANGS[idx1] == lang1 # Never too sure
                for idx2, (lang2, b2) in enumerate(know_kns_2_lang[kn].items()):
                    assert self.config.LANGS[idx2] == lang2 # Never too sure
                    if b1 and b2: # kn appears in both lang
                        know_heatmap[idx1, idx2] += 1    

        return (layers_count, sem_layers_count, syn_layers_count, know_layers_count, heatmap, sem_heatmap, syn_heatmap, know_heatmap) 
            

    def compute_knowledge_neurons_by_uuid(
                        self,
                        predicate_id: str,
                        t_thresh: float = None, # In github 0.3, in the paper 0.2
                        p_threshs: List[float] = None,  # In github 0.5, in the paper 0.7
                        seen_uuids: List[str] = None
                        ) -> Dict[float, Dict[str, List[Tuple[float, float]]]]:
        """
            Compute KNs for one relation from ParaRel (e.g. P264).
            Note that parameters are chosen based on this implementation: 
            https://github.com/EleutherAI/knowledge-neurons
            So here are the differences with the paper:
                t = in github 0.3, in the paper 0.2
                p = in github 0.5, in the paper 0.7
            We agreed to lower the threshold as we are already being to selective (i.e.
            fewer KNs than Github)
            
            TBD: Add adaptative Refining by increasing/decreasing p by 0.05 if the
                 number of KNs is not within [3,5].
                 
            Args:
                predicate_id (str) ParaRel relation id e.g. P1624
                t_thresh (float) t threshold from the paper
                p_thresh (float) p threshold from the paper
                
            Returns:
                kns (dict) keys: p threshold       values: (dict) keys: uuid      values: list of KNs in the format (layer, neuron) 
                           ex: {0.5: {'uuid1': [(11, 2045), (12, 751)], 'uuid2': [(5, 3014)]}}
        
        """
        
        # Get Dataset
        if predicate_id in self.data.keys():
            dataset = self.data[predicate_id]
        else:
            print(f"Relation {predicate_id} doesn't have enought prompts to compute KNs.")
            return None
        
        if dataset is None:
            return None
        
        if self.config.WANDB:
            vs = []

        # Compute IG attributions
        kns = {p: {} for p in p_threshs} # key p thresh, values dict, keys uuid
        for progess_idx, uuid in tqdm.tqdm(enumerate(dataset.keys()), total = len(dataset)):
            seen_bool = True
            for p in p_threshs:
                if seen_uuids[p] and uuid in seen_uuids[p]:
                    pass
                else:
                    seen_bool = False
            if seen_bool:
                continue
            
            # Log
            if self.config.WANDB:
                wandb.log({"Remaining uuids": len(dataset) - progess_idx})
            
            # Get template Tokens & Y 
            sentences_tok = dataset[uuid]['sentences_tok']    
            Y = dataset[uuid]['Y']
            
            # Target Tokenization
            target = self.tokenizer([Y]*len(sentences_tok), 
                                    return_tensors = 'pt',
                                    add_special_tokens=False)['input_ids'].to(self.device) # No need to pas

            # Pad & Create Attention Mask
            input_ids, attention_mask = pad_input_ids(sentences_tok)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Get target_pos
            if self.is_autoregressive:
                target_pos  = attention_mask.sum(dim=-1)-1 # that's a neat trick
            else:
                # assert MLM?
                target_pos = torch.where(input_ids == self.tokenizer.mask_token_id)[1]    
            
            ### Attr ###
            # little trick to avoid Cuda out of memory
            k, batch_size = 0, min(len(sentences_tok), self.config.KNS_BATCH_SIZE)
            fail_flag = False # Possible that even with only one sentence it crashes.
            uuid_attr_lst = []
            while k < len(sentences_tok):
                _input_ids = input_ids[k:k + batch_size, :]
                _attention_mask = attention_mask[k:k + batch_size, :]
                _target = target[k:k + batch_size, :] # /!\ I don't know why it wasn't like that
                _target_pos = target_pos[k:k + batch_size]
                try:
                    uuid_attr = []
                    for l in range(self.model.config.num_hidden_layers):
                        attr = self.IG_batch(
                                input_ids = _input_ids,
                                attention_mask = _attention_mask,
                                target = _target,
                                target_pos = _target_pos,
                                layer_num = l
                                )
                        # attr shape [BS, num_neurons]
                        uuid_attr.append(attr.unsqueeze(1)) # create layer dim (shape [BS, 1, num neurons])
                    k += batch_size
                    uuid_attr_lst.append(torch.cat(uuid_attr, dim=1)) # cat along layer dim (shape [BS, num_layers, num neurons])
                except RuntimeError as e:
                    # Not sure it works this BS reduction thingy
                    print(e)
                    print("error!")
                    if "CUDA out of memory" in str(e):
                        if batch_size == 1:
                            print("Couldn't load this uuid on GPU. Skipping it.")
                            fail_flag = True
                            break
                        print(f"Reducing batch size from {batch_size} to {batch_size // 2}")
                        batch_size //= 2
                        k = 0 # reseting batch count
                        uuid_attr_lst = []
                    else:
                        raise e
                if fail_flag:
                    continue

            uuid_attr = torch.vstack(uuid_attr_lst)

            ## Refining

            # for each prompt, retain the neurons with attribution scores greater than the attribution threshold t, obtaining the coarse set of knowledge neurons
            # BUT the max is computed for each prompt, not for all prompts at onec /!\
            max_attr = uuid_attr.max(dim = -1).values.max(dim = -1).values
            max_attr = max_attr.unsqueeze(1).unsqueeze(2) # to match uuid_attr shape
            threshold = t_thresh*max_attr # t of the paper

            prompts_idx, layer_idx, neuron_idx = torch.where(uuid_attr > threshold)

            neuron2prompt = {}
            for k in range(len(prompts_idx)):
                layer, neuron = layer_idx[k].item(), neuron_idx[k].item()
                if (layer, neuron) in neuron2prompt.keys():
                    neuron2prompt[(layer, neuron)] += 1
                else:
                    neuron2prompt[(layer, neuron)] = 1

            # considering all the coarse sets together, retain the knowledge neurons shared by more than p% prompts.
            for p_thresh in p_threshs:
                uuid_kns = [k for k,v in neuron2prompt.items() if v >= dataset[uuid]['num_prompts']*p_thresh]

                kns[p_thresh][uuid] = uuid_kns
                
            if self.config.WANDB:
                if progess_idx%50 == 10:
                    columns = ['uuid']
                    for p_thresh in p_threshs:
                        columns.append(f'num. KNs (p={p_thresh})')
                    table = wandb.Table(data = vs, columns=columns)
                    wandb.log({'Examples': table})
                    vs = []
                        
                else:
                    v = [uuid]
                    for p_thresh in p_threshs:
                        v.append(len(kns[p_thresh][uuid]))
                    vs.append(v)
                    

        return kns
        
    def IG_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        target_pos: torch.Tensor,
        layer_num: int,
        n_steps: int = 20
        ) -> torch.Tensor:
        """
            Compute Integrated Gradients of the second layer of
            the model's layer_num^th encoder FFN.
            This where KNs are supposed to hide.
            
            This code is using the Captum library that does not provide
            ways of attributing specific layer activations.
            
            For those who care, Captum provides ways of attributing the 
            INPUT with respect to a specific layer activation or neuron
            activation with the LayerIntegratedGradients and 
            NeuronIntegratedGradients.
            
            Returns:
                attr (Tensor) shape [batch_size, layer_size]
        """
        
        def forward_pos_func(hidden_states: torch.Tensor) -> torch.Tensor:
            """
                This function is a trick to evaluate a Pytorch model
                from a particular layer. 
                In this case the second Linear layer from the FFN of
                a particular encoder.
                
                To do this we register a hook at the desired layer that
                will change the output by the hidden_states tensor. 
                
                It must have a way of doing this more easily and efficiently
                as we are passing a tensor through the entire model instead 
                of just layer->output, but hey it works!
            """
        
            def _custom_hook(
                    module: torch.nn.Module, 
                    input: torch.Tensor, 
                    output: torch.Tensor) -> None:
                """
                    Pytorch hook that change the output by a
                    tensor hidden_states.
                    Note that passing through inplace operations
                    is mandatory to enable gradients flow.
                """
                output -= output
                output += hidden_states

            # Works only for BERT right now
            intermediate_layer = get_model_intermediate_layer(
                                        model = self.model,
                                        model_name = self.model_name,
                                        layer_num = layer_num,
                                        t5_part=self.config.T5_PART
                                        )
            hook_handle = intermediate_layer.register_forward_hook(_custom_hook)
            #hook_handle = self.model.bert.encoder.layer[layer_num].intermediate.register_forward_hook(_custom_hook)

            res = self.model(
                    input_ids = input_ids.repeat(n_steps, 1),
                    attention_mask = attention_mask.repeat(n_steps, 1)
                    ) # fact has a new first dim which is the discretization to compute the int

            hook_handle.remove()

            outputs = res.logits[torch.arange(res.logits.shape[0]),
                                 target_pos.repeat(n_steps),
                                 target.flatten().repeat(n_steps)] # Get only attribution of the target_pos
            
            return outputs

        # Gradients will be used so ensure they are 0
        self.model.zero_grad()
        
        # Get the activation of the desired layer
        la = LayerActivation(
                        self.model, 
                        get_model_intermediate_layer(
                            model = self.model,
                            model_name = self.model_name,
                            layer_num = layer_num,
                            t5_part=self.config.T5_PART
                            )
                        )
        hidden_states = la.attribute(
                            input_ids,
                            additional_forward_args = attention_mask
                            ) # Shape [Batch_size, L, d]
        
        # This one is pure security as LayerActivation shall not modify gradient
        self.model.zero_grad()
        
        # Compute Integrated Gradient using the forward_pos_func defined above
        # and attributiong the target layer activations
        ig = IntegratedGradients(forward_pos_func)
        attr = ig.attribute(hidden_states, n_steps = n_steps)
        
        # For some reasons, OPT has hidden dim [batch_size*length, hidden_dim]
        # whereas BERT has hidden dim [batch_size, length, hidden_dim] for instance
        if 'opt' in self.model_name:
            attr = attr.view(target.shape[0], -1, attr.shape[-1])
            
        return attr[torch.arange(target_pos.shape[0]), target_pos, :]
    
    
    def eval_one_rela_by_uuid(
                        self,
                        predicate_id: str,
                        mode: str = None,
                        rela_kns: dict[str, List[Tuple[int, int]]] = None,
                        correct_category: bool = False,
                        custom_dataset: List[str] = None,
                        db_fact: float = 2.
                        ) -> Tuple[ Dict[str, Dict[str, float]], Dict[str, List[float]]]:
        """

            ...

        """
        self.model.eval()

        assert not(mode) or rela_kns

        # Get Dataset
        if custom_dataset:
            dataset = custom_dataset
        else:
            dataset = self.data[predicate_id]
        
        if dataset is None:
            return None

        if correct_category:
            Ys_ids = []
            for uuid in dataset.keys():
                Y = dataset[uuid]['Y']
                _Y_ids = self.tokenizer(Y, add_special_tokens=False).input_ids
                assert len(_Y_ids) == 1
                Ys_ids.append(_Y_ids[0])
            Ys_ids = torch.tensor(Ys_ids)

        # Compute T-Rex Scores
        scores = {}
        probs = {}
        softmax = torch.nn.Softmax(dim=1)
        for uuid in tqdm.tqdm(dataset.keys(), total=len(dataset)):
            
            # Get template Tokens & Y 
            sentences_tok = dataset[uuid]['sentences_tok']    
            Y = dataset[uuid]['Y']
            
            # Target Tokenization
            target = self.tokenizer([Y]*len(sentences_tok), 
                                    return_tensors = 'pt',
                                    add_special_tokens=False)['input_ids'].to(self.device) # No need to pas

            # Pad & Create Attention Mask
            input_ids, attention_mask = pad_input_ids(sentences_tok)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Get target_pos
            if self.is_autoregressive:
                target_pos_i = torch.arange(input_ids.shape[0])
                target_pos_j  = attention_mask.sum(dim=-1)-1 # that's a neat trick
            else:
                # assert MLM?
                target_pos_i, target_pos_j = torch.where(input_ids == self.tokenizer.mask_token_id)
            
            
            # Register hooks
            if mode:
                # Get KNs
                if isinstance(rela_kns, dict):
                    _kns = rela_kns[uuid]
                else:
                    assert isinstance(rela_kns, list)
                    _kns = rela_kns 
                if len(_kns) == 0:
                    continue # skip uuid if there are no kns
                    
                hook_handles = self.register_nks_hooks(
                                            _kns,
                                            mode = mode,
                                            mask_pos = (target_pos_i, target_pos_j),
                                            batch_size = input_ids.shape[0],
                                            sequence_length = input_ids.shape[1],
                                            db_fact = db_fact,
                                            num_neurons = get_intermediate_dim(self.model, self.model_name, t5_part = self.config.T5_PART)
                                            )

            # Forward pass
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, 
                                    attention_mask=attention_mask).logits

            # Remove hooks
            if mode:
                KnowledgeNeurons.remove_hooks(hook_handles)

            _, ids = torch.topk(logits[target_pos_i, target_pos_j], k = 100)
            ids = ids.cpu()
            target = target.cpu()

            scores[uuid] = {'P@1': 0,
                            'P@5': 0,
                            'P@20': 0,
                            'P@100': 0}
            scores[uuid]['P@1'] = ((target[:] == ids[:,:1]).any(axis = 1)).sum().item()
            scores[uuid]['P@5'] = ((target[:] == ids[:,:5]).any(axis = 1)).sum().item()
            scores[uuid]['P@20'] = ((target[:] == ids[:,:20]).any(axis = 1)).sum().item()
            scores[uuid]['P@100'] = ((target[:] == ids[:,:100]).any(axis = 1)).sum().item()
            
            if correct_category:
                scores[uuid]['ccp@1'] = 0.
                scores[uuid]['ccp@5'] = 0.
                scores[uuid]['ccp@20'] = 0.
                scores[uuid]['ccp@100'] = 0.
                # Number of answers of the model in the right category
                # It supposes that no further filtration will be made ie. Ys are 1 token long
                for l in range(ids.shape[0]):
                    scores[uuid]['ccp@1'] += tensors_intersection_size(ids[l,:1], Ys_ids)/1
                    scores[uuid]['ccp@5'] += tensors_intersection_size(ids[l,:5], Ys_ids)/5
                    scores[uuid]['ccp@20'] += tensors_intersection_size(ids[l,:20], Ys_ids)/20
                    scores[uuid]['ccp@100'] += tensors_intersection_size(ids[l,:100], Ys_ids)/100
            
            assert ids.shape[0] == len(sentences_tok)
            scores[uuid] = {k:v/len(sentences_tok) for k,v in scores[uuid].items()}
            
            output_probs = softmax(logits[target_pos_i, target_pos_j].cpu().float())
            probs[uuid] = output_probs[torch.arange(output_probs.shape[0]), target.flatten()].tolist()

        return scores, probs
    
    
    @staticmethod
    def remove_hooks(hook_handles: List[torch.utils.hooks.RemovableHandle]) -> None:
        """
            Remove hooks from the model.
        """
        for hook_handle in hook_handles:
            hook_handle.remove()

    def register_nks_hooks(
                        self,
                        kns: List[Tuple[int,int]],
                        mode: str,
                        mask_pos: Tuple[torch.Tensor, torch.Tensor],
                        batch_size: int,
                        sequence_length: int,
                        num_neurons: int = 3072,
                        db_fact: float = 2.) -> List[torch.utils.hooks.RemovableHandle]:
        """
            Register hooks in the second layer of some transfomer's encoder FFN.
            
            The encoders l that will be affected are defined in kns parameters that contains the
            knowledge neurons.
            
            This hook multiplies the activations of n^th neuron of this layer by 2 if mode = 'db'
            and by 0 if mode = 'wo'. n here is defined in kns.
            
            Overall kns = [..., (l,n), ...]
            
            TBD: get num_neurons from the model parameters directly.
            
            Args:
                kns (list) contains the knowledge neurons like (layer num, neuron num)
                mode (str) enhance (db) or erase (wo) knowledge
                mask_pos (tuple) this activation modification only applies at the mask pos
                batch_size (int) usefull to define fact
                squence_length (int) usefull to define fact
                num_neurons (int) cf. TBD
            Returns:
                hook_handles (list) needed to remove hooks, otherwise everything is broken!
        
        """
        assert mode in ['wo', 'db']

        # Get neurons by layer
        layer2neurons = {}
        for kn in kns:
            layer, neuron = kn
            if layer in layer2neurons.keys():
                layer2neurons[layer].append(neuron)
            else:
                layer2neurons[layer] = [neuron]

        # Get by what we will multiply KNs activations
        layer2fact = {}
        for layer in layer2neurons.keys():
            fact = 1.*torch.ones(batch_size, sequence_length, num_neurons)
            for neuron in layer2neurons[layer]:
                if mode == 'wo':
                    fact[mask_pos[0], mask_pos[1], neuron] = 0.
                elif mode == 'db':
                    fact[mask_pos[0], mask_pos[1], neuron] = db_fact
                
            if 'opt' in self.model_name:
                # OPT is running a hidden dim which is [batch_size * length, hidden_dim]
                fact = fact.view(batch_size * sequence_length, num_neurons)
                
            layer2fact[layer] = fact.to(self.device)

        # Create Custom Hooks
        def _hook_template(
                        module: torch.nn.Module, 
                        input: torch.Tensor, 
                        output: torch.Tensor, 
                        fact: torch.Tensor
                        ) -> None:
            """
                Modify the output by multiplying them by fact.
                
                Note that a Pytorch hook can only takes module,
                input and output as arguments so we used a trick
                here by defining this "hook template" and by 
                registering the hook using a lambda function that
                get rid of the fact argument.
            """
            output *= fact
            
        hook_handles = []
        for layer in layer2fact.keys():
            intermediate_layer = get_model_intermediate_layer(
                                        model = self.model,
                                        model_name = self.model_name,
                                        layer_num = layer,
                                        t5_part=self.config.T5_PART
                                        )
            hook_handle = intermediate_layer.register_forward_hook(
                                        lambda module,input,output,fact=layer2fact[layer]: _hook_template(module, input, output, fact)
                                        )
            hook_handles.append(hook_handle)

        return hook_handles
    
    @staticmethod
    def _overlap_metrics(kns_dict_1: Dict[str, List[ Tuple[float, float]]],
                         kns_dict_2: Dict[str, List[ Tuple[float, float]]]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
            For each uuid get its KNs and compute the overlap.
            
            Returns the average.
        
        """
        overlap = 0.
        prop_overlap = 0. # TBD
        layer_kns_1, layer_kns_2 = {}, {}
        seen_kns_1, seen_kns_2 = set(), set()
        seen_overlap_kns = set()
        layer_overlap_kns = {}
        num_kns_1 = 0.
        num_kns_2 = 0.
        for uuid in kns_dict_1.keys():
            assert uuid in kns_dict_2.keys()
            
            kns1, kns2 = kns_dict_1[uuid], kns_dict_2[uuid]
            kns1 = [(e[0], e[1]) for e in kns1]
            kns2 = [(e[0], e[1]) for e in kns2]
            
            # Get layer
            for kn in kns1:
                layer,_=kn
                if kn not in seen_kns_1:
                    if layer in layer_kns_1.keys():
                        layer_kns_1[layer] += 1
                    else:
                        layer_kns_1[layer] = 1
                seen_kns_1.add(kn)
            for kn in kns2:
                layer,_=kn
                if kn not in seen_kns_2:
                    if layer in layer_kns_2.keys():
                        layer_kns_2[layer] += 1
                    else:
                        layer_kns_2[layer] = 1
                seen_kns_2.add(kn)
            
            # num kns
            num_kns_1 += len(kns1)
            num_kns_2 += len(kns2)
            # overlap
            kns_overlap = set(kns1).intersection(set(kns2))
            
            # Overlap KNs layer
            for kn in kns_overlap:
                layer, _ = kn
                if kn not in seen_overlap_kns: # don't count the same kn twice 
                    if layer in layer_overlap_kns.keys():
                        layer_overlap_kns[layer] += 1
                    else:
                        layer_overlap_kns[layer] = 1
                seen_overlap_kns.add(kn)
            
            # Here we compute the size of the overlap
            # We could also compute the proportion of KNs shared
            overlap += len(kns_overlap)
            
        return {'overlap': overlap/len(kns_dict_1),
                'num_kns_1': num_kns_1/len(kns_dict_1),
                'num_kns_2': num_kns_2/len(kns_dict_2),
                'layer_kns_1': layer_kns_1,
                'layer_kns_2': layer_kns_2,
                'layer_overlap_kns': layer_overlap_kns}
        
        
    def _kns_analysis(self,
                      pararel_kns_dict: Dict[str, List[ Tuple[float, float]]],
                      autoprompt_kns_dict: Dict[str, List[ Tuple[float, float]]]):
        """
        
        key: uuid
        value: list of KNs [layer, neuron]
        
        """
        
        
        
        ####### SEMANTICS & SYNTAX KNS ########
        # Here the dict X_kns_dict containes the KNs of one relation ships for every
        # instantiations. Be intersacting up to a threshold we are only retaining KNs
        # encoding "capital of" for instance.
        
        # Doing Coarse First
        _pararel_sem_syn_kns = self._sem_syn_kns(pararel_kns_dict)
            
        _autoprompt_sem_syn_kns = self._sem_syn_kns(autoprompt_kns_dict)
        
        # Refining by intersecting with autoprompt
        _refined_sem_kns, _pararel_refined_syn_kns, _autoprompt_refined_syn_kns = self._refined_semantics_kns(_pararel_sem_syn_kns, _autoprompt_sem_syn_kns)
        
        
        ####### KNOWLEDGE KNS ########
        # We get them by removing from KNs the SemKNs U SynKNs
        
        _pararel_know_kns = self._knowledge_kns(pararel_kns_dict, _pararel_sem_syn_kns)
        _autoprompt_know_kns = self._knowledge_kns(autoprompt_kns_dict, _autoprompt_sem_syn_kns)
        
        # Get shared know_kns
        _shared_know_kns, _pararel_only_know_kns, _autoprompt_only_know_kns = set(), set(), set()
        _shared_know_kns_uuid, _pararel_only_know_kns_uuid, _autoprompt_only_know_kns_uuid = {}, {}, {}
        _thresh = find_closest_elem(list(_refined_sem_kns.keys()), self.config.ACCROSS_UUIDS_THRESHOLD)
        for uuid in _pararel_know_kns.keys():
                _shared_know_kns_uuid[uuid] = []
                _pararel_only_know_kns_uuid[uuid] = []
                for kn in _pararel_know_kns[uuid][_thresh]:
                    if kn in _autoprompt_know_kns[uuid][_thresh]:
                        _shared_know_kns.add(kn)
                        _shared_know_kns_uuid[uuid].append(kn)
                    else:
                        _pararel_only_know_kns.add(kn)
                        _pararel_only_know_kns_uuid[uuid].append(kn)
        for uuid in _autoprompt_know_kns.keys():
                _autoprompt_only_know_kns_uuid[uuid] = []
                for kn in _autoprompt_know_kns[uuid][_thresh]:
                    if kn in _pararel_know_kns[uuid][_thresh]:
                        assert kn in _shared_know_kns
                    else:
                        _autoprompt_only_know_kns.add(kn)
                        _autoprompt_only_know_kns_uuid[uuid].append(kn)
        
        
        ###### KNs CATEGORY ######
        
        category_layer2count = self._kns_category(
                                    pararel_kns_dict, 
                                    autoprompt_kns_dict,
                                    _refined_sem_kns,
                                    _pararel_refined_syn_kns,
                                    _autoprompt_refined_syn_kns
                                    )
        
        
        return {'pararel_sem_syn_kns': _pararel_sem_syn_kns,
                'autoprompt_sem_syn_kns': _autoprompt_sem_syn_kns,
                'sem_kns': _refined_sem_kns,
                'pararel_syn_kns':  _pararel_refined_syn_kns,
                'autoprompt_syn_kns': _autoprompt_refined_syn_kns,
                'shared_know_kns_set': _shared_know_kns,
                'pararel_only_know_kns_set': _pararel_only_know_kns,
                'autoprompt_only_know_kns_set': _autoprompt_only_know_kns,
                'shared_know_kns': _shared_know_kns_uuid,
                'pararel_only_know_kns': _pararel_only_know_kns_uuid,
                'autoprompt_only_know_kns': _autoprompt_only_know_kns_uuid,
                'category': category_layer2count}
    
    def _sem_syn_kns(self, kns_dict: Dict[str, List[ Tuple[float, float]]]) -> Dict[float, List[ Tuple[float, float] ]]:
        """
            This function averages out the instantiation of KNs.
            
            Syntax was partly removed along with noise during the refining.
            But a part of syntax remains.
        
        """
        
        thresholds = np.linspace(0, 1, 21)
        
        num_uuids = len(kns_dict)
        
        neuron2uuid_num = {}
        for _, kns in kns_dict.items():
            for kn in kns:
                tuple_kn = (kn[0], kn[1]) # Make it hashable
                if tuple_kn in neuron2uuid_num.keys():
                    neuron2uuid_num[tuple_kn] += 1
                else:
                    neuron2uuid_num[tuple_kn] = 1 
        
        res= {}
        for thresh in thresholds:
            thresh_sem_syn_kns = [k for k,v in neuron2uuid_num.items() if v >= num_uuids*thresh]
            res[thresh] = thresh_sem_syn_kns
            
        return res
    
    def _refined_semantics_kns(self,
                               pararel_kns: Dict[float, List[Tuple[float, float]]],
                               autoprompt_kns: Dict[float, List[Tuple[float, float]]]) -> Tuple[Dict[float, List[Tuple[float, float]]]]:
        
        refined_sem_kns = {}
        refined_syn_kns_pararel = {}
        refined_syn_kns_autoprompt = {}
        for thresh in pararel_kns.keys():
            _sem_syn_kns_pararel = pararel_kns[thresh]
            _sem_syn_kns_autoprompt =autoprompt_kns[thresh]
            
            _sem_syn_pararel_set = set(_sem_syn_kns_pararel)
            _sem_syn_autoprompt_set = set(_sem_syn_kns_autoprompt)
            
            # Get semantics
            refined_sem_kns[thresh] = list(_sem_syn_autoprompt_set.intersection(_sem_syn_pararel_set))
            
            # Get syntax
            refined_syn_kns_pararel[thresh] = list(_sem_syn_pararel_set.difference(refined_sem_kns[thresh]))
            refined_syn_kns_autoprompt[thresh] = list(_sem_syn_autoprompt_set.difference(refined_sem_kns[thresh]))
            
        return refined_sem_kns, refined_syn_kns_pararel, refined_syn_kns_autoprompt
    
    def _knowledge_kns(self,
                       full_kns: Dict[str, List[ Tuple[float, float]]], 
                       sem_syn_kns: Dict[float, List[Tuple[float, float]]]) -> Dict[str, Dict[float, List[Tuple[float, float]]]]:
                           
        kkns = {}
        for uuid in full_kns.keys():
            
            # Get KNs
            _kns = full_kns[uuid] 
            hashable_kns = [(e[0], e[1]) for e in _kns] 
            kkns_by_thresh = {}
            for thresh, _sem_syn_kns in sem_syn_kns.items():
                kkns_by_thresh[thresh] = list(set(hashable_kns).difference(set(_sem_syn_kns)))
            
            kkns[uuid] = kkns_by_thresh
            
        return kkns
    
    def _sem_kns2rela(self, 
                      res_dict: dict, 
                      thresholds: List[float], 
                      rela_names: List[str]) -> Dict[float, Dict[Tuple[float, float], Tuple[List[str], List[str]]]]:
        res_by_thresh = {}
        for thresh in thresholds:
            res = {}
            for rela in rela_names:
                _sem_kns = res_dict[rela]['sem_kns'][thresh] 
                for kn in _sem_kns:
                    if kn not in res.keys():
                        res[kn] = ([rela], [TREX_CATEGORIES[rela]])
                    else:
                        res[kn][0].append(rela)
                        res[kn][1].append(TREX_CATEGORIES[rela])
            res_by_thresh[thresh] = res
            
        return res_by_thresh
    
    def _sem_syn_know_dist(self,
                           res_dict: dict,
                           pararel_kns: Dict[str, Dict[str, List[ Tuple[float, float]]]],
                           autoprompt_kns: Dict[str, Dict[str, List[ Tuple[float, float]]]],
                           thresholds: List[float],
                           rela_names: List[str]) -> ...:
        pararel_res, autoprompt_res = [], []
        pararel_res_by_rela, autoprompt_res_by_rela = {}, {}
        for rela in rela_names:
            pararel_res_by_uuid, autoprompt_res_by_uuid = [], []
            for uuid in pararel_kns[rela].keys():
                pararel_res_by_thresh, autoprompt_res_by_thresh = [], []
                for thresh in thresholds:
                    if thresh != find_closest_elem(thresholds, self.config.ACCROSS_UUIDS_THRESHOLD):
                        continue # /!\
                    # Get SEM,SYN & KNOW KNs #
                    _sem_kns = res_dict[rela]['sem_kns'][thresh]
                    
                    _pararel_syn_kns = res_dict[rela]['pararel_syn_kns'][thresh]
                    _autoprompt_syn_kns = res_dict[rela]['autoprompt_syn_kns'][thresh]
                    
                    _pararel_only_know_kns = res_dict[rela]['pararel_only_know_kns'][uuid]
                    _autoprompt_only_know_kns = res_dict[rela]['autoprompt_only_know_kns'][uuid]
                    _shared_know_kns = res_dict[rela]['shared_know_kns'][uuid]
                    
                    # Get KNs #
                    _pararel_kns = pararel_kns[rela][uuid]
                    _autoprompt_kns = autoprompt_kns[rela][uuid]
                    
                    _pararel_kns = [(kn[0], kn[1]) for kn in _pararel_kns]
                    _autoprompt_kns = [(kn[0], kn[1]) for kn in _autoprompt_kns]
                    
                    # Get know kns division
                    # Pararel POV
                    pararel_only_n_know_kns = len(_pararel_only_know_kns)
                    shared_n_know_kns = len(_shared_know_kns)
                            
                    # Autoprompt POV
                    autoprompt_only_n_know_kns = len(_autoprompt_only_know_kns)
                    
                    
                    # Compute Num #
                    pararel_n_kns = len(_pararel_kns)
                    pararel_n_sem_kns = len(set(_pararel_kns).intersection(set(_sem_kns)))
                    pararel_n_syn_kns = len(set(_pararel_syn_kns).intersection(set(_pararel_kns)))
                    assert pararel_n_kns == pararel_only_n_know_kns + shared_n_know_kns + pararel_n_sem_kns + pararel_n_syn_kns
                    
                    autoprompt_n_kns = len(_autoprompt_kns)
                    autoprompt_n_sem_kns = len(set(_sem_kns).intersection(set(_autoprompt_kns)))
                    autoprompt_n_syn_kns = len(set(_autoprompt_syn_kns).intersection(set(_autoprompt_kns)))
                    assert autoprompt_n_kns == autoprompt_only_n_know_kns + shared_n_know_kns + autoprompt_n_sem_kns + autoprompt_n_syn_kns
                    
                    # Store #
                    pararel_res_by_thresh.append([pararel_n_sem_kns, pararel_n_syn_kns, pararel_only_n_know_kns, shared_n_know_kns])
                    autoprompt_res_by_thresh.append([autoprompt_n_sem_kns, autoprompt_n_syn_kns, autoprompt_only_n_know_kns, shared_n_know_kns])
                    
                pararel_res_by_uuid.append(pararel_res_by_thresh)
                autoprompt_res_by_uuid.append(autoprompt_res_by_thresh)
                
                pararel_res.append(pararel_res_by_thresh)
                autoprompt_res.append(autoprompt_res_by_thresh)

            pararel_res_by_uuid = np.array(pararel_res_by_uuid) # Shape [n_uuid, n_thresh, 4]
            autoprompt_res_by_uuid = np.array(autoprompt_res_by_uuid)
            
            pararel_res_by_rela[rela] = pararel_res_by_uuid.mean(axis=0) # shape [n_thresh, 4]
            autoprompt_res_by_rela[rela] = autoprompt_res_by_uuid.mean(axis=0) 
            
        
        pararel_res_arr = np.array(pararel_res) 
        autoprompt_res_arr = np.array(autoprompt_res)
        # Mean
        pararel_res = pararel_res_arr.mean(axis = 0) # shape [n_thresh, 4] but in fact n_thresh = 1 here 
        autoprompt_res = autoprompt_res_arr.mean(axis = 0)
        # Std
        pararel_res_se = pararel_res_arr.std(axis = 0)/np.sqrt(pararel_res_arr.shape[0]) # shape [n_thresh, 4]
        autoprompt_res_se = autoprompt_res_arr.std(axis = 0)/np.sqrt(autoprompt_res_arr.shape[0])

        return (pararel_res, autoprompt_res, pararel_res_se, autoprompt_res_se, pararel_res_by_rela, autoprompt_res_by_rela)
    
    
    def _kns_category(
                    self,
                    pararel_kns_dict: Dict[str, List[ Tuple[float, float]]],
                    autoprompt_kns_dict: Dict[str, List[ Tuple[float, float]]],
                    sem_kns: Dict[float, List[ Tuple[float, float] ] ],
                    pararel_syn_kns: Dict[float, List[ Tuple[float, float] ] ],
                    autoprompt_syn_kns: Dict[float, List[ Tuple[float, float] ] ],
                    ) -> Dict[float, Dict[str, Dict[int, float]]]:
        """
        
        
            ### OLD VERSION PROBABLY ###
        
            We have categories:
                (i) Shared Across Instantiations (of its relevant Relation) of its language
                (ii) Shared Across Languages (for now only English-Autoprompt) it exists an 
                     uuid such that it belongs to ParaRel & Autoprompt 
                     
            RK: Here it is possible that a neuron verifies (i) for ParaRel but not for Autoprompt
            so it will NOT be a Semantics KNs as describe above. But if it exists only one uuid
            where it is shared with Autoprompt, here it will count as one (see below). 
            So there's a tension. This will reduce the amount of Syntax KNs as a consequences.
            Knowledge KNs will be kept unchanged. BUT we could imagine that AutoPrompt & ParaRel 
            finds different Semantics KNs.
                     
                
            This way: 
                - Semantics = (i) + (ii)
                - Syntax = (i) + Not(ii)
                - Knowledge Multilingual = Not(i) + (ii)
                - Knowledge Language Specific = Not(i) + Not(ii)
                
            As all of this requires thresholds to work we will just keep the proportion of
            everything and decide later which threshold apply to which.
            
            Example: (8, 1024) (read from P19) belongs on (i) 84% of the instantiations of P19
                                                          (ii) for language 1 (e.g. Autoprompt) was shared in 76% of the time
                                                          
            ### NEW VERSION ###
            
        
            All semantics KNs will be shared (hence hatch or no)
            None of the Syntax KNs will be shared
            A KN can be Syntax for autoprompt & Knowledge for Pararel so some KN will be counted twice
        
        """
        
        ### Here one rela only ###
        
        ### GET EACH KN PROPERTIES ###
        
        res = {}
        # We'll do threshold by threshold 
        for thresh in sem_kns.keys():
            
            # Get Semantics & Syntax KNs
            _sem_kns = sem_kns[thresh]
            _pararel_syn_kns = pararel_syn_kns[thresh]
            _autoprompt_syn_kns = autoprompt_syn_kns[thresh]
            
            # We define each neuron properties
            _res = {}
            for uuid in pararel_kns_dict.keys():
                _pararel_uuid_kns = [(kn[0], kn[1]) for kn in pararel_kns_dict[uuid]]
                _autoprompt_uuid_kns = [(kn[0], kn[1]) for kn in autoprompt_kns_dict[uuid]] # Need to be hashable
                
                # Check Pararel KNs
                for kn in _pararel_uuid_kns:
                    _sem = kn in _sem_kns
                    _syn = kn in _pararel_syn_kns
                    if kn in _res.keys():
                        _res[kn]['pararel'] = True
                        _res[kn]['pararel_syn'] = _syn
                        _res[kn]['pararel_know'] = not(_sem or _syn)
                    else:
                        _res[kn] = {'pararel': True,
                                    'sem': _sem, # Semantics KNs are all shared so we'll have pararel = autoprompt = True & pararel_syn = autoprompt_syn = False (idem Knowledge)
                                    'pararel_syn': _syn, # It is not possible for pararel syntax & autoprompt syntax to be both True (else they would be semantics KNs!)
                                    'autoprompt_syn': False,
                                    'pararel_know': not(_sem or _syn),
                                    'autoprompt_know': False,
                                    'autoprompt': False}
                
                # Check Autoprompt KNs        
                for kn in _autoprompt_uuid_kns:
                    _sem = kn in _sem_kns
                    _syn = kn in _autoprompt_syn_kns
                    if kn in _res.keys():
                        _res[kn]['autoprompt'] = True
                        _res[kn]['autoprompt_syn'] = _syn
                        _res[kn]['autoprompt_know'] = not(_sem or _syn)
                    else:
                        _res[kn] = {'pararel': False,
                                    'sem': _sem, 
                                    'pararel_syn': False, 
                                    'autoprompt_syn': _syn,
                                    'pararel_know': False,
                                    'autoprompt_know': not(_sem or _syn),
                                    'autoprompt': True}
                
                
            ### NOW PROPORTION OF EACH ###
            
            sem_layer2count = {k: 0 for k in range(self.model_layers_num[self.model_name])}
            pararel_syn_layer2count = {k: 0 for k in range(self.model_layers_num[self.model_name])}
            autoprompt_syn_layer2count = {k: 0 for k in range(self.model_layers_num[self.model_name])}
            pararel_know_layer2count = {k: 0 for k in range(self.model_layers_num[self.model_name])}
            autoprompt_know_layer2count = {k: 0 for k in range(self.model_layers_num[self.model_name])}
            pararel_autoprompt_know_layer2count = {k: 0 for k in range(self.model_layers_num[self.model_name])}
            
            sem_count, syn_count, know_count = 0, 0, 0
            
            for kn, props in _res.items():
                layer, _ = kn
                if props['sem']:
                    assert not(props['pararel_syn'] or props['autoprompt_syn'] or props['pararel_know'] or props['autoprompt_know']) # supposed to be impossible so sanity check
                    sem_count += 1
                    sem_layer2count[layer] += 1
                    continue
                
                if props['pararel_syn']:
                    assert not(props['autoprompt_syn'])
                    assert not(props['pararel_know'])
                    syn_count += 1
                    pararel_syn_layer2count[layer] += 1
                elif props['autoprompt_syn']:
                    assert not(props['pararel_syn']) # useless because of elif but you know
                    assert not(props['autoprompt_know'])
                    syn_count += 1
                    autoprompt_syn_layer2count[layer] += 1
                    
                if props['pararel_know'] and props['autoprompt_know']:
                    know_count += 1
                    pararel_autoprompt_know_layer2count[layer] += 1
                elif props['pararel_know']:
                    know_count += 1
                    pararel_know_layer2count[layer] += 1
                elif props['autoprompt_know']:
                    know_count += 1
                    autoprompt_know_layer2count[layer] += 1
                
            
            # We want proportion
            """
            if not(sem_count == 0):
                sem_layer2count = {k: v/sem_count for k,v in sem_layer2count.items()}    
            if not(syn_count == 0):
                pararel_syn_layer2count = {k: v/syn_count for k,v in pararel_syn_layer2count.items()}
                autoprompt_syn_layer2count = {k: v/syn_count for k,v in autoprompt_syn_layer2count.items()}
            if not(know_count == 0):
                pararel_know_layer2count = {k: v/know_count for k,v in pararel_know_layer2count.items()}
                autoprompt_know_layer2count = {k: v/know_count for k,v in autoprompt_know_layer2count.items()}
                pararel_autoprompt_know_layer2count = {k: v/know_count for k,v in pararel_autoprompt_know_layer2count.items()}
            """ 
                
            res[thresh] = {
                        'sem': sem_layer2count,
                        'pararel_syn': pararel_syn_layer2count,
                        'autoprompt_syn': autoprompt_syn_layer2count,
                        'pararel_know': pararel_know_layer2count,
                        'autoprompt_know': autoprompt_know_layer2count,
                        'pararel_autoprompt_know': pararel_autoprompt_know_layer2count,
                        'sem_count': sem_count,
                        'syn_count': syn_count,
                        'know_count': know_count
                        }
            
        return res      
            
                        
                
                
            
            
                
                        
        