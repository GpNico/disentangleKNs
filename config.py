
import os

class Config:
    
    ### CONFIG ###
    
    WANDB = True
    
    PLOT_OLD_FIGS = False
    PLOT_ERROR_BARS = True
    MODELS_TO_PLOT = ['bert-base-uncased', 'bert-large-uncased', 'opt-350m', 'opt-6.7b', 'Llama-2-7b-hf']
    MULTILINGUAL_MODELS_TO_PLOT = ['bert-base-multilingual-uncased', 'flan-t5-xl']
    
    # Data
    PATH_TO_MLAMA = os.path.join('data', 'mlama1.1')
    PATH_TO_MPARAREL = os.path.join('data', 'mpararel')
    PATH_TO_TREX = os.path.join('data', 'trex')
    PATH_TO_AUTOREGRESSIVE_PARAREL = os.path.join('data', 'autoregressive_pararel')
    PATH_TO_SAVED_PROMPTS = os.path.join('data', 'autoprompt')
    PROMPT_MIN_PRECISION = 0.1 # Like Marco
    SINGLE_TOKEN_TARGET = True # /!\
    MIN_N_PROMPTS = 5
    
    # Knowledge Neurons
    PATH_TO_KNS_DIR = os.path.join('results', 'knowledge_neurons')
    KNS_BATCH_SIZE = 16
    T_THRESH = 0.2 # paper 0.2
    P_THRESHS = [0.5, 0.6, 0.7, 0.8, 0.9] # Paper 0.7 (SToring multiple p thresholds)
    P_THRESH = 0.7 # the one chosen afterwards (maybe chose one per model)
    T5_PART = 'encoder' # T5 has a encoder-decoder architechture so we could try both
    
    ACCROSS_UUIDS_THRESHOLD = 0.7 # sem_syn_kns thresholds
    TRIVIAL_PROMPT_ACTIVATION_FACT = 10.
    
    # TREx
    ACCURACY_RANKS = [1,5,10,20,50,100] # ks in P@k
    
    # MultiLingual
    LANGS = ['en', 'fr', 'es', 'ca', 'da', 'de', 'fi', 'it', 'nl', 'pl', 'pt', 'ru', 'sv']
    
    # Autoprompt
    N_SEEDS = 10
    N_TRIGGER_TOKENS = 5
    AUTOPROMPT_ITERS = 500
    AUTOPROMPT_BATCH_SIZE = 56 # default
    
    
    ### DEBUG CONFIG ###
    # Made to be faster when testing code
    
    DEBUG_WANDB = False
    
    # Data
    DEBUG_MIN_N_PROMPTS = 4
    
    # TREx
    DEBUG_ACCURACY_RANKS = [1,5,20,100]
    
    # MultiLingual
    DEBUG_LANGS = ['es', 'en', 'fr']
    
    # Autoprompt
    DEBUG_N_SEEDS = 4
    DEBUG_AUTOPROMPT_ITERS = 200
    
    
    def __init__(self, debug: bool) -> None:
        if not(debug):
            return
        
        # Change attributes to DEBUG_ attributes
        for attr, value in Config.__dict__.items():
            if '__' in attr:
                continue
            if 'DEBUG' in attr:
                attr_name = attr[6:]
                setattr(Config, attr_name, value)
