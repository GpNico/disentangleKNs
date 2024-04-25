
import argparse
from src.utils import sort_by_seeds, load_config


if __name__ == '__main__':
    
    # Argparse
    parser = argparse.ArgumentParser()
    
    # Model & Data
    parser.add_argument('--model_name', 
                        default='all',
                        type=str, 
                        help="model we want to sort the seeds.")
    args = parser.parse_args()
    
    # Config
    config = load_config(debug = False)
    
    # Sort
    sort_by_seeds(model_name = args.model_name, config = config)