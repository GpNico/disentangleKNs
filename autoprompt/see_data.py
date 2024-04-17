"""
    Helper file to vizualize data (as it is saved ibn jsonl)
"""


import os
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    
    
    file_path = os.path.join(
        'data',
        'mlama1.1'
    )
    
    print('Processing...')
    for lang in os.listdir(file_path):
        if lang == 'README.md':
            continue
        print(f"\t{lang}")
        lang_path = os.path.join(file_path, lang)
        for file in os.listdir(lang_path):
            if '.jsonl' in file: # Check that it is indeed a file
                name = file.replace('.jsonl', '')
                folder_path = os.path.join(
                        lang_path,
                        name
                        )
                
                jsonObj = pd.read_json(
                    path_or_buf=os.path.join(
                        lang_path,
                        file
                        ), 
                    lines=True
                    )
                
                if len(jsonObj) == 0:
                    print(f"\t{file} is empty.")
                    continue
                
                try:
                    train, dev = train_test_split(jsonObj, test_size = 0.2)
                except:
                    print(jsonObj)
                    print(f"\tIssue with: {file}.")
                    continue
                
                train.reset_index(inplace = True)
                dev.reset_index(inplace = True)
                
                os.makedirs(
                    folder_path,
                    exist_ok=True
                )

                with open(os.path.join(folder_path, 'train.jsonl'), 'w') as f:
                    f.write(train.to_json(orient='records', lines=True))
                with open(os.path.join(folder_path, 'dev.jsonl'), 'w') as f:
                    f.write(dev.to_json(orient='records', lines=True))
                
                os.remove(
                    os.path.join(
                        lang_path,
                        file
                        )
                    )
    
    