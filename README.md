# A Typology of Knowledge Neurons, and its Application to Shared Knowledge in Multilingual Models

This repo is associated with the paper *A Typology of Knowledge Neurons, and its Application to Shared Knowledge in Multilingual Models* submitted to **NeurIPS 2024**. 

## `mParaRel`

To access the `mParaRel` dataset in the `data/mpararel` folder.

The dataset creation code is in the `translate_pararel.py` file. You can use it as follows:

```
python translate_pararel.py --lang fr
```

## MonoLingual Experiments

To run multilingual experiments, run:
```
./run_monolingual.sh
```
Give execute permission to the script beforehand:
```
chmode +x ./run_monolingual.sh
```
By default, the model is `bert-base-uncased`, but you can edit `run_monolingual.sh` and change the `MODEL_NAME` variable to: 
- `bert-base-uncased`
- `bert-large-uncased`
- `opt-350m`
- `opt-6.7b`
- `Llama-2-7b-hf`

You can uncomment the line:
```
python main.py --model_name $MODEl_NAME --dataset trex --run_autoprompt
```
to launch AutoPrompt seed calculation (but 10 seeds are already contained in the data folder). 

## MultiLingual Experiments

To run multilingual experiments, run:
```
./run_multilingual.sh
```
Give execute permission to the script beforehand:
```
chmode +x ./run_multilingual.sh
```
You can also uncomment the line:
```
python main.py --model_name $MODEl_NAME --dataset mlama --run_autoprompt
```
to launch AutoPrompt seed calculation (but 10 seeds are already contained in the data folder). to calculate AutoPrompt seeds for all 10 languages (but 10 seeds are contained in the data folder). 

## AutoPrompt

The folder `autoprompt` contains the code from the [official AutoPrompt repo](https://github.com/ucinlp/autoprompt/) slightly modified to include more models. 

## Remark

Note that the previous code was run on NVIDIA Tesla A100 GPUs for models with over 500 million parameters, and on NVIDIA Tesla V100 GPUs for the others. In addition, the code is highly parallelizable, so each AutoPrompt seed or relation can be computed independently. The code has been run on a cluster to take advantage of this, however the code given is designed to be run on a single GPU. Expect around 24 hours of computation time per model.
