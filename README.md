# RE_improved_baseline

Code for report [An Improved Baseline for Sentence-level Relation Extraction ](https://arxiv.org/abs/2102.01373).

## Requirements
* torch >= 1.8.1
* transformers >= 3.4.0
* wandb
* ujson
* tqdm

## Dataset
The expected structure of files is:
```
RE_improved_baseline
 |-- dataset
 |    |-- tacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- dev_rev.json
 |    |    |-- test_rev.json
 |    |-- retacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
```
