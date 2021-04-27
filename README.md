# RE_improved_baseline

Code for technical report [An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373).

## Requirements
* torch >= 1.8.1
* transformers >= 3.4.0
* wandb
* ujson
* tqdm

## Dataset
The TACRED dataset can be obtained from [this link](https://nlp.stanford.edu/projects/tacred/). The TACREV and Re-TACRED dataset can be obtained following the instructions in [tacrev](https://github.com/DFKI-NLP/tacrev) and [Re-TACRED](https://github.com/gstoica27/Re-TACRED), respectively. The expected structure of files is:
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

## Training and Evaluation
