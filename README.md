# RE_improved_baseline

Code for technical report "[An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373)".

## Requirements
* torch >= 1.8.1
* transformers >= 3.4.0
* wandb
* ujson
* tqdm

The Pytorch version must be at least 1.8.1 as our code relies on the both the ``torch.cuda.amp`` and the ``torch.utils.checkpoint``, which are introduced in the 1.8.1 release.

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
The commands and hyper-parameters for running experiments can be found in the ``scripts`` folder. For example, to train roberta-large, run
```bash
>> sh run_roberta_tacred.sh    # TACRED and TACREV
>> sh run_roberta_retacred.sh  # Re-TACRED
```
The evaluation results are synced to the wandb dashboard. The results on TACRED and TACREV can be obtained in one run as they share the same training set.

For all tested pre-trained language models, training can be conducted with one RTX 2080 Ti card.
