# Distillation_QA_benchmark



## Description 

We aim to benchmark various distillation algorithms for QA and NER tasks. 

Code for experiements can be put in directory /src.

## Experiments

You can do KD experiments now using our code [here](./src/Experiments/README.md). We only support experiments on QA tasks now but I am very confident that we will support NER tasks soon.

## Dataset

For QA task, you can download SQuAD dataset by using gluon-nlp command:

``` shell
nlp_data prepare_squad --version 1.1
```

For downloading other datasets, follow this: https://github.com/dmlc/gluon-nlp/tree/master/scripts/datasets/question_answering

If you want to download all data of NaturalQuestions(42G), you can check out the official statement of NaturalQuestions here: https://ai.google.com/research/NaturalQuestions/download



## TO DOs

1. Benchmark DynaBERT, TinyBERT in NER and QA task.