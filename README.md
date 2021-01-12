# Distillation_QA_benchmark



## Description 

We aim to benchmark various distillation algorithms for QA and NER tasks. 

Code for experiements can be put in directory /src.

## Dataset

For QA task, you can download SQuAD dataset by using gluon-nlp command:

``` shell
nlp_data prepare_squad --version 1.1
```

For downloading other datasets, follow this: https://github.com/dmlc/gluon-nlp/tree/master/scripts/datasets/question_answering

If you want to download all data of NaturalQuestions(42G), you can check out the official statement of NaturalQuestions here: https://ai.google.com/research/NaturalQuestions/download



## TO DOs

1. I am going to develop a toolkit which can be used to do KD related experiments easily and support customization. The idea is to use both textbrewer and transformers and the project lies in directory /src/Experiments/.

2. Benchmark TextBrewer, DynaBERT, TinyBERT in NER, classification tasks and QA task.