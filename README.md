# Distillation_QA_benchmark



## Description 

We aim to benchmark various distillation algorithms for QA and NER tasks. 

Code for experiements can be put in directory /models.

## Dataset

For QA task, you can download simplified Natural Questions dataset by execute:

``` shell
python prepare_natural_questions.py
```

If you want to download all data(42G), you can execute

```shell
python prepare_natural_questions.py --all True
```

You can also find the official statement of NaturalQuestions here: https://ai.google.com/research/NaturalQuestions/download

For other datasets, see this: https://github.com/dmlc/gluon-nlp/tree/master/scripts/datasets/question_answering



## TO DOs

Benchmark TextBrewer, DynaBERT, TinyBERT in NER, classification tasks and QA task.