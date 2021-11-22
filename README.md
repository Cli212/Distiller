# Distiller: A Systematic of Knowledge Distillation in NLP



## Description 

We aim to benchmark various distillation algorithms for QA and NER tasks. 

Code for experiements can be put in directory /src.

## Experiments

You can do KD experiments now using our code [here](experiments/). We only support experiments on QA tasks now but I am very confident that we will support NER tasks soon.

## Ray Tune

To use [ray](https://docs.ray.io/en/master/index.html) cluster to run the experiments, first

```shell
pip install -e .
```

then go to [ray_directory](ray_directory) 

## Dataset

For QA task, you can download SQuAD dataset by using gluon-nlp command:

``` shell
nlp_data prepare_squad --version 1.1
```

For downloading other datasets, follow this: https://github.com/dmlc/gluon-nlp/tree/master/scripts/datasets/question_answering

If you want to download all data of NaturalQuestions(42G), you can check out the official statement of NaturalQuestions here: https://ai.google.com/research/NaturalQuestions/download



## Reference
BibTeX entry of the EMNLP Workshop Version:
```@inproceedings{he2021distiller,
  title={Distiller: A Systematic Study of Model Distillation Methods in Natural Language Processing},
  author={He, Haoyu and Shi, Xingjian and Mueller, Jonas and Zha, Sheng and Li, Mu and Karypis, George},
  booktitle={Proceedings of the Second Workshop on Simple and Efficient Natural Language Processing},
  pages={119--133},
  year={2021}
}```
