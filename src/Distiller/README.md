## Description

`This is where you can do experiements for knowledge distillation.`

From my expectations, our code can be in the similar style. That is to say, if we want to add new task or new dataset into our code, we can write as few as possible new code  by copying existing code.



## How to experiment

Firstly, fork this project, and move to this directory.

Install dependencies by executing:

```shell
pip install -r requirements.txt
```

Download your preferred pretrained models and datasets(if the dataset is not squad, you should rewrite the function [read_examples_from_file](./examples/question_answering/preprocessing.py#L173))

After downloading all the data you need, update [finetune.sh](./finetune.sh) and [distillation.sh](distillation.sh) for setting hyperparameters.

Finally, you can executing finetune.sh and distillation.sh by order and get your distilled model.



