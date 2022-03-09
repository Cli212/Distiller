
import os
import json
import copy
import torch
import numpy as np
import argparse
from Distiller.glue_preprocess import load_and_cache_examples, glue_compute_metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

class STLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):
        self.max_mul = max_mul - 1
        self.turning_point = steps_per_cycle // (ratio + 1)
        self.steps_per_cycle = steps_per_cycle
        self.decay = decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        residual = self.last_epoch % self.steps_per_cycle
        multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
        if residual <= self.turning_point:
            multiplier *= self.max_mul * (residual / self.turning_point)
        else:
            multiplier *= self.max_mul * (
                (self.steps_per_cycle - residual) /
                (self.steps_per_cycle - self.turning_point))
        return [lr * (1 + multiplier) for lr in self.base_lrs]

def eval(args, model, tokenizer):
    dataset, s_dataset, features, s_features, examples = load_and_cache_examples(args, tokenizer, mode="dev",
                                                                                 return_examples=True)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    preds = []
    label_list = []
    model.eval()
    for batch in tqdm(eval_dataloader):

        # labels = batch['labels']
        # batch = tuple(t.to(args.device) for t in batch)
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            if model.config.model_type in ['distilbert']:
                batch.pop('token_type_ids')
            outputs = model(**batch)
        # outputs = model(**batch)
        predictions = outputs.logits.detach().cpu()
        if args.task_name not in ["stsb","cloth"]:
            predictions = predictions.argmax(dim=-1)
        else:
            predictions = predictions[:, 0]
        label_list.extend(batch['labels'].cpu().tolist())
        preds.extend(predictions.tolist())

    # eval_metric_compute = metric.compute()
    eval_metric = glue_compute_metrics(args.task_name, np.array(preds), np.array(label_list))
    if args.task_name == 'mnli':
        mnli_args = copy.deepcopy(args)
        mnli_args.task_name = 'mnli-mm'
        dataset, s_dataset, features, s_features, examples = load_and_cache_examples(mnli_args, tokenizer, mode="dev",
                                                                                     return_examples=True)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        preds = []
        label_list = []
        model.eval()
        for batch in tqdm(eval_dataloader):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with torch.no_grad():
                if model.config.model_type in ['distilbert']:
                    batch.pop('token_type_ids')
                outputs = model(**batch)
            # outputs = model(**batch)
            predictions = outputs.logits.detach().cpu()
            predictions = predictions.argmax(dim=-1)
            label_list.extend(batch['labels'].cpu().tolist())
            preds.extend(predictions.tolist())
        mm_eval_metric = glue_compute_metrics('mnli-mm', np.array(preds), np.array(label_list))
        eval_metric["mnli-mm/acc"] = mm_eval_metric["mnli-mm/acc"]
        eval_metric['acc'] = (eval_metric['acc']+eval_metric["mnli-mm/acc"]) / 2

    model.train()
    print(f"Eval result: {eval_metric}")
    return eval_metric


def main(args):
    best_result = 0.0
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    if not os.path.exists(args.data_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.data_dir)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    config = AutoConfig.from_pretrained(args.model_path)
    config.num_labels = args.num_labels
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,use_fast=False,
                                                config=config)
    dataset = load_and_cache_examples(args, tokenizer, "train", False)
    train_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*len(train_dataloader)*args.epoch, num_training_steps=len(train_dataloader)*args.epoch)
    model.to(args.device)
    model.train()
    if args.train:
        for i in range(args.epoch):
            print(f"Epoch {i+1}")
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch = {key: value.to(args.device) for key, value in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
            eval_result = eval(args, model, tokenizer)
            model.train()
            if eval_result['acc'] > best_result:
                best_result = eval_result['acc']
                model_to_save = model.module if hasattr(model,
                                                        "module") else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                if tokenizer:
                    tokenizer.save_pretrained(args.output_dir)
                torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                output_eval_file = os.path.join(args.output_dir, f"eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    writer.write(f"Output: {json.dumps(eval_result, indent=2)}\n")
                with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
                    arg_dict = vars(args)
                    arg_dict['device'] = str(arg_dict['device'])
                    json.dump(arg_dict, f)

    if args.eval:
        eval_result = eval(args, model, tokenizer)
        print(eval_result)



    # model_to_save = model.module if hasattr(model,
    #                                           "module") else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(args.output_dir)
    # if tokenizer:
    #     tokenizer.save_pretrained(args.output_dir)
    # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
    #     arg_dict = vars(args)
    #     arg_dict['device'] = str(arg_dict['device'])
    #     json.dump(arg_dict, f)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="finetuned/")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--local_rank", default=-1)
    parser.add_argument("--task_name", default="kaggle")
    parser.add_argument("--overwrite_cache", default=False)
    parser.add_argument("--learning_rate", default=5e-5,type=float)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_labels", default=2, type=int, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir + args.task_name + '/' + args.model_path
    main(args)

