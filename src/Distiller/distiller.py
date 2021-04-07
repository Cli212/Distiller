import os
import json
import glob
import torch
import logging
import random
import numpy as np
from tqdm import tqdm
from configs import parse
from autoaug import AutoAugmenter
from evaluate import evaluate_squad
from transformers import AutoConfig, AutoTokenizer
from squad_preprocess import convert_examples_to_features
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering
from textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller
from utils import squad_evaluate, load_and_cache_examples, CustomDataLoader, DataProvider, MyDataset, Logger
# from evaluate import evaluate
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME
from squad_preprocess import read_examples_from_file
from multiprocessing import Process, shared_memory
logger = Logger("all.log",level="debug").logger

task_dict = {'squad2': AutoModelForQuestionAnswering,
             'squad': AutoModelForQuestionAnswering,
             'token_classification': AutoModelForSequenceClassification,
             'sequence_classification': AutoModelForTokenClassification}


def BertForQAAdaptor(batch, model_outputs, no_mask=False, no_logits=False):
    dict_obj = {'hidden':  model_outputs.hidden_states, 'attention': model_outputs.attentions,"loss":model_outputs.loss}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch['attention_mask']
    if no_logits is False:
        dict_obj['logits'] = (model_outputs.start_logits,model_outputs.end_logits)
    return dict_obj

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def cal_layer_mapping(args, t_config, s_config):
    matches = []
    t_num_layers = t_config.num_hidden_layers
    s_num_layers = s_config.num_hidden_layers
    k = t_num_layers/s_num_layers
    for feature in args.intermediate_features:
        if args.intermediate_strategy == "skip":
            if feature == "hidden":
                for i in range(s_num_layers+1):
                    matches.append({'layer_T': int(i*k),'layer_S':i, 'feature':feature, 'loss':'hidden_mse', 'weight':1})
            elif feature == "attention":
                for i in range(s_num_layers):
                    matches.append({'layer_T': int((i+1)*k-1), 'layer_S': i, 'feature':feature, 'loss':'attention_ce', 'weight':1})
            else:
                continue
        elif args.intermediate_strategy == "last":
            if feature == "hidden":
                for i in range(s_num_layers+1):
                    matches.append({'layer_T': int(t_num_layers-s_num_layers+i), 'layer_S': i, 'feature':feature, 'loss':'hidden_mse', 'weight':1})
            elif feature == "attention":
                for i in range(s_num_layers):
                    matches.append({"layer_T": int(t_num_layers-s_num_layers+i),"layer_S":i, 'feature':feature, 'loss':'attention_ce', 'weight':1})
            else:
                continue
        elif args.intermediate_strategy == "EMD":
            pass
            ## TO DO
        else:
            break
    return matches


# class CustomDataLoader(DataLoader):


def train(args, examples, train_dataset, t_model, s_model, tokenizer, augmenter=None, matches=None, predict_callback=None):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # mix_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    def collate_fn(batch):
        new_batch = batch.copy()
        input_ids = [i['input_ids'] for i in new_batch]
        text_list = augmenter.augment([tokenizer.decode(input_id) for input_id in input_ids])
        return batch
    # train_dataloader = CustomDataLoader(train_dataset, examples, args=args, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn, tokenizer=tokenizer, augmenter=augmenter)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
    train_dataloader = DataProvider(train_dataset, examples, args, tokenizer, augmenter)
    # mix_dataloader = DataLoader(train_dataset, sampler=mix_sampler,
    #                             batch_size=args.train_batch_size) if args.mixup else None
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(examples) // args.gradient_accumulation_steps * args.num_train_epochs
        # t_total =
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in s_model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in s_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler_class = get_linear_schedule_with_warmup
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler_args = {'num_warmup_steps': int(args.warmup_steps * t_total), 'num_training_steps': t_total}
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     s_model, optimizer = amp.initialize(s_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and args.local_rank == -1:
        t_model = torch.nn.DataParallel(t_model)
        s_model = torch.nn.DataParallel(s_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        s_model = torch.nn.parallel.DistributedDataParallel(s_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        t_model = torch.nn.parallel.DistributedDataParallel(t_model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
    actual_batch_size = args.per_gpu_train_batch_size
    if augmenter:
        actual_batch_size *= 2
    if args.mixup:
        actual_batch_size *=2
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", actual_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                actual_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if args.train:
        intermediate_matches = matches
        # if args.intermediate_strategy == "skip":
        #     intermediate_matches = []
        #     for match in matches:
        #         intermediate_matches += matches[match]
        logger.info(f"{intermediate_matches}")
        distill_config = DistillationConfig(
            temperature=args.temperature,
            intermediate_matches=intermediate_matches,
            kd_loss_weight=args.kd_loss_weight,
            kd_loss_type=args.kd_loss_type)
        train_config = TrainingConfig(gradient_accumulation_steps=args.gradient_accumulation_steps, device="cuda",
                                      log_dir=os.path.join(args.output_dir, "log"), output_dir=args.output_dir,
                                      fp16=args.fp16, mixup=args.mixup)
        adaptor_T = BertForQAAdaptor
        adaptor_S = BertForQAAdaptor
        distiller = GeneralDistiller(train_config, distill_config, t_model, s_model, adaptor_T, adaptor_S, )

        with distiller:
            distiller.train(optimizer, scheduler=None, dataloader=train_dataloader,
                            num_epochs=args.num_train_epochs, callback=predict_callback)
            # distiller.train(optimizer,train_dataloader,args.num_train_epochs,
            #                 scheduler_class=scheduler_class, scheduler_args=scheduler_args,
            #                 max_grad_norm=1.0, callback=predict_callback, mixup_value=args.mixup_value,
            #                 mix_dataloader=mix_dataloader, local_rank=args.local_rank)
        return


def data_aug_process(augmenter, examples, tokenizer, args):

    while True:
        def example_iter():
            i = 0
            while i < len(examples):
                if (i + 32) >= len(examples):
                    yield [j.context_text for j in examples[i:]], i
                else:
                    yield [j.context_text for j in examples[i:i + 32]], i
                i += 32

        new_examples = examples.copy()
        pbar = tqdm(total=int(len(examples) / 32) + 1, desc="Data augmentation")
        for iter_sample in example_iter():
            text, i = iter_sample
            for ii, dd in enumerate(augmenter.augment(text)):
                new_examples[i + ii].context_text = dd
            pbar.update()
        features, dataset = convert_examples_to_features(new_examples, tokenizer, args.max_seq_length,
                                                         args.doc_stride,
                                                         args.max_query_length,
                                                         is_training=True,
                                                         threads=args.thread
                                                         )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = MyDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions,
                            all_end_positions)

def main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    ## load pretrained models and tokenizers
    if args.local_rank not in [-1,0]:
        torch.distributed.barrier()

    t_config = AutoConfig.from_pretrained(args.T_config_file if args.T_config_file else args.T_model_name_or_path)
    s_config = AutoConfig.from_pretrained(args.S_config_file if args.S_config_file else args.S_model_name_or_path)
    t_config.output_hidden_states = True
    t_config.output_attentions = True
    s_config.output_hidden_states = True
    s_config.output_attentions = True
    tokenizer = AutoTokenizer.from_pretrained(args.T_model_name_or_path, do_lower_case=args.do_lower_case,
                                                use_fast=False,
                                                config=t_config)
    # s_tokenizer = AutoTokenizer.from_pretrained(args.S_model_name_or_path, do_lower_case=args.do_lower_case,
    #                                             use_fast=False,
    #                                             config=s_config)
    ## Initialize augmenter
    augmenter = None
    if args.augmenter_config_path:
        augmenter = AutoAugmenter.from_config(args.augmenter_config_path,"cpu" if args.n_gpu==0 else "gpu")
    model_class = task_dict.get(args.task_type)
    t_model = model_class.from_pretrained(args.T_model_name_or_path, config=t_config)
    ## If the student borrow layers from teachers, it must borrow complete layers. Their hidden size and attention size
    # must be the same
    if args.random_student:
        s_model = model_class.from_config(s_config)
    else:
        s_model = model_class.from_pretrained(args.S_model_name_or_path, config=s_config)
    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)
    s_model.to(args.device)
    t_model.to(args.device)
    def predict_callback(model, step):
        if args.do_eval and args.local_rank in [-1, 0]:
            examples, features, results = evaluate_squad(args, model, tokenizer)
            evaluation = squad_evaluate(args, tokenizer, examples, features, results, prefix=f"{step}",
                                        write_prediction=False)
            logger.info("***** Eval results *****")
            logger.info(json.dumps(evaluation, indent=2) + '\n')

            output_eval_file = os.path.join(args.output_dir, f"{step}_eval_results.txt")
            logger.info(f"Write evaluation result to {output_eval_file}...")
            with open(output_eval_file, "a") as writer:
                writer.write(f"Output: {json.dumps(evaluation, indent=2)}\n")
            return evaluation['exact'], evaluation['f1']
        else:
            return None
    ## Training
    if args.train:
        # examples = read_examples_from_file(args.data_dir, mode="train", task_type=args.task_type)
        matches = cal_layer_mapping(args, t_config, s_config)
        train_dataset, features, examples = load_and_cache_examples(args, tokenizer, mode="train", return_examples=True)
        # p = Process(target=data_aug_process, args=(augmenter,examples,tokenizer,args))
        # p.start()
        # if args.S_model_name_or_path != args.T_model_name_or_path:
        #     s_train_dataset = load_and_cache_examples(args, s_tokenizer, mode="train", model_name_or_path=args.S_model_name_or_path, examples=examples)
        train(args, examples, train_dataset, t_model, s_model, tokenizer, augmenter, matches, predict_callback)

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = s_model.module if hasattr(s_model, "module") else s_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out

        model.to(args.device)
        # Good practice: save your training arguments together with the trained model

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
            model.to(args.device)
            examples, features, results = evaluate_squad(args, model, tokenizer, prefix=global_step)
            evaluation = squad_evaluate(args, tokenizer, examples, features, results)
            logger.info("***** Eval results *****")
            logger.info(json.dumps(evaluation, indent=2) + '\n')

            output_eval_file = os.path.join(args.output_dir, "final_eval_results.txt")
            logger.info(f"Write evaluation result to {output_eval_file}...")
            with open(output_eval_file, "a") as writer:
                writer.write(f"Output: {json.dumps(evaluation, indent=2)}\n")
    return

if __name__ == '__main__':
    args = parse()
    if args.S_model_name_or_path is None:
        args.S_model_name_or_path = args.T_model_name_or_path
    main(args)