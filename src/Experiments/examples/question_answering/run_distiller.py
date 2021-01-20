import config
import glob
import logging
import os
import random
import numpy as np
import torch
from textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller
from matches import matches
from utils import write_evaluation, load_and_cache_examples
from evaluate import evaluate
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME

logger = logging.getLogger(__name__)

# ALL_MODELS = config.ALL_MODELS
MODEL_CLASSES = config.MODEL_CLASSES


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model_T, model, tokenizer, predict_callback):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler_class = get_linear_schedule_with_warmup
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler_args = {'num_warmup_steps':int(args.warmup_steps*t_total), 'num_training_steps':t_total}
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and args.local_rank == -1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if args.do_train and args.do_distill:
        intermediate_matches = None
        if isinstance(args.matches, (list, tuple)):
            intermediate_matches = []
            for match in args.matches:
                intermediate_matches += matches[match]
        logger.info(f"{intermediate_matches}")
        distill_config = DistillationConfig(
            temperature=args.temperature,
            intermediate_matches=intermediate_matches)
        train_config = TrainingConfig(device="cuda",log_dir=args.output_dir,output_dir=args.output_dir,local_rank=args.local_rank)
        adaptor_T = BertForQAAdaptor
        adaptor_S = BertForQAAdaptor
        distiller=GeneralDistiller(train_config,distill_config,model_T,model,adaptor_T,adaptor_S,)
        # distiller.train(optimizer,train_dataloader,args.num_train_epochs,
        #                 scheduler_class=scheduler_class, scheduler_args=scheduler_args,
        #                 max_grad_norm=1.0, callback=predict_callback)

        with distiller:
            distiller.train(optimizer, scheduler=None, dataloader=train_dataloader,
                              num_epochs=args.num_train_epochs, callback=predict_callback)
        return


def BertForQAAdaptor(batch, model_outputs, no_mask=False, no_logits=False):
    dict_obj = {'hidden':  model_outputs.hidden_states, 'attention':   model_outputs.attentions}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch['attention_mask']
    if no_logits is False:
        dict_obj['logits'] = (model_outputs.start_logits,model_outputs.end_logits)
    return dict_obj


def main(args):
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.bert_config_file_T if args.bert_config_file_T else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.output_hidden_states = True
    config.output_attentions = True
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,
                                                use_fast=False)
    model_T = model_class.from_pretrained(args.model_name_or_path,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

    model_name_or_path = args.model_name_or_path_student if args.model_name_or_path_student else args.model_name_or_path
    config = config_class.from_pretrained(args.bert_config_file_S if args.bert_config_file_S else model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.output_hidden_states = True
    config.output_attentions = True
    # config.num_hidden_layers=args.num_hidden_layers
    try:
        model = model_class.from_pretrained(model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    except:
        logger.info("Fail to load pre-trained parameters, the model will be randomly initialized")
        model = model_class(config)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    model_T.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    def predict_callback(model, step):
        if args.do_eval and args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                        do_lower_case=args.do_lower_case)
            examples, features, results = evaluate(args, model, tokenizer)
            write_evaluation(args, tokenizer, examples, features, results, prefix=str(step) + " step",write_prediction=False)
        model.train()
    # Training
    if args.do_train :
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
        # global_step, tr_loss = \
        train(args, train_dataset, model_T, model, tokenizer, predict_callback)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        model = model_class.from_pretrained(args.output_dir)  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out

        model.to(args.device)
        # Good practice: save your training arguments together with the trained model

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            examples, features, results = evaluate(args, model, tokenizer, prefix=global_step)
            write_evaluation(args, tokenizer, examples, features, results, prefix=str(global_step) + " step")
            # if global_step:
            #     result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            # results.update(result)
        # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     for key in sorted(results.keys()):
        #         writer.write("{} = {}\n".format(key, str(results[key])))

    return

if __name__ == '__main__':
    config.parse()
    args = config.args
    main(args)