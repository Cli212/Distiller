import ray
import torch
import random
import numpy as np
import os
import json
import logging
from ray import tune
from ray.tune.schedulers import ASHAScheduler
# from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune import CLIReporter
from ..src.Distiller.configs import parse
from ..src.Distiller.autoaug import AutoAugmenter
from ..src.Distiller.distiller import train
from ..src.Distiller.utils import cal_layer_mapping
from ..src.Distiller.transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoConfig, AutoTokenizer
from torch.multiprocessing import Queue, Process, set_start_method
from ..src.Distiller.mp_aug import aug_process
import boto3


task_dict = {'squad2': AutoModelForQuestionAnswering,
             'squad': AutoModelForQuestionAnswering,
             'glue': AutoModelForSequenceClassification,
             'superglue': AutoModelForSequenceClassification}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


@ray.remote()
def train_fn(config, args):
    # Set ray tune hyper parameters
    for k,v in config.items():
        args.k = v
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
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    t_config = AutoConfig.from_pretrained(args.T_config_file if args.T_config_file else args.T_model_name_or_path)
    s_config = AutoConfig.from_pretrained(args.S_config_file if args.S_config_file else args.S_model_name_or_path)
    args.model_type = s_config.model_type
    s_config.num_labels = t_config.num_labels
    t_config.output_hidden_states = True
    t_config.output_attentions = True
    s_config.output_hidden_states = True
    s_config.output_attentions = True
    t_tokenizer = AutoTokenizer.from_pretrained(args.T_model_name_or_path,
                                                use_fast=False,
                                                config=t_config)
    s_tokenizer = AutoTokenizer.from_pretrained(args.S_model_name_or_path,
                                                use_fast=False,
                                                config=s_config) if args.S_model_name_or_path != args.T_model_name_or_path else None
    ## Initialize augmenter

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
        if args.eval and args.local_rank in [-1, 0]:
            evaluation_result = evaluate_func(args, model, s_tokenizer if s_tokenizer else t_tokenizer, prefix=step)
            logger.info("***** Eval results *****")
            logger.info(json.dumps(evaluation_result, indent=2) + '\n')

            output_eval_file = os.path.join(args.output_dir, f"{step}_eval_results.txt")
            logger.info(f"Write evaluation result to {output_eval_file}...")
            with open(output_eval_file, "a") as writer:
                writer.write(f"Output: {json.dumps(evaluation_result, indent=2)}\n")
            model.train()
            tune.report(iterations=step, accuracy=evaluation_result['acc'])
            return list(evaluation_result.values())[0]
        else:
            return None

    ## Training
    if args.train:
        # examples = read_examples_from_file(args.data_dir, mode="train", task_type=args.task_type)
        matches = cal_layer_mapping(args, t_config, s_config)
        train_dataset, s_dataset, features, s_features, examples = load_and_cache_examples(args, t_tokenizer,
                                                                                           mode="train",
                                                                                           return_examples=True,
                                                                                           s_tokenizer=s_tokenizer)
        augmenter = None
        q = None
        if args.aug_type:
            augmenter = AutoAugmenter.from_config(args.aug_type)
            q = Queue()
            process = Process(target=aug_process,
                              args=(q, examples, train_dataset, augmenter, args, t_tokenizer, s_tokenizer))
            process.start()
            # process.join()
        train(args, examples, train_dataset, t_model, s_model, t_tokenizer, augmenter, matches, predict_callback, q=q)

def main(args, gpus_per_trial=4):
    search_space = {
        "intermediate_strategy": tune.grid_search(["skip", "last", "EMD"]),
        "kd_loss_type": tune.grid_search(["ce", "mse"]),
        "intermediate_loss_type": tune.grid_search(["ce", "mse", "cos", "pkd", "nce"]),
        "aug_pipeline": tune.grid_search([True, False]),
        "mixup": tune.grid_search([True, False])}
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=args.num_train_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["accuracy"])
    from functools import partial
    # distributed_train_cifar = DistributedTrainableCreator(
    #     partial(train_fn, args=args),
    #     num_gpus_per_worker=10,
    #     num_cpus_per_worker=8
    # )
    result = tune.run(
        partial(train_fn, args=args),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=search_space,
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(
    #     best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    ray.init(address='auto', _redis_password='5241590000000000')
    args = parse()
    set_start_method('spawn')
    if args.S_model_name_or_path is None:
        args.S_model_name_or_path = args.T_model_name_or_path
    if args.task_type in ["squad", "squad2"]:
        args.task_name = args.task_type
        from ..src.Distiller.evaluate import evaluate_squad as evaluate_func
        from ..src.Distiller.squad_preprocess import load_and_cache_examples
    elif args.task_type == "glue":
        from ..src.Distiller.evaluate import evaluate_glue as evaluate_func
        from ..src.Distiller.glue_preprocess import load_and_cache_examples
    logger = logging.getLogger(__name__)
    main(args)



