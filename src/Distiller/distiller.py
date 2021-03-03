import torch
import logging
import random
import numpy as np
from configs import parse
from autoaug import AutoAugmenter
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering



logger = logging.getLogger(__name__)

task_dict = {'question_answering': AutoModelForQuestionAnswering,
             'token_classification': AutoModelForSequenceClassification,
             'sequence_classification': AutoModelForTokenClassification}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args)


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
    t_tokenizer = AutoTokenizer.from_pretrained(args.T_model_name_or_path, do_lower_case=args.do_lower_case,
                                                use_fast=False,
                                                config=t_config)
    s_tokenizer = AutoTokenizer.from_pretrained(args.S_model_name_or_path, do_lower_case=args.do_lower_case,
                                                use_fast=False,
                                                config=s_config)
    ## Initialize augmenter
    if args.augmenter_config_path:
        augmenter = AutoAugmenter.from_config(args.augmenter_config_path)
    model_class = task_dict.get(args.task_type)
    t_model = model_class.from_pretrained('T_model_name_or_path', config=t_config)
    ## If the student borrow layers from teachers, it must borrow complete layers. Their hidden size and attention size
    # must be the same
    if args.random_student:
        s_model = model_class.from_config(s_config)
    else:
        s_model = model_class.from_pretrained('S_model_name_or_path', config=s_config)
    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    ## Training
    if args.train:



if __name__ == '__main__':
    args = parse()
    if args.S_model_name_or_path is None:
        args.S_model_name_or_path = args.T_model_name_or_path
    main(args)