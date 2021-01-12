import importlib
import config
import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer
from transformers import CamembertConfig, CamembertForTokenClassification, CamembertTokenizer

PROCESSING_DICT = {'EQA': 'EQApreprocessing', 'NER': 'NERpreprocessing'}

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, DistilBertConfig)),
    ())

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}".format(mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = preprocessing.read_examples_from_file(args.data_dir, mode, args.version)
        if args.task.upper() == "EQA":
            features = preprocessing.convert_examples_to_features(examples, tokenizer, args.max_seq_length,
                                                                  args.doc_stride,
                                                                  args.max_query_length,
                                                                  is_training = True if mode=='train' else False,
                                                                  threads = args.thread
                                                                  )
        elif args.task.upper() == "NER":
            features = preprocessing.convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                                  cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                                  # xlnet has a cls token at the end
                                                                  cls_token=tokenizer.cls_token,
                                                                  cls_token_segment_id=2 if args.model_type in [
                                                                      "xlnet"] else 0,
                                                                  sep_token=tokenizer.sep_token,
                                                                  sep_token_extra=bool(args.model_type in ["roberta"]),
                                                                  # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                                  pad_on_left=bool(args.model_type in ["xlnet"]),
                                                                  # pad on the left for xlnet
                                                                  pad_token=tokenizer.convert_tokens_to_ids(
                                                                      [tokenizer.pad_token])[0],
                                                                  pad_token_segment_id=4 if args.model_type in [
                                                                      "xlnet"] else 0,
                                                                  pad_token_label_id=pad_token_label_id
                                                                  )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def main(args):




if __name__ == '__main__':
    config.parse()
    args = config.args
    preprocessing = importlib.import_module(PROCESSING_DICT[args.task.upper()])
    main(args)
