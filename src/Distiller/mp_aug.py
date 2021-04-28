import torch
from torch.multiprocessing import cpu_count, Pool, Queue
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import time

def example_iter(examples):
    i = 0
    while i < len(examples):
        if (i + 32) >= len(examples):
            # yield [j.context_text for j in examples[i:]],i
            yield examples[i:]
        else:
            # yield [j.context_text for j in examples[i:i+32]], i
            yield examples[i:i + 32]
        i += 32


def augment_data(iter_sample, augmenter, task_type):
    result = iter_sample.copy()
    if task_type in ['squad', 'squad2']:
        for ii, dd in enumerate(augmenter.augment([i.context_text for i in iter_sample])):
            result[ii].context_text = dd
    elif task_type == "glue":
        for ii, dd in enumerate(augmenter.augment([i.text_a for i in iter_sample])):
            result[ii].text_a = dd
        if hasattr(iter_sample[0],"text_b") and iter_sample[0].text_b:
            for ii, dd in enumerate(augmenter.augment([i.text_b for i in iter_sample])):
                result[ii].text_b = dd
    return result


def generate_aug_data(examples, original_dataset, augmenter, args, tokenizer, s_tokenizer=None):
    if args.task_type == "glue":
        from .glue_preprocess import convert_features_to_dataset, convert_examples_to_features
    elif args.task_type in ["squad", "squad2"]:
        from .squad_preprocess import convert_features_to_dataset, convert_examples_to_features
    else:
        raise NotImplementedError
    threads = min(args.thread, cpu_count())
    from functools import partial
    with Pool(threads) as p:
        # global examples
        # examples = self.examples
        annotate_ = partial(
            augment_data,
            augmenter=augmenter,
            task_type=args.task_type
        )
        aug_examples = list(
            tqdm(
                p.imap(annotate_, example_iter(examples), chunksize=256),
                total=int(len(examples) / 32) + 1,
                desc="Data augmentation",
                disable=False,
            )
        )
    new_examples = []
    for i in aug_examples:
        new_examples.extend(i)
    del aug_examples
    features = convert_examples_to_features(new_examples, tokenizer, args.max_seq_length,
                                            task=args.task_name
                                            )
    s_features = None
    if s_tokenizer:
        s_features = convert_examples_to_features(new_examples, s_tokenizer,
                                                  args.max_seq_length,
                                                  task=args.task_name
                                                  )

    dataset = convert_features_to_dataset(features, s_features)
    new_dataset = ConcatDataset([original_dataset, dataset])
    if args.local_rank == 0:
        torch.distributed.barrier()
    return new_dataset

def aug_process(queue:Queue, examples, original_dataset, augmenter, args, tokenizer, s_tokenizer=None):
    while True:
        if queue.empty():
            new_dataset = generate_aug_data(examples, original_dataset, augmenter, args, tokenizer, s_tokenizer)
            queue.put(new_dataset)
        else:
            time.sleep(300)
            continue

        # s_dataset = convert_features_to_dataset(s_features, is_training=True)
        # s_new_dataset = ConcatDataset([self.s_dataset, s_dataset])
        # self.s_dataset = s_new_dataset