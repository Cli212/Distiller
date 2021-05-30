import torch
from torch.multiprocessing import cpu_count, Pool, Queue, Manager
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import time
from .autoaug import AutoAugmenter

def example_iter(examples, batch_size):
    i = 0
    while i < len(examples):
        if (i + batch_size) >= len(examples):
            # yield [j.context_text for j in examples[i:]],i
            yield examples[i:]
        else:
            # yield [j.context_text for j in examples[i:i+32]], i
            yield examples[i:i + batch_size]
        i += batch_size


def augment_data(iter_sample, augmenter, task_type, max_length=512, tokenizer=None, model=None):
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
        if tokenizer and model:
            labels = torch.LongTensor([int(i.label) for i in result]).to(model.device)
            inputs = tokenizer(
                [(example.text_a, example.text_b) for example in result],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
                return_tensors="pt"
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model(**inputs, labels=labels)
            predictions = outputs.logits.detach().cpu()
            if model.config.finetuning_task != "stsb":
                predictions = predictions.argmax(dim=-1)
            else:
                predictions = predictions[:, 0]
            for i,d in enumerate(predictions.tolist()):
                result[i].label = str(d)
    return result

from functools import wraps

def generate_aug_data(examples, original_dataset, augmenter, args, tokenizer, s_tokenizer=None, model=None, batch_size=32):
    if args.task_type == "glue":
        from .glue_preprocess import convert_features_to_dataset, convert_examples_to_features
    elif args.task_type in ["squad", "squad2"]:
        from .squad_preprocess import convert_features_to_dataset, convert_examples_to_features
    else:
        raise NotImplementedError
    # m = Manager()
    # lock = m.Lock()
    threads = min(args.thread, cpu_count())
    from functools import partial
    # with Pool(threads) as p:
    #     # global examples
    #     # examples = self.examples
    #     # augmenter.augs[0].model.model = augmenter.augs[0].model.model.share_memory()
    #     annotate_ = partial(
    #         augment_data,
    #         augmenter=augmenter,
    #         task_type=args.task_type
    #     )
    #     aug_examples = list(
    #         tqdm(
    #             p.map(annotate_, example_iter(examples, batch_size)),
    #             total=int(len(examples) / batch_size) + 1,
    #             desc="Data augmentation",
    #             disable=False,
    #         )
    #     )
    if len(augmenter)>0:
        if model:
            annotate_ = partial(
                augment_data,
                augmenter=augmenter,
                task_type=args.task_type,
                max_length=args.max_seq_length,
                tokenizer=tokenizer,
                model=model
            )
        else:
            annotate_ = partial(
                augment_data,
                augmenter=augmenter,
                task_type=args.task_type,
                max_length=args.max_seq_length
            )
        new_examples = []
        for example in tqdm(example_iter(examples, batch_size), total=int(len(examples) / batch_size) + 1, desc="Data Augmentation"):
            new_examples.extend(annotate_(example))
        # new_examples = []
        # for i in aug_examples:
        #     new_examples.extend(i)
        # del aug_examples
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
        return new_dataset
    else:
        return original_dataset

def aug_process(rank, queue:Queue, examples, original_dataset, augmenter, args, tokenizer, s_tokenizer=None, model=None):
    while True:
        if queue.empty():
            new_dataset = generate_aug_data(examples, original_dataset, augmenter, args, tokenizer, s_tokenizer, model)
            queue.put(new_dataset)
        else:
            time.sleep(10)
            continue

        # s_dataset = convert_features_to_dataset(s_features, is_training=True)
        # s_new_dataset = ConcatDataset([self.s_dataset, s_dataset])
        # self.s_dataset = s_new_dataset