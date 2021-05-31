from Distiller.glue_preprocess import load_and_cache_examples, Processor
from Distiller.glue_preprocess import MrpcProcessor
from Distiller.glue_preprocess import convert_examples_to_features, convert_features_to_dataset
import os
import torch
import argparse
import pandas as pd
from Distiller.transformers import AutoConfig, AutoTokenizer
from Distiller.transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from torch.utils.data import SequentialSampler, DataLoader


def main(args):
    config = AutoConfig.from_pretrained(args.model_path)
    args.model_type = config.model_type
    ## load pretrained models and tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config)

    model.to('cuda')
    processor = MrpcProcessor()
    examples = processor.get_test_examples(args.dataset_path)
    features = convert_examples_to_features(examples, tokenizer, task=args.task_name, max_length=args.max_seq_length,
                                            label_list=processor.get_labels(),
                                            output_mode=glue_output_modes[args.task_name])
    dataset = convert_features_to_dataset(features, is_training=False)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=32)
    # if args.task_name is not None:
    #     metric = load_metric("glue", args.task_name)
    preds = []
    label_list = []
    model.eval()
    for step, batch in enumerate(eval_dataloader):

        # labels = batch['labels']
        # batch = tuple(t.to(args.device) for t in batch)
        batch = {key: value.to('cuda') for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        # outputs = model(**batch)
        predictions = outputs.logits.detach().cpu()
        if args.task_name != "stsb":
            predictions = predictions.argmax(dim=-1)
        else:
            predictions = predictions[:, 0]
        label_list.extend(batch['labels'].cpu().tolist())
        preds.extend(predictions.tolist())
    pd.DataFrame({'prediction':preds}).to_csv(args.output_path)


glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "stsb": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--task_name", type=str, default="cola",
                        choices=["cola", "sst-2", "mrpc", "stsb", "qqp", "mnli", "mnli-mm", "qnli", "rte", "wnli"],
                        help="Only used when task type is glue")
    parser.add_argument("--max_seq_length", default=128)
    parser.add_argument("--tokenizer_path", default="huawei-noah/TinyBERT_General_4L_312D")
    parser.add_argument("--output_path", default="./predictions/")

    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    args.output_path = os.path.join(args.output_path, args.model_path.split('/')[-1]+"_"+args.task_name+".csv")
    main(args)
