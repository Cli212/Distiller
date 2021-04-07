import argparse


def parse():
    parser = argparse.ArgumentParser()
    ## required arguments
    # parser.add_argument("--T_model_type", type=str, required=True, help="model type of teacher model")
    # parser.add_argument("--S_model_type", type=str, required=True, help="model type of student model")
    # parser.add_argument("--task_type", default="question_answering", choices=
    #                                 ['question_answering', 'token_classification', 'sequence_classification'])
    parser.add_argument("--T_model_name_or_path", type=str, required=True, help="teacher model name or path")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--task_type", default="squad2", choices=["squad", "squad2", "glue", "natural_questions", "multi_woz"])
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## optional arguments
    parser.add_argument("--S_model_name_or_path", type=str, default=None, help="student model name or path")
    parser.add_argument("--T_config_file", type=str, help="config file path of teacher model")
    parser.add_argument("--S_config_file", type=str, help="config file path of student model")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--temperature", default=1, type=float, required=False)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--augmenter_config_path", type=str, default=None)
    parser.add_argument("--layer_mapping_strategy", default='skip', choices=["skip", "first", "last"])
    parser.add_argument("--random_student", action="store_true", help="If true, the student model will initiate "
                                                                      "randomly")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--intermediate_strategy", default=None, choices=[None, "skip","last","EMD"])
    parser.add_argument("--intermediate_features", nargs="+", default=[], choices=["hidden","attention"])
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--kd_loss_weight", default=1.0, type=float, help="weight of kd loss")
    parser.add_argument("--kd_loss_type", default="ce", choices=["ce", "mse"])
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--repeated_aug", default=False, action="store_true")
    parser.add_argument("--do_lower_case", default=False, action="store_true")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--thread", default=8, type=int)
    parser.add_argument("--overwrite_cache", default=False, action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    print(args)
