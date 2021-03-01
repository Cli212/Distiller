import argparse


def parse():
    parser = argparse.ArgumentParser()
    ## required arguments
    # parser.add_argument("--T_model_type", type=str, required=True, help="model type of teacher model")
    # parser.add_argument("--S_model_type", type=str, required=True, help="model type of student model")
    parser.add_argument("--task_type", default="question_answering", choices=
                                    ['question_answering', 'token_classification', 'sequence_classification'])
    parser.add_argument("--T_model_name_or_path", type=str, required=True, help="teacher model name or path")
    parser.add_argument("--S_model_name_or_path", type=str, required=True, help="student model name or path")

    ## optional arguments
    parser.add_argument("--S_model_name_or_path", type=str, default=None, help="student model name or path")
    parser.add_argument("--T_config_file", type=str, help="config file path of teacher model")
    parser.add_argument("--S_config_file", type=str, help="config file path of student model")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--augmenter_config_path", type=str, default=None)
    parser.add_argument("--layer_mapping_strategy", default='skip', choices=["skip", "first", "last"])
    parser.add_argument("--random_student", action="store_true", help="If true, the student model will initiate "
                                                                      "randomly")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    print(args)
