import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from configs import parse
from src.Distiller.utils import Logger
from src.Distiller.distiller import main as train_fn


def main(args, config):
    for k,v in config.items():
        args.k = v
    train_fn(args)


if __name__ == "__main__":
    args = parse()
    ray.init(address=12131)
    if args.S_model_name_or_path is None:
        args.S_model_name_or_path = args.T_model_name_or_path
    if args.task_type in ["squad","squad2"]:
        args.task_name = args.task_type
        from src.Distiller.evaluate import evaluate_squad as evaluate_func
        from src.Distiller.squad_preprocess import convert_examples_to_features, load_and_cache_examples, DataProvider, MyDataset
        from src.Distiller.adapters import BertForQAAdaptor as adaptor_func
    elif args.task_type == "glue":
        from src.Distiller.evaluate import evaluate_glue as evaluate_func
        from src.Distiller.glue_preprocess import convert_examples_to_features, load_and_cache_examples, DataProvider
        from src.Distiller.adapters import BertForGLUEAdptor as adaptor_func
    logger = Logger(f"{args.output_dir}/all.log", level="debug").logger
    from functools import partial
    annotate_ = partial(main, args)
    search_space = {
              "intermediate_strategy": tune.grid_search(["skip", "last", "EMD"]),
              "kd_loss_type": tune.grid_search(["ce", "mse"]),
              "intermediate_loss_type": tune.grid_search(["ce", "mse", "cos", "pkd", "nce"]),
              "aug_type": tune.grid_search(["random", "contextual", "back_translation"]),
              "mixup": tune.grid_search([True, False])}
    analysis = tune.run(
        annotate_,
        scheduler=ASHAScheduler(metric="accuracy", mode="max"),
        config=search_space)