import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from configs import parse
# from ..src.Distiller.utils import Logger
from ..src.Distiller.distiller import main as train_fn


def main(args, config):
    for k,v in config.items():
        args.k = v
    train_fn(args)


if __name__ == "__main__":
    args = parse()
    ray.init(address=12131)
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