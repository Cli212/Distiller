import importlib
import config

import textbrewer
from textbrewer import GeneralDistiller, BasicDistiller, TrainingConfig, DistillationConfig
PROCESSING_DICT = {'EQA': 'EQApreprocessing', 'NER': 'NERpreprocessing'}


def main(args):




if __name__ == '__main__':
    config.parse()
    args = config.args
    preprocessing = importlib.import_module(PROCESSING_DICT[args.task.upper()])
    main(args)
