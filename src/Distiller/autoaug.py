import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util import Action
import json

class AutoAugmenter:
    def __init__(self, aug_args):
        augmenter_table = {"context": naw.ContextualWordEmbsAug,
                           "random": naw.RandomWordAug,
                           "back_translation": naw.BackTranslationAug}
        # self.augs = []
        # for i in aug_args:
        #     self.augs.append(augmenter_table.get(i.pop("augmenter"))(**i))
        self.aug = augmenter_table.get(aug_args.pop("augmenter"))(**aug_args)

    @classmethod
    def from_config(cls, augmenter_config_path, *configs):
        with open(augmenter_config_path) as f:
            aug_args = json.load(f)
        return cls(aug_args)

    def augment(self, data):
        # result = []
        # for aug in self.augs:
        #     result.extend(aug.augment(data))
        return self.aug.augment(data)

    def __len__(self):
        # return len(self.augs)
        return 1
from datasets import load_dataset