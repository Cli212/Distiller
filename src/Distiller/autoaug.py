from .nlpaug.augmenter import char as nac
from .nlpaug.augmenter import word as naw
from .nlpaug import flow as naf
from .nlpaug.util.audio.loader import AudioLoader
from .nlpaug.util import Action
import json

class AutoAugmenter:
    def __init__(self, aug_args, aug_type):
        augmenter_table = {"contextual": naw.ContextualWordEmbsAug,
                           "random": naw.RandomWordAug,
                           "back_translation": naw.BackTranslationAug}
        # self.augs = []
        # for i in aug_args:
        #     self.augs.append(augmenter_table.get(aug_type)(**i))
        self.aug = augmenter_table.get(aug_type)(**aug_args)

    @classmethod
    def from_config(cls, aug_type):
        augmenter_config_path = f"{aug_type}_augmenter_config.json"
        with open(augmenter_config_path) as f:
            aug_args = json.load(f)
        return cls(aug_args, aug_type)

    def augment(self, data):
        # result = []
        # for aug in self.augs:
        #     result.extend(aug.augment(data))
        return self.aug.augment(data)

    def __len__(self):
        # return len(self.augs)
        return 1