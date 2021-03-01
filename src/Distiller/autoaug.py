import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

class AutoAugmenter:
    def __init__(self, aug):
        self.aug = aug

    def from_config(self, config_path):
