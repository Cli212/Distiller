import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util import Action

class AutoAugmenter:
    def __init__(self, aug):
        self.aug = aug

    def from_config(self, config_path):
