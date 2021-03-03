import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

aug  = naw.ContextualWordEmbsAug(model_type='bart', model_path="../../models/bart-base")
text = "We try to build a meta-learning algorithm named Distiller"
augmented_data = aug.augment(text)
print(augmented_data)