import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
text = ["We propose to do a systematic study of knowledge distillation in natural language processing","What are you doing","how are you"]

model_dir = 'src/Distiller/models/'
aug = naw.WordEmbsAug(
    model_type='glove', model_path=model_dir+'glove.6B.200d.txt',
    action="substitute")
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)

aug = nas.ContextualWordEmbsForSentenceAug(
    model_path='gpt2')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)