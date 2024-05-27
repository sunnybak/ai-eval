"""
Please refer to this example notebook for details on NLP Aug library:
https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb
"""

import nlpaug.augmenter.char as char_augmenter
import nlpaug.augmenter.word as word_augmenter

# Character level augmentation
def typo_error_generator(text, error_rate=0.1, lang="en"):
    """
    Introduces typos based on the QWERTY keyboard layout.
    error_rate: float
        Rate of typos to introduce in the text.
    lang: str
        Language of the text. It should be one of ['en', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru']
    """
    aug = char_augmenter.KeyboardAug(
        aug_char_p=error_rate,
        lang=lang,
        include_special_char=False,
    )
    augmented_text = aug.augment(text)
    return augmented_text[0]

def ocr_error_generator(text, error_rate=0.1):
    """
    Introduces errors based on OCR (Optical Character Recognition) mistakes.
    For example, OCR may recognize I as 1 incorrectly.
    error_rate: float
        Rate of OCR errors to introduce in the text.
    """

    aug = char_augmenter.OcrAug(
        aug_char_p=error_rate,
    )
    augmented_text = aug.augment(text)
    return augmented_text[0]

def random_char_error_generator(text, error_rate=0.1, mode="swap"):
    """
    mode: str
        Mode of random char augmenter. It should be one of ['insert', 'substitute', 'swap', 'delete']
    """
    aug = char_augmenter.RandomCharAug(
        aug_char_p=error_rate,
        action=mode,
    )
    augmented_text = aug.augment(text)
    return augmented_text[0]

# Word level augmentation
def spelling_mistake_generator(text, error_rate=0.1):
    """
    Introduces spelling mistakes in the text based on common mistakes.
    error_rate: float
        Rate of spelling mistakes to introduce in the text.
    """
    aug = word_augmenter.SpellingAug(
        aug_p=error_rate,
    )
    augmented_text = aug.augment(text)
    return augmented_text[0]

def synonym_replacement_generator(text, replacement_rate=0.1):
    """
    Replaces some words in the text with their synonyms.
    error_rate: float
        Rate of words to replace with synonyms in the text.
    """
    aug = word_augmenter.SynonymAug(
        aug_p=replacement_rate,
    )
    augmented_text = aug.augment(text)
    return augmented_text[0]

def antonym_replacement_generator(text, replacement_rate=0.01):
    """
    Replaces some words in the text with their antonyms.
    error_rate: float
        Rate of words to replace with antonyms in the text.
    """
    aug = word_augmenter.AntonymAug(
        aug_p=replacement_rate,
    )
    augmented_text = aug.augment(text)
    return augmented_text[0]

def random_word_error_generator(text, error_rate=0.1, mode="delete"):
    """
    Introduces random word errors in the text.
	mode: str
		mode can be 'substitute', 'swap', 'delete', 'split' or 'crop'.
        If value is 'swap', adjacent words will be swapped randomly.
        If value is 'delete', word will be removed randomly.
        If value is 'crop', a set of contunous word will be removed randomly.
        If value is 'split', a word will be split into two words randomly.
    error_rate: float
        Rate of random word errors to introduce in the text.
    """
    aug = None
    if mode == "split":
        aug = word_augmenter.SplitAug(
            aug_p=error_rate,
        )
    else:
        aug = word_augmenter.RandomWordAug(
            action=mode,
            aug_p=error_rate,
        )
    augmented_text = aug.augment(text)
    return augmented_text[0]

__all__ = [
    'typo_error_generator',
    'ocr_error_generator',
    'random_char_error_generator',
    'spelling_mistake_generator',
    'synonym_replacement_generator',
    'antonym_replacement_generator',
    'random_word_error_generator',
]
