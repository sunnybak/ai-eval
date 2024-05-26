"""
Please refer to this example notebook for details on NLP Aug library:
https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb
"""

import nlpaug.augmenter.char as char_augmenter

# Character level augmentation
def typo_error_generator(text, error_rate=None, lang="en"):
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
    return augmented_text

def ocr_error_generator(text, error_rate=None):
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
    return augmented_text

def random_char_error_generator(text, error_rate=None, mode="swap"):
    """
    mode: str
        Mode of random char augmenter. It should be one of ['insert', 'substitute', 'swap', 'delete']
    """
    aug = char_augmenter.RandomCharAug(
        aug_char_p=error_rate,
        action=mode,
    )
    augmented_text = aug.augment(text)
    return augmented_text

__all__ = [
    'typo_error_generator',
    'ocr_error_generator',
    'random_char_error_generator',
]
