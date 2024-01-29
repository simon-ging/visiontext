"""
original nltk tokenizer lose the space information.

here the space information is preserved and the functions outputs list of tuples like
[['skiing', ' '], ['on', ' '], ['a', ' '], ['mountain', ' '], ['slope', ''], ['.', '']]

which can be used to reconstruct the original text input.

"""
from __future__ import annotations

import itertools

import nltk
from nltk.data import load
from nltk.tokenize.destructive import NLTKWordTokenizer


def ensure_setup_nltk(language="english"):
    try:
        load(f"tokenizers/punkt/{language}.pickle")
    except LookupError:
        nltk.download("punkt")


def tokenize_sentences_and_connectors(text: str, language="english") -> list[tuple[str, str]]:
    tokenizer = load(f"tokenizers/punkt/{language}.pickle")  # nltk caches internally
    return apply_nltk_tokenizer(tokenizer, text)


_treebank_word_tokenizer = None


def get_treebank_word_tokenizer():
    global _treebank_word_tokenizer
    if _treebank_word_tokenizer is None:
        _treebank_word_tokenizer = NLTKWordTokenizer()
    return _treebank_word_tokenizer


def tokenize_words_and_connectors(
    text: str, language="english", split_eos=True
) -> list[tuple[str, str]]:
    """
    Split text into tokens and remember all spaces so the text can be reconstructed.

    Args:
        text: input text
        language: tokenizer language
        split_eos: if true (default) eos tokens like "." will be returned as a separate list item.
            if false they will be appended to the previous word.

    Returns:
        list of tuples (word, connector) that can reconstruct the original text:
            word1 + connector1 + word2 + connector2 + ... = text
    """

    sentences_and_connectors = (
        [(text, "")] if not split_eos else tokenize_sentences_and_connectors(text, language)
    )
    words_and_connectors = []
    for sentence, sentence_connector in sentences_and_connectors:
        words_conns_here = apply_nltk_tokenizer(get_treebank_word_tokenizer(), sentence)
        for iw, (word, connector) in enumerate(words_conns_here):
            if iw == len(words_conns_here) - 1:
                connector = f"{connector}{sentence_connector}"
            words_and_connectors.append([word, connector])
    return words_and_connectors


def rebuild_from_words_and_connectors(words_and_connectors: list[tuple[str, str]]) -> str:
    return "".join(itertools.chain(*words_and_connectors))


def apply_nltk_tokenizer(tokenizer, text: str) -> list[tuple[str, str]]:
    """

    Args:
        tokenizer: nltk tokenizer
        text: input text

    Returns: list of tuples (word, connector) that can reconstruct the original text:
        word1 + connector1 + word2 + connector2 + ... = text

    """
    tokenizer_output = list(tokenizer.span_tokenize(text))
    if len(tokenizer_output) == 0:
        return []
    starts, stops = (list(a) for a in zip(*tokenizer_output))
    starts.append(len(text))

    output = []
    for n in range(len(stops)):
        word = text[starts[n] : stops[n]]
        connector = text[stops[n] : starts[n + 1]]
        output.append((word, connector))
    return output


def main():
    inp = "The image features two people standing on a snow-covered slope, both wearing skis and holding ski poles. They are posing for a picture, likely enjoying their time skiing on the mountain. The individuals are standing next to each other, with one person on the left and the other on the right. Their skis are placed in various positions, with one set of skis extending horizontally and the other set angled upward. The scene captures the excitement and fun of skiing on a mountain slope."
    print(inp)
    words_and_connectors = tokenize_words_and_connectors(inp)
    print(words_and_connectors)
    words_and_connectors2 = tokenize_words_and_connectors(inp, split_eos=True)
    print(words_and_connectors2)
    for w1c1, w2c2 in zip(words_and_connectors, words_and_connectors2):
        if w1c1 != w2c2:
            print(w1c1, "========|========", w2c2)

    reconstruct = "".join(itertools.chain(*words_and_connectors))
    print(reconstruct == inp)
    print()

    sentences_and_connectors = tokenize_sentences_and_connectors(inp)
    print(sentences_and_connectors)
    reconstruct = "".join(itertools.chain(*sentences_and_connectors))
    print(reconstruct == inp)
    # todo add tests, also test things like "" and "\n" as they can break things


if __name__ == "__main__":
    main()
