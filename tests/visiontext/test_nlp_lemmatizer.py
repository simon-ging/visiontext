"""
spacy does not support python 3.8 so we only run these tests when using the "full" requirements
"""

import sys

import pytest

from visiontext.nlp.lemmatizer import get_lemmatizer

input_sentence = "The dogs are running up and down the hills. ßßß"
output_words = ["The", "dog", "are", "run", "up", "and", "down", "the", "hill", ".", "ßßß"]


sv = sys.version_info


@pytest.mark.skipif(sv < (3, 10), reason=f"Spacy requires python >= 3.10, current version: {sv}")
@pytest.mark.full
def test_lemmatizer_simple():
    lemmatizer = get_lemmatizer(use_db=False)
    output = lemmatizer.lemmatize(input_sentence)
    assert output == output_words

    outputs = lemmatizer.batch_lemmatize([input_sentence])
    assert outputs == [output_words]


@pytest.mark.skipif(sv < (3, 10), reason=f"Spacy requires python >= 3.10, current version: {sv}")
@pytest.mark.full
def test_lemmatizer_db(tmp_path_factory: pytest.TempPathFactory):
    tmp_file = tmp_path_factory.mktemp("data").joinpath("TEMPtest.h5")
    lemmatizer = get_lemmatizer(h5_file=tmp_file, use_db=True)
    output = lemmatizer.lemmatize(input_sentence)
    assert output == output_words

    outputs = lemmatizer.batch_lemmatize([input_sentence])
    assert outputs == [output_words]

    outputs_again = lemmatizer.batch_lemmatize([input_sentence])
    assert outputs_again == outputs
