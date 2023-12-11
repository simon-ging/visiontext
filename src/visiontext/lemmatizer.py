import datetime
import os
from pathlib import Path
from typing import List, Optional

import h5py
from attr import define, field
from packg.strings import quote_with_urlparse
from loguru import logger
from spacy import Language
from packg.paths import get_data_dir
from visiontext.spacytools import load_spacy_model, SPACY_DEFAULT_EN


class LemmatizerInterface:
    def lemmatize(self, in_str: str) -> List[str]:
        raise NotImplementedError

    def batch_lemmatize(self, sentences: List[str]) -> List[List[str]]:
        raise NotImplementedError

    def get_unique_name(self) -> str:
        raise NotImplementedError


@define(slots=False)
class LemmatizerSpacy(LemmatizerInterface):
    name: str = SPACY_DEFAULT_EN
    _lemmatizer: Optional[Language] = field(init=False, repr=False, default=None)
    verbose: bool = False

    def get_unique_name(self) -> str:
        return f"spacy_{self.name}"

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            self._lemmatizer = load_spacy_model(self.name)
        return self._lemmatizer

    def lemmatize(self, in_str: str) -> List[str]:
        doc = self.lemmatizer(in_str)
        words_out = []
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"]:
                words_out.append(token.lemma_)
            else:
                words_out.append(token.text)
        return words_out

    def batch_lemmatize(self, sentences: List[str]) -> List[List[str]]:
        sentences_set = set(sentences)
        missing_sentences = sentences_set
        if self.verbose:
            logger.info(f"Lemmatizing {len(missing_sentences)} sentences")
        output_dict = {}
        for sentence in missing_sentences:
            words = self.lemmatize(sentence)
            output_dict[sentence] = words
        output_list = [output_dict[sentence] for sentence in sentences]
        return output_list


SEP_CHAR = "\x00"


@define(slots=False)
class LemmatizerDbWrapper(LemmatizerInterface):
    """
    Wrap a database (h5) to store the results for each sentence.
    """

    lemmatizer: LemmatizerInterface
    compute_missing: bool = True
    save_to_db: bool = True
    h5_file: Path = None

    def __attrs_post_init__(self):
        self.h5_file = (
            get_data_dir() / f"text_embeddings/lemmas/{self.lemmatizer.get_unique_name()}"
            if self.h5_file is None
            else self.h5_file
        )

    def lemmatize(self, in_str: str) -> List[str]:
        return self.batch_lemmatize([in_str])[0]

    def batch_lemmatize(self, sentences: List[str]) -> List[List[str]]:
        missing_sentences_set = set(sentences)

        output_dict = {}
        if self.h5_file.is_file():
            # read existing embeddings
            with h5py.File(self.h5_file, "r", libver="latest", swmr=True) as f:
                for sentence in list(missing_sentences_set):
                    quoted_sentence = quote_with_urlparse(sentence, prefix="q")
                    if quoted_sentence in f:
                        h5_strarr = f[quoted_sentence]
                        output_dict[sentence] = list(h5_strarr.asstr())
                        missing_sentences_set.remove(sentence)

        for text_input, words in output_dict.items():
            output_dict[text_input] = words

        if len(missing_sentences_set) > 0:
            new_output_dict = {}
            missing_sentences = sorted(list(missing_sentences_set))
            words_list = self.lemmatizer.batch_lemmatize(missing_sentences)

            for sentence, words in zip(missing_sentences, words_list):
                for word in words:
                    assert (
                        SEP_CHAR not in word
                    ), f"SEP_CHAR {SEP_CHAR} found in word {word} from sentence {sentence}"
                new_output_dict[sentence] = words

            if self.save_to_db:
                # save lemmas to db
                # single write multi read - use lockfile to make sure only one process writes
                os.makedirs(self.h5_file.parent, exist_ok=True)
                lockfile = self.h5_file.parent / f"{self.h5_file.name}.lock"
                assert (
                    not lockfile.is_file()
                ), f"lockfile {lockfile} exists with content {lockfile.read_text(encoding='utf-8')}"
                lockfile.write_text(f"locked at {datetime.datetime.now()}", encoding="utf-8")

                with h5py.File(self.h5_file, "a", libver="latest") as f:
                    f.swmr_mode = True
                    for i, (sentence, words) in enumerate(new_output_dict.items()):
                        quoted_sentence = quote_with_urlparse(sentence, prefix="q")
                        # words_raw = SEP_CHAR.join(words)
                        if quoted_sentence in f:
                            continue
                        f.create_dataset(quoted_sentence, data=words, dtype=h5py.string_dtype())
                    f.flush()
                lockfile.unlink()
            output_dict.update(new_output_dict)
        output_list = [output_dict[sentence] for sentence in sentences]
        return output_list


def get_lemmatizer(
    lemm_type: str = "spacy",
    lemm_name: str = SPACY_DEFAULT_EN,
    verbose: bool = False,
    use_db: bool = True,
    compute_missing: bool = True,
    save_to_db: bool = True,
    h5_file: Path = None,
) -> LemmatizerInterface:
    """

    Args:
        lemm_type: must be spacy
        lemm_name: name of the lemmatizer
        verbose:
        use_db: cache to h5 file
        compute_missing: compute if missing in cache
        save_to_db: save to db after computing
        h5_file: path to h5 file

    Returns:

    """
    assert lemm_type == "spacy", f"lemm_type {lemm_type} not supported"
    lemmatizer = LemmatizerSpacy(name=lemm_name, verbose=verbose)
    if not use_db:
        return lemmatizer
    lemmatizer_db = LemmatizerDbWrapper(
        lemmatizer=lemmatizer,
        compute_missing=compute_missing,
        save_to_db=save_to_db,
        h5_file=h5_file,
    )
    return lemmatizer_db
