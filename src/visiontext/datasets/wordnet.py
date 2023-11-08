"""
https://www.nltk.org/howto/wordnet.html

- Hypernyms = parents, hyponyms = children
- wnid has 8 numbers e.g. n00000001
- some of them do have more than 1 parent so its not a tree

(
    "n00001740",
    {
        "parent_wnids": [],
        "lemmas": ["entity"],
        "name": "entity.n.01",
        "definition": "that which is perceived or known or inferred to have its own distinct existence (living or nonliving)",
        "parent_names": [],
        "children_names": ["physical_entity.n.01", "abstraction.n.06", "thing.n.08"],
        "children_wnids": ["n00001930", "n00002137", "n04424418"],
        "min_depth": 0,
        "max_depth": 0,
    },
)

"""
from __future__ import annotations

from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset  # noqa
from typing import Optional

from packg.iotools import dump_json, load_json
from packg.paths import get_cache_dir
from packg.tqdmu import tqdm_max_ncols

_wordnet_nouns: Optional[dict] = None


def ensure_wordnet_is_downloaded():
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet")


def display_wnid(wnid: str, use_cache=True):
    """
    Display wordnet id

    >>> display_wnid("n01578575")
    n01578575 corvine_bird (birds of the crow family)

    Args:
        wnid: wordnet synset id
        use_cache: store results in cache for load_wordnet_nouns

    Returns:

    """
    wordnet_nouns = load_wordnet_nouns(use_cache=use_cache)
    wndata = wordnet_nouns[wnid]
    print(f"{wnid} {' | '.join(wndata['lemmas'])} ({wndata['definition']})")


def synset_to_wnid(synset):
    pos = synset.pos()
    offset = synset.offset()
    wnid = f"{pos}{offset:08d}"
    return wnid


def wnid_to_synset(wnid):
    pos = wnid[0]
    offset = int(wnid[1:])
    synset = wn.synset_from_pos_and_offset(pos, offset)
    return synset


def load_wordnet_nouns(use_cache=True, strip_underscore=True):
    """
    Load all noun synsets from wordnet.

    Args:
        use_cache: store results in cache for load_wordnet_nouns
        strip_underscore: remove underscore from lemmas

    Returns:
        dictionary of entries like
        {
            "n00001740": {
                "parent_wnids": [],
                "lemmas": ["entity"],
                "name": "entity.n.01",
                "definition": (
                    "that which is perceived or known or inferred to have its "
                    "own distinct existence (living or nonliving)"
                ),
                "parent_names": [],
                "children_names": ["physical_entity.n.01", "abstraction.n.06", "thing.n.08"],
                "children_wnids": ["n00001930", "n00002137", "n04424418"],
                "min_depth": 0,
                "max_depth": 0,
            },
            ...
        }

    """
    global _wordnet_nouns
    if _wordnet_nouns is not None:
        return _wordnet_nouns
    if use_cache:
        addstr = "_stripunderscore" if strip_underscore else ""
        cache_file = get_cache_dir() / f"wordnet_nouns{addstr}.json"
        if cache_file.is_file():
            _wordnet_nouns = load_json(cache_file)
            return _wordnet_nouns
    ensure_wordnet_is_downloaded()
    all_nouns = list(wn.all_synsets("n"))
    print(f"Total number of nouns: {len(all_nouns)}")
    root: Synset = wn.synset("entity.n.01")

    wordnet_data = {}
    wnid_to_children = defaultdict(list)
    for synset in tqdm_max_ncols(all_nouns, desc="Read WordNet, pass 1/2"):
        wnid = synset_to_wnid(synset)
        parents = synset.hypernyms() + synset.instance_hypernyms()
        parent_wnids = [synset_to_wnid(p) for p in parents]
        for p_wnid in parent_wnids:
            wnid_to_children[p_wnid].append(wnid)
        lemmas = synset.lemma_names()
        name = synset.name()
        definition = synset.definition()
        if strip_underscore:
            lemmas = [l.replace("_", " ") for l in lemmas]
        wordnet_data[wnid] = {
            "parent_wnids": parent_wnids,
            "lemmas": lemmas,
            "name": name,
            "definition": definition,
        }
    wnid_to_children = dict(wnid_to_children)
    for wnid, wndata in tqdm_max_ncols(
        wordnet_data.items(), total=len(wordnet_data), desc="Read WordNet, pass 2/2"
    ):
        synset = wnid_to_synset(wnid)
        parent_wnids = wndata["parent_wnids"]
        parent_names = [wordnet_data[p]["name"] for p in parent_wnids]
        children_wnids = wnid_to_children.get(wnid, [])
        children_names = [wordnet_data[c]["name"] for c in children_wnids]
        wndata.update(
            {
                "parent_names": parent_names,
                "children_names": children_names,
                "children_wnids": children_wnids,
                "min_depth": synset.min_depth(),
                "max_depth": synset.max_depth(),
            }
        )

        # make sure children returned by wordnet are the same as the ones found above
        hyponyms = synset.hyponyms() + synset.instance_hyponyms()
        hyponym_wnids = [synset_to_wnid(h) for h in hyponyms]
        assert set(hyponym_wnids) == set(children_wnids), f"{hyponym_wnids} != {children_wnids}"
        # holonyms = synset.member_holonyms()
    if use_cache:
        dump_json(wordnet_data, cache_file, create_parent=True)  # noqa

    _wordnet_nouns = wordnet_data
    return _wordnet_nouns


def main():
    wordnet_data = load_wordnet_nouns()
    # dump_json(wordnet_data, get_anno_dir() / "wordnet/nouns.json", create_parent=True)
    print(f"Done")


if __name__ == "__main__":
    main()
