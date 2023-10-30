from nltk.corpus.reader import Synset  # noqa

from visiontext.datasets.wordnet import load_wordnet_nouns, wnid_to_synset, synset_to_wnid, display_wnid


def test_wordnet():
    wordnet_data = load_wordnet_nouns()
    print(len(wordnet_data))

    entity_key = "n00001740"
    entity_dict = wordnet_data[entity_key]
    print(repr(entity_dict))
    assert entity_dict == {
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
    }

    e_synset = wnid_to_synset(entity_key)
    assert isinstance(e_synset, Synset) and e_synset.name() == "entity.n.01"

    e_wnid_re = synset_to_wnid(e_synset)
    assert e_wnid_re == entity_key
