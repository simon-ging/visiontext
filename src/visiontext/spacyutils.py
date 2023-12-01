import spacy
from spacy.cli import download as spacy_download

SPACY_DEFAULT_EN = "en_core_web_sm"
SPACY_DEFAULT_DE = "de_core_news_lg"


def load_spacy_model(model=SPACY_DEFAULT_EN):
    try:
        nlp = spacy.load(model)
    except OSError:
        spacy_download(model)
        nlp = spacy.load(model)
    return nlp


loaded_spacy_models = {}


def get_or_load_spacy_model(model=SPACY_DEFAULT_EN):
    if model in loaded_spacy_models:
        return loaded_spacy_models[model]
    nlp = load_spacy_model(model)
    loaded_spacy_models[model] = nlp
    return nlp
