import sys
import warnings

SPACY_DEFAULT_EN = "en_core_web_sm"
SPACY_DEFAULT_DE = "de_core_news_lg"
loaded_spacy_models = {}


# Check Python version
if sys.version_info < (3, 10):
    warnings.warn(
        f"spaCy does not support Python versions below 3.10. "
        f"Current Python version: {sys.version.split()[0]}. "
        f"Module '{__name__}' will not function.",
        RuntimeWarning,
    )

    def get_or_maybe_download_spacy_model(model=SPACY_DEFAULT_EN):
        raise RuntimeError(f"spacy does not support python<3.10. Currentversion: {sys.version}")

    def maybe_download_spacy_model(model=SPACY_DEFAULT_EN):
        raise RuntimeError(f"spacy does not support python<3.10. Currentversion: {sys.version}")

else:

    import spacy
    from spacy import Language
    from spacy.cli import download as spacy_download

    def get_or_maybe_download_spacy_model(model=SPACY_DEFAULT_EN):
        if model in loaded_spacy_models:
            return loaded_spacy_models[model]
        nlp = maybe_download_spacy_model(model)
        loaded_spacy_models[model] = nlp
        return nlp

    def maybe_download_spacy_model(model=SPACY_DEFAULT_EN):
        try:
            nlp = spacy.load(model)
        except OSError:
            spacy_download(model)
            nlp = spacy.load(model)
        return nlp
