import sys

from pprint import pprint

import pytest

from packg import format_exception
from packg.testing import apply_visitor, ImportFromSourceChecker, recurse_modules

module_list = list(recurse_modules("visiontext", ignore_tests=True, packages_only=False))
pprint(module_list)


@pytest.mark.parametrize("module", module_list)
def test_imports_from_source(module: str) -> None:
    print(f"Importing: {module}")
    try:
        apply_visitor(module=module, visitor=ImportFromSourceChecker(module))
    except ModuleNotFoundError as e:
        if str(e) == "No module named 'spacy'" and sys.version < "3.9":
            version_str = str(sys.version).replace('\n', ' ').strip()
            print(
                f"{format_exception(e)} but spacy is not supported for python {version_str} "
                f"so this exception is expected."
            )
            return
