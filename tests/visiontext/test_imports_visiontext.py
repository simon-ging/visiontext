from pprint import pprint

import pytest

from packg.testing import (
    apply_visitor,
    ImportFromSourceChecker,
    recurse_modules,
)

module_list = list(recurse_modules("visiontext", ignore_tests=True, packages_only=False))
pprint(module_list)


@pytest.mark.parametrize("module", module_list)
def test_imports_from_source(module: str) -> None:
    print(f"Importing: {module}")
    apply_visitor(module=module, visitor=ImportFromSourceChecker(module))
