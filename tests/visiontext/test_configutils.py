from __future__ import annotations  # py 3.9 support

import pytest

from visiontext.configutils import load_dotlist


def test_basic_key_value():
    dotlist = ["key=value"]
    result = load_dotlist(dotlist)
    assert result == {"key": "value"}


def test_multiple_key_value():
    dotlist = ["key1=value1", "key2=value2"]
    result = load_dotlist(dotlist)
    assert result == {"key1": "value1", "key2": "value2"}


def test_nested_key_value():
    dotlist = ["parent.child=value"]
    result = load_dotlist(dotlist)
    assert result == {"parent": {"child": "value"}}


def test_comma_separated_values():
    dotlist = ["key=value1,value2,value3"]
    result = load_dotlist(dotlist)
    assert result == {"key": ["value1", "value2", "value3"]}


def test_comma_separated_numbers():
    dotlist = ["key=1,2,3"]
    result = load_dotlist(dotlist)
    assert result == {"key": [1, 2, 3]}


def test_mixed_comma_separated_values():
    dotlist = ["key=1,value,3"]
    result = load_dotlist(dotlist)
    assert result == {"key": [1, "value", 3]}


def test_trailing_comma():
    dotlist = ["key=value1,value2,"]
    result = load_dotlist(dotlist)
    assert result == {"key": ["value1", "value2"]}


def test_empty_value():
    dotlist = ["key="]
    result = load_dotlist(dotlist)
    assert result == {"key": None}


def test_boolean_values():
    dotlist = ["key1=true", "key2=false", "key3=True", "key4=False"]
    result = load_dotlist(dotlist)
    assert result == {"key1": True, "key2": False, "key3": True, "key4": False}


def test_float_values():
    dotlist = ["key1=1.5", "key2=0.0", "key3=-3.14", "key4=1e-10"]
    result = load_dotlist(dotlist)
    assert result == {"key1": 1.5, "key2": 0.0, "key3": -3.14, "key4": 1e-10}


def test_none_input():
    result = load_dotlist(None)
    assert result == {}


def test_multiple_nested_with_lists():
    dotlist = ["parent.child1=value1,value2", "parent.child2=1,2,3", "parent.child3=value"]
    result = load_dotlist(dotlist)
    assert result == {
        "parent": {"child1": ["value1", "value2"], "child2": [1, 2, 3], "child3": "value"}
    }
