from __future__ import annotations  # py 3.9 support

from omegaconf import OmegaConf

from typedparser.objects import modify_nested_object


def load_dotlist(merge_dotlist: list[str] | None) -> dict:
    """Converts a list of key=value strings into a nested dictionary with dot notation support.

    Type conversions: int, float, bool, comma-separated lists, empty values become None.

    Args:
        merge_dotlist: List of strings in the format "key=value" or "parent.child=value".

    Returns:
        A dictionary with the parsed key-value pairs, with appropriate type conversions.
    """
    if merge_dotlist is None:
        return {}
    conf_dotlist = OmegaConf.from_dotlist(merge_dotlist)
    dict_dotlist = OmegaConf.to_container(conf_dotlist, resolve=True)

    def _convert_leaf(val):
        # convert commas to lists to allow for list overrides
        if isinstance(val, str):
            if "," in val:
                val = val.split(",")
                out_val = []
                for i, v in enumerate(val):
                    v = v.strip()
                    if v == "" and i == len(val) - 1:
                        # skip last empty string, so "key=value," creates the list ["value"].
                        continue
                    out_val.append(int(v) if v.isdigit() else v)
                val = out_val
        return val

    modify_nested_object(dict_dotlist, _convert_leaf)

    return dict_dotlist
