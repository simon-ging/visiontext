from __future__ import annotations

from collections.abc import Mapping
import re
from collections import defaultdict

import numpy as np
import torch
from natsort import natsorted

from packg.strings import dict_to_str_comma_equals
from typedparser.objects import compare_nested_objects


def get_statedict_mean_std(statedict):
    all_v = []
    for k, v in statedict.items():
        v_flat = v.flatten()
        all_v.append(v_flat)
    all_v = torch.cat(all_v)
    return all_v.mean().item(), all_v.std().item(), all_v


def compare_statedicts(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
    verbose: bool = False,
    atol=1e-8,
    rtol=1e-5,
) -> bool:
    """
    Compare two state dicts.

    Comparison math is similar to
    https://pytorch.org/docs/stable/generated/torch.allclose.html
    """
    errors = []
    set_sd1 = set(sd1.keys())
    set_sd2 = set(sd2.keys())
    # mismatches = set_sd1 ^ set_sd2  # symmetric difference operator
    keys_in_1_not_2 = set_sd1 - set_sd2
    if len(keys_in_1_not_2) > 0:
        errors.append(f"Keys in sd1 but not sd2: {sorted(keys_in_1_not_2)}")
    keys_in_2_not_1 = set_sd2 - set_sd1
    if len(keys_in_2_not_1) > 0:
        errors.append(f"Keys in sd2 but not sd1: {sorted(keys_in_2_not_1)}")
    matches = set_sd1 & set_sd2  # intersection operator
    all_diffs = []
    for key in matches:
        t1 = sd1[key]
        t2 = sd2[key]
        if t1.shape != t2.shape:
            errors.append(f"Shape mismatch for key {key}: {t1.shape} != {t2.shape}")
            continue
        abs_diff = torch.abs(t1 - t2)
        all_diffs.append(abs_diff.reshape(-1))
        if abs_diff.max() > atol:
            errors.append(f"Max abs diff for key {key}: {abs_diff.max()} > {atol}")
        if abs_diff.max() > rtol * torch.abs(t2).max():
            errors.append(
                f"Max rel diff for key {key}: {abs_diff.max()} > {rtol * torch.abs(t2).max()}"
            )
    all_diffs = torch.cat(all_diffs)
    if len(errors) > 0:
        errors.append(f"Max abs diff: {all_diffs.max()} mean abs diff: {all_diffs.mean()}")
        if verbose:
            print("\n".join(errors))
        return False
    return True


def _group_by_requires_grad(param):
    if param.requires_grad:
        return "grad"
    return "no_grad"


def _get_params(parameters):
    # support inputs: model, model.parameters(), model.named_parameters()
    if hasattr(parameters, "values"):
        parameters = parameters.values()
    if hasattr(parameters, "parameters"):
        parameters = parameters.parameters()
    return parameters


def count_params_by_requires_grad(parameters) -> dict[str, int]:
    parameters = _get_params(parameters)
    groups = {"grad": 0, "no_grad": 0}
    for v in parameters:
        if v.requires_grad:
            groups["grad"] += v.numel()
        else:
            groups["no_grad"] += v.numel()
    return groups


def count_params(parameters) -> int:
    parameters = _get_params(parameters)
    total = 0
    for v in parameters:
        total += v.numel()
    return total


def group_params_and_data_for_display(param_names: list[str], param_data: list[torch.Tensor | dict | tuple[int]]) -> tuple[list[str], list[torch.Tensor | dict] | tuple[int]]:
    """
    Compress parameter names by grouping those that differ only in their first numeric part.

    Args:
        param_names: List of parameter names
        param_data: List of parameter data. It can be:
            - The actual parameter tensors (torch.Tensor)
            - The shapes of the parameters (tuple[int])
            - Dicts containing parameter groups

    Returns:
        Tuple of lists (new_compressed_param_names, param_data)
    """
    groups = defaultdict(list)
    ungrouped = []

    # For each parameter, find the first number in its name
    for name, data in zip(param_names, param_data):
        m = re.search(r"(\d+)", name)
        if isinstance(data, Mapping):
            # strip the parameter itself (it cannot be grouped) and the name, since the name is
            # given in param_names
            data_clean = {k: v for k, v in data.items() if k not in {"param", "name", "param_name"}}
        elif hasattr(data, "shape"):
            # turn parameter into shape for display
            data_clean = tuple(getattr(data, "shape"))
        else:
            data_clean = tuple(data)
            for di, d in enumerate(data_clean):
                assert isinstance(d, int), f"Input is not tensor, param group, or shape: {data}"
        if m:
            # Split param into prefix, the found number, and suffix
            start, end = m.span(1)
            prefix = name[:start]
            number_str = m.group(1)
            suffix = name[end:]
            block_number = int(number_str)
            # Use the shape, prefix, and suffix as the grouping key
            if isinstance(data_clean, Mapping):
                data_key = dict_to_str_comma_equals(data_clean)
            else:
                data_key = ",".join(map(str, data_clean))
            key = (data_key, prefix, suffix)
            groups[key].append((block_number, data_clean))
        else:
            # If no number is found, leave this parameter ungrouped
            ungrouped.append((name, data_clean))

    output = []
    # Process groups that have a numeric part
    for (data_key, prefix, suffix), nums_and_data in groups.items():
        nums, data_list = [], []
        data = None
        for num, new_data in sorted(nums_and_data, key=lambda x: x[0]):
            nums.append(num)
            if data is None:
                data = new_data
            else:
                diffs = compare_nested_objects(data, new_data)
                if len(diffs) > 0:
                    raise ValueError(f"Data mismatch: {data} != {new_data}\nDiff: {diffs}")
        if len(nums) > 1:
            # Compress multiple block numbers into a range (min-max)
            compressed_number = f"[{nums[0]}-{nums[-1]}]"
        else:
            compressed_number = str(nums[0])
        output.append((f"{prefix}{compressed_number}{suffix}", data))

    # Append any entries that didn't contain a number
    output.extend(ungrouped)
    return (list(a) for a in zip(*output))

def group_params_and_shapes_for_display(param_names, param_shapes):
    """
    Deprecated version of the below function, but for now the data grouping function just groups
    the input, so it still works when being fed shapes instead of the actual parameters.
    So we keep this function for backwards compatibility for now.
    """
    return group_params_and_data_for_display(param_names, param_shapes)

def show_param_groups_dict(param_groups_dict, print_fn=print):
    """
    Print a table of parameter groups with their names, shapes, and number of parameters.

    Input format must be parameter groups, each group containing at least params and param_names:
    {
        "group_name": {
            "params": list[torch.Tensor],
            "param_names": list[str],
            "weight_decay": float,  # optional
            "lr": float,  # optional
        }
    }
    """
    n_params_total = 0
    sorted_group_keys = sorted(param_groups_dict.keys())
    for group_name in sorted_group_keys:
        group_content = param_groups_dict[group_name]
        params = group_content["params"]
        param_names = group_content["param_names"]
        wd = group_content.get("weight_decay", 0.0)
        lr = group_content.get("lr", 0.0)
        print_fn(f"{group_name:20s} {lr=:7.1e} {wd=:7.1e}")
        param_dict = {param_name: param for param_name, param in zip(param_names, params)}
        param_shapes = []
        group_n_params = 0
        for param_name, param in natsorted(param_dict.items(), key=lambda x: x[0]):
            param_shape = tuple(param.shape)
            param_shapes.append(param_shape)
            n_params = np.prod(param_shape)
            group_n_params += n_params
        new_names, new_shapes = group_params_and_data_for_display(param_names, param_shapes)
        for param_name, param_shape in zip(new_names, new_shapes):
            print_fn(f"    {str(param_shape):20s} {param_name}")
        print_fn(f"    Total parameters in group: {int(group_n_params):_d}")
        n_params_total += group_n_params
        print_fn("")
    print_fn(f"Total parameters across all groups: {int(n_params_total):_d}")
