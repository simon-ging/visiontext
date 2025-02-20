from __future__ import annotations

from collections import defaultdict

import re
import torch


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


def group_params_and_shapes_for_display(param_names, param_shapes):
    """
    Compress parameter names by grouping those that differ only in their first numeric part.

    Args:
        param_names: List of parameter names
        param_shapes: List of parameter shapes

    Returns:
        Tuple of lists (new_compressed_param_names, param_shapes)
    """
    groups = defaultdict(list)
    ungrouped = []

    # For each parameter, find the first number in its name
    for name, shape in zip(param_names, param_shapes):
        m = re.search(r"(\d+)", name)
        if m:
            # Split param into prefix, the found number, and suffix
            start, end = m.span(1)
            prefix = name[:start]
            number_str = m.group(1)
            suffix = name[end:]
            block_number = int(number_str)
            # Use the shape, prefix, and suffix as the grouping key
            key = (shape, prefix, suffix)
            groups[key].append(block_number)
        else:
            # If no number is found, leave this parameter ungrouped
            ungrouped.append((name, shape))

    output = []
    # Process groups that have a numeric part
    for (shape, prefix, suffix), nums in groups.items():
        nums.sort()
        if len(nums) > 1:
            # Compress multiple block numbers into a range (min-max)
            compressed_number = f"[{nums[0]}-{nums[-1]}]"
        else:
            compressed_number = str(nums[0])
        output.append((f"{prefix}{compressed_number}{suffix}", shape))

    # Append any entries that didn't contain a number
    output.extend(ungrouped)
    return (list(a) for a in zip(*output))
