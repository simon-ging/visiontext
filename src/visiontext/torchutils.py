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
