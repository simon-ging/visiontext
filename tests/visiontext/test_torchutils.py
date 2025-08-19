from pathlib import Path

import torch

import visiontext.testdata
from visiontext.torchutils import group_params_and_shapes_for_display, show_param_groups_dict

example_data = [
    ((2048,), "transformer.resblocks.10.mlp.c_fc.bias"),
    ((512,), "transformer.resblocks.10.mlp.c_proj.bias"),
    ((1536,), "transformer.resblocks.11.attn.in_proj_bias"),
    ((512,), "transformer.resblocks.11.attn.out_proj.bias"),
    ((512,), "transformer.resblocks.11.ln_1.bias"),
    ((512,), "transformer.resblocks.11.ln_1.weight"),
    ((512,), "transformer.resblocks.11.ln_2.bias"),
    ((512,), "transformer.resblocks.11.ln_2.weight"),
    ((2048,), "transformer.resblocks.11.mlp.c_fc.bias"),
    ((512,), "transformer.resblocks.11.mlp.c_proj.bias"),
    ((768,), "visual.ln_post.bias"),
    ((768,), "visual.ln_post.weight"),
    ((768,), "visual.ln_pre.bias"),
    ((768,), "visual.ln_pre.weight"),
    ((2304,), "visual.transformer.resblocks.0.attn.in_proj_bias"),
    ((768,), "visual.transformer.resblocks.0.attn.out_proj.bias"),
    ((768,), "visual.transformer.resblocks.0.ln_1.bias"),
    ((768,), "visual.transformer.resblocks.0.ln_1.weight"),
    ((768,), "visual.transformer.resblocks.0.ln_2.bias"),
    ((768,), "visual.transformer.resblocks.0.ln_2.weight"),
    ((3072,), "visual.transformer.resblocks.0.mlp.c_fc.bias"),
    ((768,), "visual.transformer.resblocks.0.mlp.c_proj.bias"),
    ((2304,), "visual.transformer.resblocks.1.attn.in_proj_bias"),
    ((768,), "visual.transformer.resblocks.1.attn.out_proj.bias"),
    ((768,), "visual.transformer.resblocks.1.ln_1.bias"),
    ((768,), "visual.transformer.resblocks.1.ln_1.weight"),
    ((768,), "visual.transformer.resblocks.1.ln_2.bias"),
    ((768,), "visual.transformer.resblocks.1.ln_2.weight"),
    ((3072,), "visual.transformer.resblocks.1.mlp.c_fc.bias"),
    ((768,), "visual.transformer.resblocks.1.mlp.c_proj.bias"),
    ((2304,), "visual.transformer.resblocks.2.attn.in_proj_bias"),
    ((768,), "visual.transformer.resblocks.2.attn.out_proj.bias"),
    ((768,), "visual.transformer.resblocks.2.ln_1.bias"),
    ((768,), "visual.transformer.resblocks.2.ln_1.weight"),
    ((768,), "visual.transformer.resblocks.2.ln_2.bias"),
    ((768,), "visual.transformer.resblocks.2.ln_2.weight"),
    ((3072,), "visual.transformer.resblocks.2.mlp.c_fc.bias"),
    ((768,), "visual.transformer.resblocks.2.mlp.c_proj.bias"),
    ((2304,), "visual.transformer.resblocks.3.attn.in_proj_bias"),
    ((768,), "visual.transformer.resblocks.3.attn.out_proj.bias"),
    ((768,), "visual.transformer.resblocks.3.ln_1.bias"),
    ((768,), "visual.transformer.resblocks.3.ln_1.weight"),
    ((768,), "visual.transformer.resblocks.3.ln_2.bias"),
    ((768,), "visual.transformer.resblocks.3.ln_2.weight"),
    ((3072,), "visual.transformer.resblocks.3.mlp.c_fc.bias"),
    ((768,), "visual.transformer.resblocks.3.mlp.c_proj.bias"),
    ((2304,), "visual.transformer.resblocks.4.attn.in_proj_bias"),
]
out_names = [
    "transformer.resblocks.[10-11].mlp.c_fc.bias",
    "transformer.resblocks.[10-11].mlp.c_proj.bias",
    "transformer.resblocks.11.attn.in_proj_bias",
    "transformer.resblocks.11.attn.out_proj.bias",
    "transformer.resblocks.11.ln_1.bias",
    "transformer.resblocks.11.ln_1.weight",
    "transformer.resblocks.11.ln_2.bias",
    "transformer.resblocks.11.ln_2.weight",
    "visual.transformer.resblocks.[0-4].attn.in_proj_bias",
    "visual.transformer.resblocks.[0-3].attn.out_proj.bias",
    "visual.transformer.resblocks.[0-3].ln_1.bias",
    "visual.transformer.resblocks.[0-3].ln_1.weight",
    "visual.transformer.resblocks.[0-3].ln_2.bias",
    "visual.transformer.resblocks.[0-3].ln_2.weight",
    "visual.transformer.resblocks.[0-3].mlp.c_fc.bias",
    "visual.transformer.resblocks.[0-3].mlp.c_proj.bias",
    "visual.ln_post.bias",
    "visual.ln_post.weight",
    "visual.ln_pre.bias",
    "visual.ln_pre.weight",
]
out_shapes = [
    (2048,),
    (512,),
    (1536,),
    (512,),
    (512,),
    (512,),
    (512,),
    (512,),
    (2304,),
    (768,),
    (768,),
    (768,),
    (768,),
    (768,),
    (3072,),
    (768,),
    (768,),
    (768,),
    (768,),
    (768,),
]


def test_format_named_params():
    shapes, names = zip(*example_data)
    ret_names, ret_shapes = group_params_and_shapes_for_display(names, shapes)
    assert ret_names == out_names
    assert ret_shapes == out_shapes


def test_show_param_groups_dict():
    named_params_dict = torch.load(
        Path(visiontext.testdata.__file__).parent / "example_weights.pth"
    )
    param_groups_dict = {
        "all": {
            "param_names": list(named_params_dict.keys()),
            "params": list(named_params_dict.values()),
            "weight_decay": 0.01,
            "lr": 0.001,
        }
    }
    show_param_groups_dict(param_groups_dict)
