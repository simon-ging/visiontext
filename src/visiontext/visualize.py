import torch
from typing import Optional

from visiontext.htmltools import NotebookHTMLPrinter
from visiontext.mathutils import torch_stable_softmax


def show_classification_logits_as_html(
    logits,
    classnames,
    temp: float = 1.0,
    k: int = 5,
    true_classname: Optional[str] = None,
    true_classid: Optional[int] = None,
    font_size=1.0,
):
    pr = NotebookHTMLPrinter()
    pr.open_table(font_size=font_size)
    # pr.add_table_row(*["#", "P", f"")
    for rank, _class_idx, prob, name in build_classification_logits_table(
        logits, classnames, temp=temp, k=k, true_classname=true_classname, true_classid=true_classid
    ):
        is_header = False
        if true_classname is not None and name == true_classname:
            is_header = True
        if true_classid is not None and _class_idx == true_classid:
            is_header = True
        pr.add_table_row(*[rank + 1, f"{prob:5.1%}", name], is_header=is_header)
    pr.close_table(output=True)


def build_classification_logits_table(
    logits,
    classnames,
    temp: float = 1.0,
    k: int = 5,
    true_classname: Optional[str] = None,
    true_classid: Optional[int] = None,
):
    """

    Args:
        logits: logits over the classes, array shape (n_classes,)
        classnames: list of classnames
        temp: temperature for softmax
        k: top k to return
        true_classname: classname of true class to add to the visualization
        true_classid: classid of true class to add to the visualization

    Returns:
        list length K of tuple: [(class_index, prob, classname)]
    """
    assert not (
        true_classname is not None and true_classid is not None
    ), "Cannot specify both true_classname and true_classid"
    logits_sm = torch_stable_softmax(logits, temp=temp)
    # values, indices = torch.topk(logits_sm, k, dim=0)
    values, indices = torch.topk(logits_sm, len(classnames), dim=0)
    logits_sm_np = logits_sm.cpu().numpy()
    indices = indices.cpu().numpy()
    output = []
    added_true_class = False
    for i in range(k):
        class_index = indices[i]
        class_name = classnames[class_index]
        output.append((i, class_index, logits_sm_np[class_index], class_name))
        if true_classname is not None and class_name == true_classname:
            added_true_class = True
        if true_classid is not None and class_index == true_classid:
            added_true_class = True
    if (true_classname is not None or true_classid is not None) and not added_true_class:
        for i in range(len(classnames)):
            class_index = indices[i]
            class_name = classnames[class_index]
            if (true_classname is not None and class_name == true_classname) or (
                true_classid is not None and class_index == true_classid
            ):
                output.append((i, class_index, logits_sm_np[class_index], class_name))
                break

    return output
