from typing import Iterable

import numpy as np


def label_agreement(doc: Iterable[set], label) -> float:
    """calculates the agreement ratio of a given label over all codes

    Args:
        doc (Sequence[set]): Sequence of labels. Each frozenset is one multilable code-set
        for a single coder.

        label (Hashable): label for which the ratio should be calculated

    Returns:
        float: ratio of agreement
    """

    i = 0
    length = 0
    for codes in doc:
        length += 1
        if label in codes:
            i += 1

    if i == 0:
        return np.nan

    ratio = i / length
    min_val = 1 / length
    max_val = 1 - min_val
    return (ratio - min_val) / max_val


def label_agreements(docs: Iterable[Iterable[set]], label) -> float:
    """calculates agreement on a certain label over multiple docs

    Args:
        data Iterable[Sequence[set]]: runs label_agreement over multiple documents.
    """

    n_docs = 0
    ratio = 0.0
    for doc in docs:
        if not any(label in coder for coder in doc):
            continue
        n_docs += 1
        ratio += label_agreement(doc, label)

    return ratio / n_docs


def label_confusion(doc: list[set], label1, label2) -> float:

    agreement1 = label_agreement(doc, label1)
    agreement2 = label_agreement(doc, label2)

    if any(np.isnan(val) for val in (agreement1, agreement2)):
        # should this be zero or nan?
        return 0.0

    if agreement1 >= agreement2:
        better, worse = label1, label2
        better_agreement = agreement1
    else:
        better, worse = label2, label1
        better_agreement = agreement2

    replaced_label = []
    for coder in doc:
        replaced = {label if label != worse else better for label in coder}
        replaced_label.append(replaced)

    replaced_agreement = label_agreement(replaced_label, better)

    return replaced_agreement - better_agreement
