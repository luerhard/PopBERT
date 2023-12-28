import pytest

from src.coding.metrics import label_agreement
from src.coding.metrics import label_agreements


def test_two_coder_item_full_agreement():
    doc = [frozenset("A"), frozenset("A")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 1

    doc = [frozenset("A"), frozenset(["A", "B", "C"])]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 1


def test_two_coder_item_full_disagreement():
    doc = [frozenset("A"), frozenset("B")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 0

    doc = [frozenset("A"), frozenset(["B", "C"])]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 0


def test_3_coder_label_agreement():
    doc = [frozenset("A"), frozenset("A"), frozenset("A")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 1

    doc = [frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset("A")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 1


def test_3_coder_item_disagreement():
    doc = [frozenset("A"), frozenset("B"), frozenset("B")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 0

    doc = [frozenset(["A", "B"]), frozenset(["B", "C"]), frozenset("B")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == 0


def test_3_coder_item_partial_agreement():
    doc = [frozenset("A"), frozenset("A"), frozenset("B")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == pytest.approx(0.5)

    doc = [frozenset(["A", "B"]), frozenset(["B", "C"]), frozenset("A")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == pytest.approx(0.5)


def test_4_coder_item_partial_agreement_1():
    doc = [frozenset("A"), frozenset("A"), frozenset("B"), frozenset("B")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == pytest.approx(0.3333, abs=1e-4)


def test_4_coder_item_partial_agreement_2():
    doc = [frozenset("A"), frozenset("A"), frozenset("A"), frozenset("B")]
    agreement_ratio = label_agreement(doc, "A")
    assert agreement_ratio == pytest.approx(0.6667, abs=1e-4)


def test_multidoc_full_agreement():
    doc1 = [frozenset("A"), frozenset("A")]
    agreement_ratio = label_agreements([doc1, doc1], "A")
    assert agreement_ratio == 1


def test_multidoc_half_agreement():
    doc1 = [frozenset("A"), frozenset("A")]
    doc2 = [frozenset("A"), frozenset("B")]
    agreement_ratio = label_agreements([doc1, doc2], "A")
    assert agreement_ratio == 0.5


def test_multidoc_and_coder_partial_agreement_1():
    doc1 = [frozenset("A"), frozenset("C"), frozenset("B")]
    doc2 = [frozenset("A"), frozenset("C"), frozenset("A")]
    agreement_ratio = label_agreements([doc1, doc2], "A")
    assert agreement_ratio == pytest.approx(0.25, abs=1e-4)


def test_multidoc_and_coder_partial_agreement_2():
    doc1 = [frozenset("A"), frozenset("A"), frozenset("A")]
    doc2 = [frozenset("A"), frozenset("A"), frozenset("B")]
    agreement_ratio = label_agreements([doc1, doc2], "A")
    assert agreement_ratio == pytest.approx(0.75, abs=1e-4)
