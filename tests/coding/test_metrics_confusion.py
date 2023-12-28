import pytest

from src.coding.metrics import label_confusion


def test_full_agreement():
    doc = [frozenset("A"), frozenset("A")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == 0

    doc = [frozenset("A"), frozenset(["A", "B", "C"])]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == 0


def test_full_confusion():
    doc = [frozenset("A"), frozenset("B")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == 1

    doc = [frozenset("A"), frozenset(["B", "C"])]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == 1


def test_3_coder_agreement():
    doc = [frozenset("A"), frozenset("A"), frozenset("A")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == 0

    doc = [frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset("A")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == 0


def test_3_coder_partial_confusion1():
    doc = [frozenset("A"), frozenset("A"), frozenset("B")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == pytest.approx(0.5)

    doc = [frozenset(["A", "B"]), frozenset(["B", "C"]), frozenset("A")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == pytest.approx(0.5)


def test_3_coder_partial_confusion2():
    doc = [frozenset("A"), frozenset("A"), frozenset("B"), frozenset("B")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == pytest.approx(0.66667, abs=1e-4)

    confusion = label_confusion(doc, "B", "A")
    assert confusion == pytest.approx(0.66667, abs=1e-4)


def test_3_coder_partial_confusion3():
    doc = [frozenset("A"), frozenset("A"), frozenset("A"), frozenset("B")]
    confusion = label_confusion(doc, "A", "B")
    assert confusion == pytest.approx(0.3333, abs=1e-4)

    confusion = label_confusion(doc, "B", "A")
    assert confusion == pytest.approx(0.3333, abs=1e-4)
