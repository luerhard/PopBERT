import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar
from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc

from src.db.models.doccano import ExamplesExample  # type: ignore


@dataclass
class Sample:
    example: ExamplesExample
    nlp: Language
    label_dict: Optional[dict] = None

    NONE: ClassVar[str] = "none"

    def __post_init__(self) -> None:
        text = str(self.example.text)

        # replace SPACE + PUNKT with PUNKT + SPACE
        self.txt = re.sub(r"\s+\.", ". ", text).strip()

        # insert Space after PUNKT if following letter is uppercase
        # TODO: should this be done everytime?
        self.txt = re.sub(r"\.(?=[A-ZÄÜÖ])", ". ", self.txt)
        self.txt = re.sub(r"\?(?=[A-ZÄÜÖ])", "? ", self.txt)
        self.txt = re.sub(r"!(?=[A-ZÄÜÖ])", "! ", self.txt)

        self.txt = self.txt.strip()

        self.doc: Doc | None = None

    @cached_property
    def sents(self) -> list[str]:
        if self.doc is None:
            self.doc = self.nlp(self.txt)
        return [str(sent) for sent in self.doc.sents]

    @cached_property
    def text(self) -> str:

        sents = self.sents
        if len(sents) >= 3:
            return " ".join(sents[1:-1])
        elif 0 < len(sents) < 3:
            return " ".join(sents)

        raise Exception(f"No Sents found in: '{self.text}'!")

    @cached_property
    def labels(self):

        # Sample has not been coded yet
        if len(self.confirmed_by) == 0:
            return None

        # sample was coded by at least one user but no one has given any labels
        if len(self.label_counts.keys()) == 0:
            return [self.NONE]

        # if only one coder has labelled the sample, we need to take their answers as correct
        if len(self.confirmed_by) == 1:
            return [label for label in self.label_counts.keys()]

        # if more than one coder has labelled the sample, we only use labels
        # that have been assigned at least twice

        labels = list(self.label_counts.keys())
        if any(lab != self.NONE for lab in labels):
            labels = [
                label
                for label, count in self.label_counts.items()
                if count > 1 and label != self.NONE
            ]
        if len(labels) == 0:
            # TODO: What to do if the coders do not agree on any labels? is it None?
            # For now, I#ll set this to "no code was found"
            return [self.NONE]

        return labels

    @cached_property
    def confirmed_by(self):
        return [state.confirmed_by.username for state in self.example.state]

    @cached_property
    def user_labels(self) -> dict[str, set[str]]:
        # initialized with all confirmed users
        d: dict[str, set] = {user: set() for user in self.confirmed_by}

        # fill labels
        for label in self.example.labels:
            try:

                if self.label_dict:
                    label_text = self.label_dict[label.label.text]
                else:
                    label_text = label.label.text

                d[label.user.username].add(label_text)
            except KeyError:
                # happens when a user has labelled but not confirmed an example
                # we will ignore those cases for now and omit the labels
                pass

        # replace non-existing labels with "none"
        for user, labels in d.items():
            if len(labels) == 0:
                d[user] = {self.NONE}

        return d

    @cached_property
    def label_counts(self) -> Counter[str]:
        return Counter(label for user in self.user_labels.values() for label in user)
