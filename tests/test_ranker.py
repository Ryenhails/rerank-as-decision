import pytest

from rerank_as_decision import FakeBackend, Ranker
from rerank_as_decision.ranker import CHOICE_QUESTION, YESNO_QUESTION


def score_by_digit(text):
    return float(text[-1]) if text and text[-1].isdigit() else 0.0


class Recorder(FakeBackend):
    def __init__(self, score):
        super().__init__(score)
        self.prompts = []

    def choice(self, prompt):
        self.prompts.append(prompt)
        return super().choice(prompt)

    def yes_no(self, state, question):
        self.prompts.append((list(state), question))
        return super().yes_no(state, question)


def test_listwise_prompt_and_order():
    backend = Recorder(score_by_digit)
    r = Ranker(backend).rank("the   query", ["c1", "c5", "c3"], instruction="Find it.")
    assert r.order == [1, 2, 0] and r.calls == 1 and sum(r.scores) == pytest.approx(1)
    (p,) = backend.prompts
    assert p.state == ["Instruction: Find it.\nQuery:", "the query", "\nCandidates:", "\nCandidate A:", "c1",
                       "\nCandidate B:", "c5", "\nCandidate C:", "c3"]
    assert p.question == CHOICE_QUESTION and p.option_names == ["Candidate A", "Candidate B", "Candidate C"]
    assert p.chat_instruction().endswith("Answer with the letter of the best candidate only (A-C).")


def test_pointwise_makes_one_call_per_candidate():
    backend = Recorder(score_by_digit)
    r = Ranker(backend, readout="pointwise").rank("q", ["c1", "c5", "c3", "c4"])
    assert r.order == [1, 3, 2, 0] and r.calls == 4 and backend.calls == 4
    state, question = next(p for p in backend.prompts if p[0][-1] == "c5")
    assert state[-2:] == ["\nCandidate:", "c5"] and question == YESNO_QUESTION


def test_ties_keep_the_input_order_and_limits():
    assert Ranker(FakeBackend(lambda text: 0.0)).rank("q", ["a", "b", "c"]).order == [0, 1, 2]
    assert Ranker(FakeBackend(score_by_digit)).rank("q", ["only"]).order == [0]
    with pytest.raises(ValueError):
        Ranker(FakeBackend(score_by_digit)).rank("q", ["c"] * 17)
    with pytest.raises(ValueError):
        Ranker(FakeBackend(score_by_digit), readout="setwise")
