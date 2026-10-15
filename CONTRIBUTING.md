# Contributing

Thank you for helping. Bug reports, new backends, new readouts and results on new tasks are all welcome.
For anything larger than a small fix, open an issue first so we can agree on the shape of the change. The open items
are in [ROADMAP.md](ROADMAP.md); the ones marked **help wanted** are good places to start.

## Development setup

```bash
git clone https://github.com/Ryenhails/GlanceRank.git      # or your fork
cd GlanceRank
pip install -e ".[dev]"
ruff check glancerank scripts tests && pytest -q      # what CI runs; CPU only, a few seconds
```

The unit tests need no model and no GPU: `tests/conftest.py` starts a small local HTTP server that answers like
vLLM's chat endpoint and JEV's `/v1/decide`, and `FakeBackend` scores options without a model.

## Adding a backend (`glancerank/backends/<engine>.py`)

A backend is one file with two methods, each one forward pass of the model:

```python
def choice(self, prompt: ChoicePrompt) -> tuple[list[float], int]:
    """One probability per option (summing to 1), and the number of prompt tokens."""

def yes_no(self, state: Sequence[Part], question: str) -> tuple[float, int]:
    """The probability of "yes", and the number of prompt tokens."""
```

`ChoicePrompt` holds the state with the labelled options already in it, the question and the labels
(`glancerank/prompts.py`). Read the label probabilities from the model's next-token distribution at the answer
position. Prompt building, image handling and ranking live above the backend and are tested once. Add a test with a
stubbed engine, in the style of `tests/test_backends.py`.

## Rules

- **No fabricated numbers.** A result in the README is a committed JSON under `bench/results/` written by
  `scripts/summarize.py`, with the command that produced it. `tests/test_readme_numbers.py` checks the tables. A run
  that has not finished shows "—".
- **No hidden decoding.** A ranking is read from next-token probabilities in one forward pass. Code that samples
  text and parses it is a different method; propose it in an issue first.
- **Keep the report's prompts fixed.** `Ranker` produces the exact prompts behind the technical report's numbers.
  A change to them needs new numbers, so put new prompt designs behind an option.
- **Small pull requests.** One change per pull request, with tests. Describe what you ran in the pull request.

## Reporting a bug

Use the bug report template. Include the backend, the model, the server and its version (for vLLM, the full
`vllm serve` command), and a minimal script that reproduces the problem. Please do not paste API keys or private
data into issues.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
