# tests/test_ga_smoke.py
import sys, os

# Ensure project root is on sys.path so we can `import src` when the test file
# is executed directly (python tests/test_ga_smoke.py).  When run via pytest
# this isn’t usually required, but it doesn’t hurt either.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import ga 

def dummy_accuracy(prompt: str) -> float:
    # Toy objective: maximise number of vowels minus length penalty
    return sum(prompt.count(v) for v in "aeiou") / (1 + len(prompt))

def test_ga_runs_quick(monkeypatch):
    # Monkey‑patch gsm8k_accuracy inside the module under test
    monkeypatch.setattr(ga, "gsm8k_accuracy", dummy_accuracy)

    best = ga.run_ga(generations=3, pop_size=6)   # tiny run → <2 s
    assert isinstance(best, ga.creator.Individual)
    assert len(best) > 0
