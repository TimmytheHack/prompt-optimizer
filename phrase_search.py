# phrase_search.py

import itertools
import random

from src.fitness import gsm8k_accuracy  # :contentReference[oaicite:2]{index=2}

# ─── Configuration ────────────────────────────────────────────────────────────
PREFIX = (
    "You are a helpful math tutor. First think step-by-step. On the **last line** print\n"
    "ANSWER: <the integer answer>\n\n"
    "Be concise. Explain your reasoning.\n\n"
)

PHRASES = [
    "Let's think step by step.",
    "Explain your reasoning.",
    "Be concise.",
    "Show your work.",
    "Final answer:",
]

K = 10      # number of dev items
N_TRY = 2   # self-consistency samples
RAND = random.Random(0)

# ─── Brute-force over all subsets ─────────────────────────────────────────────
best_acc   = -1.0
best_combo = None

# include the empty set as well
for r in range(len(PHRASES) + 1):
    for combo in itertools.permutations(PHRASES, r):
        # build the prompt tail by joining snippets with spaces
        tail = " ".join(combo)
        prompt = PREFIX + tail + "\n"
        acc = gsm8k_accuracy(prompt, k=K, n_try=N_TRY, rand=RAND)
        
        print(f"Tested {combo or ['<no snippets>']}: accuracy = {acc:.3f}")
        if acc > best_acc:
            best_acc, best_combo = acc, combo

# ─── Report ─────────────────────────────────────────────────────────────────
print("\n🏆 Best raw accuracy:", best_acc)
print("Snippets:", best_combo or "<none>")
print("\nResulting prompt:\n")
print(PREFIX + " ".join(best_combo or []) + "\n")
