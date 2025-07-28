# baseline.py

import random
import string
import argparse
from src.fitness import gsm8k_accuracy
from src.ga import PREFIX

# ─── Hyperparameters ─────────────────────────────────────────────────────────
K       = 10       # how many GSM8K dev items per evaluation
N_TRY   = 2        # self-consistency samples
PENALTY = 0.002    # length penalty coefficient

# ─── Evaluation function ──────────────────────────────────────────────────────
def evaluate(prompt: str) -> float:
    """
    Returns: accuracy(prompt) - PENALTY * len(prompt)
    """
    acc = gsm8k_accuracy(prompt, k=K, n_try=N_TRY)
    return acc - PENALTY * len(prompt)

# ─── Baseline: Random Search ──────────────────────────────────────────────────
def random_search(orig: str, n_iters: int = 50):
    best_p = orig
    best_s = evaluate(best_p)
    print(f"[Random] start score = {best_s:.4f}")

    # restrict to the same alphabet your GA uses
    alphabet = list(string.ascii_lowercase + " ,.:;\n")
    for i in range(n_iters):
        idx  = random.randrange(len(orig))
        cand = orig[:idx] + random.choice(alphabet) + orig[idx+1:]
        score = evaluate(cand)
        if score > best_s:
            best_p, best_s = cand, score
            print(f" iter {i:3d} → new best = {best_s:.4f}")
    return best_p, best_s

# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50,
                   help="number of random mutations to try")
    args = p.parse_args()

    orig = PREFIX.rstrip()
    best_prompt, best_score = random_search(orig, args.iters)

    print("\n🏆 Best random-search score:", f"{best_score:.4f}")
    print("Prompt:\n" + best_prompt)
