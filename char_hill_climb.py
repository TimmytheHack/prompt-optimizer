# char_hill_climb.py

import random
from src.fitness import gsm8k_accuracy

# 1) Configuration
from src.ga import PREFIX  # now your frozen, polished prompt
K, N_TRY = 10, 2
PENALTY = 0.0           # ignore length here; pure raw accuracy
RAND = random.Random(0)

# 2) Initialize
best = PREFIX.rstrip() + "\n"
best_acc = gsm8k_accuracy(best, k=K, n_try=N_TRY, rand=RAND)
alphabet = list("abcdefghijklmnopqrstuvwxyz ,.:;\n")

print(f"Starting raw accuracy = {best_acc:.3f}")

# 3) Greedy search
improved = True
while improved:
    improved = False
    for i in range(len(best)):
        for c in alphabet:
            candidate = best[:i] + c + best[i+1:]
            acc = gsm8k_accuracy(candidate, k=K, n_try=N_TRY, rand=RAND)
            if acc > best_acc:
                best_acc, best = acc, candidate
                print(f" New best acc={best_acc:.3f} (pos {i}, char '{c}')")
                improved = True
                break
        if improved:
            break

# 4) Report
print("\n🏆 Final raw accuracy:", best_acc)
print("Final prompt:\n")
print(best)
