# fast_phrase_tune.py
import itertools, random
from src.fitness import gsm8k_accuracy

PREFIX = (
    "You are a helpful math tutor. First think step-by-step. On the **last line** print\n"
    "ANSWER: <the integer answer>\n\n"
)

# Your two slots…
SLOT1 = ["Be concise.", "Be succinct.", "Keep it brief."]
SLOT2 = ["Explain your reasoning.", "Show your work.", "Detail your reasoning."]

K, N_TRY = 10, 2
RAND = random.Random(0)

best_acc, best_combo = -1, None
for a, b in itertools.product(SLOT1, SLOT2):
    for order in [(a,b), (b,a)]:
        prompt = PREFIX + " ".join(order) + "\n"
        acc = gsm8k_accuracy(prompt, k=K, n_try=N_TRY, rand=RAND)
        print(f"Test {order!r} → acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_combo = acc, order

print("\n🏆 Best raw accuracy:", best_acc)
print("Winning snippets:", best_combo)
print("\nResulting prompt:\n", PREFIX + " ".join(best_combo) + "\n")
