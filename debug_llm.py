import random
from src.fitness import gsm8k_accuracy
from src.ga import PREFIX, SNIPPETS

BASE_PROMPT = PREFIX.rstrip()
# find the indices in SNIPPETS
best_tail_snippets = [
    SNIPPETS.index("Use bullet points for each step."),
    SNIPPETS.index("Final answer:"),
    SNIPPETS.index("Show your calculations.")
]

# rebuild full prompt
tail = " ".join(SNIPPETS[i] for i in best_tail_snippets)
full_prompt = BASE_PROMPT + (f"\n{tail}\n" if tail else "")

# evaluate
acc = gsm8k_accuracy(
    full_prompt,
    k=50,           # sample size
    n_try=5,        # self-consistency
    rand=random.Random(0),
    backend="ollama",
    model="mistral:7b-instruct"
)
print(f"Full Dev Accuracy: {acc:.3%}")
