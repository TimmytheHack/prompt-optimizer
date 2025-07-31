import time
import random
from statistics import mean
from src.fitness import gsm8k_accuracy
from src.ga import PREFIX, SNIPPETS

# Reconstruct your best prompt tail
BASE_PROMPT = PREFIX.rstrip()
best_tail = [
    "Use bullet points for each step.",
    "Final answer:",
    "Show your calculations."
]
tail_txt = " ".join(best_tail)
FULL_PROMPT = BASE_PROMPT + (f"\n{tail_txt}\n" if tail_txt else "")

# Settings
K = 50        # sample size per eval
N_TRY = 5     # self-consistency
SEED = 0

# 1) Measure average latency per question
start = time.time()
acc = gsm8k_accuracy(
    FULL_PROMPT,
    k=K,
    n_try=N_TRY,
    rand=random.Random(SEED),
    backend="ollama",
    model="mistral:7b-instruct"
)
duration = time.time() - start

print(f"Dev accuracy: {acc:.3%}")
print(f"Total eval time: {duration:.1f}s")
print(f"≈ {duration / (K * N_TRY):.2f}s per LLM call")

# 2) Rough token‐count estimate (prompt + one question + one answer)
#    💡 Ollama doesn’t expose token counts directly, so we approximate by word count
sample_q = "Q: 1+1?\nA:"
sample = FULL_PROMPT + "\n" + sample_q
words = len(sample.split())
print(f"Approx. {words} words/prompt+query (≈{words*1.3:.0f} tokens)")
