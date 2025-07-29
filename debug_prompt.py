# debug_prompt.py
from src.fitness import gsm8k_accuracy
import random

PROMPT = (
    "You are a helpful math tutor.\n"
    "Think step-by-step.\n"
    "On the final line write:\n"
    "ANSWER: "
)

if __name__ == "__main__":
    print("Scoring… this takes a few minutes on CPU.")
    acc = gsm8k_accuracy(PROMPT, k=20, n_try=2,
                     backend="openai", model="gpt-4o-mini",
                     rand=random.Random(0))
    print(acc)
