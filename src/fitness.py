import json, re
from src.llm_wrapper import call_llama

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Think step-by-step internally, "
    "but output only the final numeric answer."
)

_dev = json.load(open("data/gsm8k_dev.json"))

def gsm8k_accuracy(prompt_suffix: str = "", k=len(_dev)):
    correct = 0
    for item in _dev[:k]:
        q, gold = item["question"], item["answer"]
        full_prompt = f"{SYSTEM_PROMPT} {prompt_suffix}\nQ: {q}\nA:"
        text = call_llama(full_prompt)

        ints = re.findall(r"-?\d+", text)

        if ints and ints[-1] == gold:
            correct += 1
    return correct / k
