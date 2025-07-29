#fitness.py
import json, re, asyncio, collections, random
from typing import Optional
from src.llm_wrapper import call_llama_async

_dev = json.load(open("data/gsm8k_dev.json"))
_NUM = re.compile(r"-?\d+")

def _extract_int(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):             # 1) line with “answer”
        if "answer" in line.lower():
            ints = _NUM.findall(line)
            if ints:
                return ints[-1]
    for line in reversed(lines):             # 2) last line with any int
        ints = _NUM.findall(line)
        if ints:
            return ints[-1]
    return None

async def _one_accuracy(prompt, item, *, model, backend):
    full = f"{prompt}\nQ: {item['question']}\nA:"
    return await call_llama_async(full, model=model, backend=backend)

async def _accuracy(prompt: str, *, k: int, n_try: int,
                    rand: random.Random, model: str, backend: str) -> float:

    sample = rand.sample(_dev, k) if k < len(_dev) else _dev

    outs = []
    for item in sample:
        for _ in range(n_try):
            outs.append(
                await _one_accuracy(prompt, item, model=model, backend=backend)
            )

    # ② regroup outputs back into their questions
    votes_per_q = collections.defaultdict(list)
    idx = 0
    for item in sample:
        for _ in range(n_try):
            o = outs[idx]
            if not isinstance(o, Exception):
                votes_per_q[item["question"]].append(o)
            idx += 1

    # ③ majority vote & score  (unchanged)
    correct = 0
    for item in sample:
        answers = votes_per_q[item["question"]]
        preds = [_extract_int(o) for o in answers]
        if not preds:
            continue
        vote = collections.Counter(preds).most_common(1)[0][0]
        gt   = _extract_int(item["answer"] or "")
        if gt is not None and vote == gt:
            correct += 1

    return correct / k

    

def gsm8k_accuracy(
        prompt: str,
        *,
        k: Optional[int] = None,
        n_try: int = 1,
        rand: Optional[random.Random] = None,
        backend: str = "ollama",
        model:   str = "mistral:7b-instruct",
) -> float:
    """
    Accuracy on a random subset (size k) of GSM-8K dev.
    Pass a `random.Random` instance in `rand` to make the
    sampling deterministic across calls.
    """
    if k is None:
        k = len(_dev)
    if rand is None:
        rand = random
    return asyncio.run(_accuracy(prompt, k=k, n_try=n_try, rand=rand, model=model, backend=backend))