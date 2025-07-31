# src/cli.py

import argparse
import json
import random
import asyncio
import os

from ga import run_ga, PREFIX, SNIPPETS
from fitness import gsm8k_accuracy

def main():
    p = argparse.ArgumentParser(prog="prompt-optimizer")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    run = sub.add_parser("tune", help="Run GA tuning")
    run.add_argument("--gens", type=int, default=10)
    run.add_argument("--pop", type=int, default=30)
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--sc", type=int, default=2)
    run.add_argument("--penalty", type=float, default=0.002)
    run.add_argument("--backend", default="ollama")
    run.add_argument("--model", default="mistral:7b-instruct")

    bench = sub.add_parser("bench", help="Benchmark best prompt")
    bench.add_argument("--k", type=int, default=50)
    bench.add_argument("--sc", type=int, default=5)

    args = p.parse_args()

    if args.cmd == "tune":
        best = asyncio.run(run_ga(
            generations=args.gens,
            pop_size=args.pop,
            k=args.k,
            sc=args.sc,
            penalty=args.penalty,
            backend=args.backend,
            model=args.model
        ))
        tail = " ".join(SNIPPETS[i] for i in best)
        print("🏆 Best snippet tail:")
        print(tail)
        # persist indices for benchmarking
        with open("best_tail.json", "w") as f:
            json.dump(best, f)
    else:
        # load persisted best-tail indices
        if not os.path.exists("best_tail.json"):
            print("❌ No best_tail.json found; run `prompt-opt tune` first.")
            return
        with open("best_tail.json") as f:
            best_tail = json.load(f)
        BASE_PROMPT = PREFIX.rstrip()
        full_prompt = BASE_PROMPT + ("\n" + " ".join(SNIPPETS[i] for i in best_tail) + "\n")
        acc = gsm8k_accuracy(
            full_prompt,
            k=args.k,
            n_try=args.sc,
            rand=random.Random(0),
            backend="ollama",
            model="mistral:7b-instruct"
        )
        print(f"Benchmark accuracy: {acc:.3%}")

if __name__ == "__main__":
    main()
