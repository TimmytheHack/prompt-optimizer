from deap import base, creator, tools, algorithms
import random, string
from tqdm import tqdm
import csv, datetime
import asyncio
import fitness

POP = 30
GENS = 10
ELITE = 1
PREFIX = (
    # Introduction
    "You are a helpful math tutor. On the **last line** print\n"
    "ANSWER: <the integer answer>\n\n"

    # ─── Few-Shot Examples ───────────────────────────────────────────
    "Q: Sadie slept 8 hours on Monday. For the next two days, she slept 2 hours less, each, because she had to complete some assignments. If the rest of the week she slept 1 hour more than those two days, how many hours did she sleep in total throughout the week?\n"
    "A: Mon=8; next 2 days=8-2=6 each → 6*2=12; remaining 4 days=6+1=7 each → 7*4=28; total=8+12+28=48\n"
    "ANSWER: 48\n\n"

    "Q: Rosie can run 10 miles per hour for 3 hours. After that, she runs 5 miles per hour. How many miles can she run in 7 hours?\n"
    "A: First 3 h:10*3=30 mi; remaining 4 h:5*4=20 mi; total=30+20=50\n"
    "ANSWER: 50\n\n"

    "Q: Jennie is helping at her mom's office. She has a pile of 60 letters needing stamps, and a pile of letters already stamped. She puts stamps on one-third of the letters needing stamps. If there are now 30 letters in the pile of already-stamped letters, how many were in that pile when Jennie began?\n"
    "A: She stamped 60/3=20; so originally there were 30-20=10 already stamped\n"
    "ANSWER: 10\n\n"
    # ────────────────────────────────────────────────────────────────────

    # Instruction (the driver will add the actual Q/A)
    "Now solve the next problem:\n"
)

SNIPPETS = [
    "Let's think step by step.",
    "Explain your reasoning.",
    "Be concise.",
    "Show your work.",
    "Final answer:",
    "Answer only with the integer.",
    "Show your calculations.",
    "Break down each step.",
    "Answer succinctly.",
    "No explanation—just the numeric result.",
    "Include units if applicable.",
    "Outline each algebraic step.",
    "Answer succinctly in one sentence.",
    "Outline each algebraic step clearly.",
    "Use bullet points for each step."
]
MAX_SNIPPETS = 4 


if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def mutate_snippets(ind, indpb=0.2, addp=0.3, delp=0.3):
    # 1) change existing entries
    for i in range(len(ind)):
        if random.random() < indpb:
            new = random.randrange(len(SNIPPETS))
            ind[i] = new
    # 2) append a *new* snippet if not already present
    if random.random() < addp and len(ind) < MAX_SNIPPETS:
        choices = [i for i in range(len(SNIPPETS)) if i not in ind]
        if choices:
            ind.append(random.choice(choices))
    # 3) drop a snippet
    if random.random() < delp and len(ind) > 1:
        del ind[random.randrange(len(ind))]
    return ind,



def crossover(p1, p2):
    # if either parent is too short to split, skip crossover
    size = min(len(p1), len(p2))
    if size <= 1:
        return p1, p2

    # pick a cut point in [1, size-1]
    cut = random.randint(1, size - 1)

    # build children (preserving Individual class & fitness metadata)
    child1 = p1.__class__(p1[:cut] + p2[cut:])
    child2 = p2.__class__(p2[:cut] + p1[cut:])
    return child1, child2

# ---------------------------------------------------------------------------
# Fallback "hill climb" – placeholder so unit tests don’t crash.  It simply
# returns the individual unchanged.  Replace with a real local optimiser if
# desired.

def hill_climb(ind):
    """Trivial hill-climb stub: no change, just echo back the individual.

    Returns a tuple (best_individual, best_score) so that the caller keeps the
    expected interface without depending on a heavy implementation.
    """

    score = ind.fitness.values[0] if ind.fitness.valid else 0.0
    return ind, score

async def run_ga(
        generations: int = GENS,
        pop_size: int   = POP,
        *,
        k: int = 10,          # ← how many dev‑items per prompt
        sc: int = 2,          # ← self‑consistency n_try
        penalty: float = 0.002,
        backend: str = "ollama",
        model:   str = "mistral:7b-instruct",
    ):

    bar = tqdm(total=generations+1, desc="GA", mininterval=0.5)
    # --- toolbox setup ---
    toolbox = base.Toolbox()
    

    # attributes & individuals
    # snippet-index tails
    toolbox.register("attr_snip", random.randrange, len(SNIPPETS))
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_snip, MAX_SNIPPETS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation wraps gsm8k_accuracy
    cache: dict[str, float] = {}
    rnd = random.Random(42)
    BASE_PROMPT = PREFIX.rstrip()
    baseline_acc = await fitness._accuracy(
        BASE_PROMPT,
        k=k,
        n_try=sc,
        rand=rnd,
        backend=backend,
        model=model
    )
    bar.update(1)
    
    async def evaluate(ind):
        # join only non‐empty snippets, separated by spaces
        tail = " ".join(SNIPPETS[i] for i in ind)
        prompt = BASE_PROMPT + (f"\n{tail}\n" if tail else "")
        # 1) get or compute raw accuracy
        if prompt not in cache:
            acc = await fitness._accuracy(
                prompt, k=k, n_try=sc, rand=rnd, model=model, backend=backend
            )
            cache[prompt] = acc
        acc = cache[prompt]

        # 2) if we haven't beaten baseline, ignore length penalty
        if acc <= baseline_acc:
            fit_val = acc
        else:
            fit_val = acc - penalty * len(prompt)

        return (fit_val,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    # light char‐swap, heavy phrase‐ops
    toolbox.register("mutate", mutate_snippets)
    cxpb = 0.3; mutpb = 0.7
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- GA main loop ---
    # Seed with a few human templates (as indices)
    SEEDS = [
        [0],  # "Let's think step by step."
        [4],  # "Final answer:"
        []    # blank
    ]
    pop = [creator.Individual(seed) for seed in SEEDS]
    # fill out the rest randomly
    while len(pop) < pop_size:
        pop.append(toolbox.individual())

    log = open('ga_history.csv', 'w', newline='')
    writer = csv.writer(log)
    writer.writerow(['ts', 'gen', 'best', 'avg', 'prompt'])
    
    # Create a manual progress bar so we can call `update()` **after** all work in the
    # generation has finished.  Otherwise the bar appears to "freeze" during the long
    # evaluation step because the automatic update happens *before* the heavy work.
    best_hist, patience = [], 6
    

    for gen in range(generations):
        rnd = random.Random(42 + gen)
        cache.clear()
         # schedule all evaluations under fitness._SEM
        tasks = [evaluate(ind) for ind in pop]
        fits = []
        # ← create the “inner” progress bar for this generation
        with tqdm(total=len(pop), desc=f"G{gen}", leave=False) as inner:
            for coro in asyncio.as_completed(tasks):
                fit = await coro
                fits.append(fit)
                inner.update(1)

        # assign fitness to the evaluated population
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        # --- elitism / variation / replacement ---
        elite     = tools.selBest(pop, ELITE)
        offspring = toolbox.select(pop, len(pop) - ELITE)
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cxpb, mutpb=mutpb)

        # re-evaluate offspring (async parallel under our loop)
        fits_off = await asyncio.gather(*(evaluate(ind) for ind in offspring))
        for ind, fit in zip(offspring, fits_off):
            ind.fitness.values = fit

        pop[:] = offspring + elite

        # --- stats & logging ---
        fits     = [ind.fitness.values[0] for ind in pop]
        avg_fit  = sum(fits) / len(fits)
        best     = tools.selBest(pop, 1)[0]

      # turn the best Integer‐list into a human‐readable snippet tail
        best_tail = " ".join(SNIPPETS[i] for i in best)
   
        writer.writerow([
            datetime.datetime.now().isoformat(),
            gen,
            best.fitness.values[0],
            avg_fit,
            best_tail
        ])
        log.flush()
        bar.set_postfix(best=f"{best.fitness.values[0]:.3f}", avg=f"{avg_fit:.3f}")
        bar.update(1)

        # early stopping
        best_hist.append(best.fitness.values[0])
        if len(best_hist) > patience and max(best_hist[-patience-1:-1]) >= best_hist[-1]:
            print(f"No improvement in {patience} generations — early stop.")
            break

    bar.close()
    log.close()

    # optional local optimisation – currently a no-op stub
    best, _ = hill_climb(best)
    return best

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=GENS)
    p.add_argument("--pop",  type=int, default=POP)
    p.add_argument("--k",    type=int, default=10)
    p.add_argument("--sc",   type=int, default=2)
    p.add_argument("--penalty", type=float, default=0.002,
                   help="length-penalty coefficient (score - penalty*len)")
    p.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                   help="LLM provider")
    p.add_argument("--model", default="mistral:7b-instruct", help="model ID/tag")
    args = p.parse_args()

    best = asyncio.run(
        run_ga(args.gens, args.pop,
               k=args.k, sc=args.sc, penalty=args.penalty,
               backend=args.backend, model=args.model)
    )
    # turn the integer indices back into text snippets
    base = PREFIX.rstrip()
    best_tail = " ".join(SNIPPETS[i] for i in best)
    print("\n🏆 Best prompt:\n", base + (f"\n{best_tail}\n" if best_tail else ""))