from deap import base, creator, tools, algorithms
import random, string
from src.fitness import gsm8k_accuracy
from tqdm import tqdm
import csv, datetime
from concurrent.futures import ThreadPoolExecutor

POP = 30
GENS = 10
ELITE = 1
PREFIX = (
    "You are a helpful math tutor. First think step-by-step. On the **last line** print\n"
    "ANSWER: <the integer answer>\n\n"
    "Be concise. Explain your reasoning.\n\n"
)  # v1.0-prompted
TAIL_LEN = 40
ALPHABET  = list(string.ascii_lowercase + " ,.:;\n")
PHRASES = [
    "Let's think step by step.",
    "Explain your reasoning.",
    "Be concise.",
    "Show your work.",
    "Final answer:",
]
PHRASE_INSERT_P = 0.5   # 50% chance to try inserting a snippet
PHRASE_DELETE_P = 0.2   # 20% chance to try deleting a snippet

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def mutate_prompt(ind, indpb=0.1):
    # 1) character‐level noise (as before)
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] = random.choice(ALPHABET)

    # 2) phrase insertion
    if random.random() < PHRASE_INSERT_P:
        phrase = random.choice(PHRASES)
        # pick a random insert position
        pos = random.randrange(len(ind)+1)
        # insert the phrase’s characters
        for c in phrase:
            ind.insert(pos, c)

    # 3) phrase deletion (remove one known phrase if present)
    if random.random() < PHRASE_DELETE_P:
        tail_str = "".join(ind)
        for phrase in PHRASES:
            idx = tail_str.find(phrase)
            if idx != -1:
                # delete that slice
                for _ in phrase:
                    del ind[idx]
                break

    # 4) keep within bounds
    if len(ind) > TAIL_LEN:
        del ind[TAIL_LEN:]
    elif len(ind) < 1:
        # ensure at least one char
        ind.append(random.choice(ALPHABET))

    return ind,

def crossover(p1, p2):
    cut = random.randint(1, min(len(p1), len(p2)) - 1)

    # Use the parent’s __class__ to preserve type (and .fitness)
    child1 = p1.__class__(p1[:cut] + p2[cut:])
    child2 = p2.__class__(p1[cut:] + p2[:cut])

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

def run_ga(
        generations: int = GENS,
        pop_size: int   = POP,
        *,
        k: int = 10,          # ← how many dev‑items per prompt
        sc: int = 2,          # ← self‑consistency n_try
        penalty: float = 0.002,
        backend: str = "ollama",
        model:   str = "mistral:7b-instruct",
        sema: int = 6,
    ):

    bar = tqdm(total=generations+1, desc="GA", mininterval=0.5)
    # --- toolbox setup ---
    toolbox = base.Toolbox()
    import src.fitness as fitness
    fitness._SEM = fitness.Semaphore(sema)

    # attributes & individuals
    pool = ThreadPoolExecutor(max_workers=min(pop_size, 8))   # GA‑level parallelism
    toolbox.register("map", pool.map)
    toolbox.register("attr_char", random.choice, ALPHABET)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_char, TAIL_LEN)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation wraps gsm8k_accuracy
    cache: dict[str, float] = {}
    rnd = random.Random(42)
    BASE_PROMPT = PREFIX.rstrip()
    baseline_acc = gsm8k_accuracy(
        BASE_PROMPT,
        k=k,
        n_try=sc,
        rand=random.Random(0),
        backend=backend,
        model=model
    )
    bar.update(1)
    
    def evaluate(ind):
        prompt = BASE_PROMPT + "".join(ind).rstrip()
        # 1) get or compute raw accuracy
        if prompt not in cache:
            acc = gsm8k_accuracy(prompt, k=k, n_try=sc, rand=rnd,
                                   backend=backend, model=model)
            cache[prompt] = acc
        acc = cache[prompt]

        # 2) if we haven't beaten baseline, ignore length penalty
        if acc <= baseline_acc:
            fitness = acc
        else:
            fitness = acc - penalty * len(prompt)

        return (fitness,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    # light char‐swap, heavy phrase‐ops
    toolbox.register("mutate", mutate_prompt, indpb=0.02)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- GA main loop ---
    SEEDS = [
        "Give only the integer.",
        "Final answer:",
        ""                       # blank tail
    ]
    pop = [creator.Individual(list(s.strip().ljust(TAIL_LEN)[:TAIL_LEN])) for s in SEEDS]
    while len(pop) < pop_size:
        pop.append(toolbox.individual())
    for ind in pop:
        if all(c == " " for c in ind):
            ind[0] = random.choice(ALPHABET)

    log = open('ga_history.csv', 'w', newline='')
    writer = csv.writer(log)
    writer.writerow(['ts', 'gen', 'best', 'avg', 'prompt'])
    
    # Create a manual progress bar so we can call `update()` **after** all work in the
    # generation has finished.  Otherwise the bar appears to "freeze" during the long
    # evaluation step because the automatic update happens *before* the heavy work.
    best_hist, patience = [], 4
    

    for gen in range(generations):
        rnd = random.Random(42 + gen)
        cache.clear()
        # --- evaluate population (parallel) ---
        fits = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        # --- elitism / variation / replacement ---
        elite     = tools.selBest(pop, ELITE)
        offspring = toolbox.select(pop, len(pop) - ELITE)
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.3, mutpb=0.9)

        # re‑evaluate offspring (parallel!)
        fits = list(toolbox.map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop[:] = offspring + elite

        # --- stats & logging ---
        fits     = [ind.fitness.values[0] for ind in pop]
        avg_fit  = sum(fits) / len(fits)
        best     = tools.selBest(pop, 1)[0]

        writer.writerow([datetime.datetime.now().isoformat(),
                        gen, best.fitness.values[0], avg_fit, ''.join(best)])
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
                   help="length-penalty coefficient (score − penalty*len)")
    p.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                   help="LLM provider")
    p.add_argument("--model", default="mistral:7b-instruct", help="model ID/tag")
    p.add_argument("--sema", type=int, default=6, help="max concurrent LLM calls (fitness semaphore)")
    args = p.parse_args()

    best = run_ga(args.gens, args.pop,
              k=args.k, sc=args.sc, penalty=args.penalty,
              backend=args.backend, model=args.model,
              sema=args.sema)
    print("\n🏆 Best prompt:\n", "".join(best))
