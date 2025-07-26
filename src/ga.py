from deap import base, creator, tools, algorithms
import random, string
from src.fitness import gsm8k_accuracy
from tqdm import tqdm

POP = 30
GENS = 10
ELITE = 1
ALPHABET = list(string.ascii_lowercase + " ")

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def mutate_prompt(individual, indpb=0.1):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(ALPHABET)
    # insert or delete
    if random.random() < 0.1 and len(individual) < 300:
        individual.insert(random.randrange(len(individual)), random.choice(ALPHABET))
    if random.random() < 0.1 and len(individual) > 10:
        del individual[random.randrange(len(individual))]
    return individual,

def crossover(p1, p2):
    cut = random.randint(1, min(len(p1), len(p2)) - 1)

    # Use the parent’s __class__ to preserve type (and .fitness)
    child1 = p1.__class__(p1[:cut] + p2[cut:])
    child2 = p2.__class__(p1[cut:] + p2[:cut])

    return child1, child2

def run_ga(generations=GENS, pop_size=POP):
    toolbox = base.Toolbox()

    # attributes & individuals
    toolbox.register("attr_char", random.choice, ALPHABET)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_char, 40)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation wraps gsm8k_accuracy
    cache = {} 
    def evaluate(ind):
        prompt = "".join(ind).rstrip()
        if prompt not in cache:
            score = gsm8k_accuracy(prompt)
            penalty = 0.002 * len(prompt)      # 0.2% per character
            cache[prompt] = score - penalty
        return (cache[prompt],)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate_prompt, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- GA main loop ---
    SEEDS = [
        "Answer with one integer.",
        "Write just the number.",
        "Respond only with the final numeric result."
    ]
    pop = [creator.Individual(list(s.ljust(40)[:40])) for s in SEEDS]
    while len(pop) < pop_size:
        pop.append(toolbox.individual())
    
    bar = tqdm(range(generations), desc="GA")

    for gen in bar:
        # 1. evaluate population
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        #  Compute average fitness for display
        avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)

        # 2‑4. (elitism, variation, replacement)  ← keep your existing code here
        elite = tools.selBest(pop, ELITE)
        offspring = toolbox.select(pop, len(pop) - ELITE)
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.4)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring + elite

        # update bar footer
        best = tools.selBest(pop, 1)[0]
        bar.set_postfix(best=f"{best.fitness.values[0]:.3f}",
                        avg=f"{avg_fit:.3f}")
    return best

if __name__ == "__main__":
    run_ga()    