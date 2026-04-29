import asyncio
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helper: DAG constraint enforcement
# ---------------------------------------------------------------------------

def make_dag(adj: np.ndarray) -> np.ndarray:
    """
    Remove back-edges from a binary adjacency matrix to guarantee a DAG.
    Uses a simple topological-sort-based approach: keep only edges (i -> j)
    where i < j in the sorted order (upper-triangular after reordering).
    This is a lightweight heuristic sufficient for small swarm graphs.
    """
    n = adj.shape[0]
    dag = np.zeros_like(adj)
    for i in range(n):
        for j in range(i + 1, n):   # only forward edges
            dag[i, j] = adj[i, j]
    return dag


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def init_population(pop_size: int, n_nodes: int, rng: np.random.Generator) -> list:
    """
    Create an initial population of random DAG adjacency matrices.
    Each individual is a binary upper-triangular matrix of shape (n_nodes, n_nodes).
    """
    population = []
    for _ in range(pop_size):
        # Random binary upper-triangular matrix (no self-loops)
        raw = rng.integers(0, 2, size=(n_nodes, n_nodes)).astype(float)
        individual = make_dag(raw)
        population.append(individual)
    return population


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def tournament_selection(population: list, fitnesses: list,
                         tournament_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Select one individual via tournament selection.
    Randomly sample `tournament_size` candidates and return the best.
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx].copy()


def crossover(parent_a: np.ndarray, parent_b: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    """
    Single-point crossover on the flattened edge vector.
    Returns one offspring, then enforces the DAG constraint.
    """
    flat_a = parent_a.flatten()
    flat_b = parent_b.flatten()
    point = rng.integers(1, len(flat_a))           # crossover point
    child_flat = np.concatenate([flat_a[:point], flat_b[point:]])
    child = child_flat.reshape(parent_a.shape)
    return make_dag(child)


def mutate(individual: np.ndarray, mutation_rate: float,
           rng: np.random.Generator) -> np.ndarray:
    """
    Bit-flip mutation: each edge is flipped independently with probability
    `mutation_rate`. DAG constraint is re-enforced afterwards.
    """
    mask = rng.random(individual.shape) < mutation_rate
    mutated = np.logical_xor(individual, mask).astype(float)
    return make_dag(mutated)


# ---------------------------------------------------------------------------
# Core EA loop
# ---------------------------------------------------------------------------

async def optimize_ea(
    swarm,
    evaluator,
    pop_size: int = 10,
    num_generations: int = 20,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    elitism: int = 1,
    seed: int = 42,
    display_freq: int = 5,
) -> np.ndarray:
    """
    Evolutionary Algorithm for GPTSwarm edge topology optimisation.

    Replaces the REINFORCE-based `optimize()` in optimization.py.

    Parameters
    ----------
    swarm        : GPTSwarm Swarm object (used to get n_nodes)
    evaluator    : GPTSwarm Evaluator object (async evaluate method)
    pop_size     : number of individuals in the population
    num_generations : number of EA generations
    mutation_rate   : per-edge bit-flip probability
    tournament_size : candidates sampled per tournament
    elitism      : number of best individuals carried over unchanged
    seed         : random seed for reproducibility
    display_freq : print progress every N generations

    Returns
    -------
    best_adj : np.ndarray
        Binary adjacency matrix of the best graph found.
    """
    rng = np.random.default_rng(seed)

    # Infer number of nodes from the swarm's composite graph
    n_nodes = len(swarm.composite_graph.nodes)

    # --- Initialise population ---
    population = init_population(pop_size, n_nodes, rng)
    fitnesses = [0.0] * pop_size

    best_adj = population[0].copy()
    best_fitness = -np.inf
    fitness_history = []   # track best fitness per generation

    pbar = tqdm(range(num_generations), desc="EA optimisation")

    for gen in pbar:

        # --- Evaluate all individuals ---
        tasks = [evaluator.evaluate_adj(ind) for ind in population]
        results = await asyncio.gather(*tasks)
        fitnesses = [float(r) for r in results]

        # Track global best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_adj = population[gen_best_idx].copy()

        fitness_history.append(best_fitness)

        if gen % display_freq == 0 or gen == num_generations - 1:
            print(
                f"Gen {gen:3d} | best={best_fitness:.3f} "
                f"| mean={np.mean(fitnesses):.3f} "
                f"| std={np.std(fitnesses):.3f}"
            )

        # --- Build next generation ---
        # Sort by fitness (descending) for elitism
        sorted_indices = np.argsort(fitnesses)[::-1]
        next_population = [population[i].copy() for i in sorted_indices[:elitism]]

        while len(next_population) < pop_size:
            parent_a = tournament_selection(population, fitnesses, tournament_size, rng)
            parent_b = tournament_selection(population, fitnesses, tournament_size, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, mutation_rate, rng)
            next_population.append(child)

        population = next_population
        pbar.set_postfix({"best": f"{best_fitness:.3f}"})

    print(f"\nEA complete. Best fitness: {best_fitness:.3f}")
    return best_adj, fitness_history