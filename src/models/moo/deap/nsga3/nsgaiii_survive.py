from itertools import chain

import numpy as np
from deap.tools import sortLogNondominated


def selection_NSGA3(individuals, k, ref_points):
    pareto_fronts = sortLogNondominated(individuals, k)

    # Extract fitnesses as a np array in the nd-sort order
    # Use wvalues * -1 to tackle always as a minimization problem
    fitnesses = np.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitnesses *= -1

    # Get best and worst point of population, contrary to pymoo
    best_point = np.min(fitnesses, axis=0)
    worst_point = np.max(fitnesses, axis=0)

    extreme_points = find_extreme_points(fitnesses, best_point)
    front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    # Get counts per niche for individuals in all front but the last
    niche_counts = np.zeros(len(ref_points), dtype=np.int64)
    index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    # Choose individuals from all fronts but the last
    chosen = list(chain(*pareto_fronts[:-1]))

    # Use niching to select the remaining individuals
    sel_count = len(chosen)
    n = k - sel_count
    selected = niching(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
    chosen.extend(selected)

    return chosen

def find_extreme_points(fitnesses, best_point):
    'Finds the individuals with extreme values for each objective function.'

    # Translate objectives
    ft = fitnesses - best_point

    # Find achievement scalarizing function (asf)
    asf = np.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = np.max(ft * asf[:, np.newaxis, :], axis=2)

    # Extreme point are the fitnesses with minimal asf
    min_asf_idx = np.argmin(asf, axis=1)
    return fitnesses[min_asf_idx, :]


def find_intercepts(extreme_points, best_point, current_worst, front_worst):
    """Find intercepts between the hyperplane and each axis with
    the ideal point as origin."""
    # Construct hyperplane sum(f_i^n) = 1
    b = np.ones(extreme_points.shape[1])
    A = extreme_points - best_point
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        intercepts = current_worst
    else:
        intercepts = 1 / x

        if (not np.allclose(np.dot(A, x), b) or
                np.any(intercepts <= 1e-6) or
                np.any((intercepts + best_point) > current_worst)):
            intercepts = front_worst

    return intercepts


def associate_to_niche(fitnesses, reference_points, best_point, intercepts):
    """Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014)."""
    # Normalize by ideal point and intercepts
    fn = (fitnesses - best_point) / (intercepts - best_point)

    # Create distance matrix
    fn = np.repeat(np.expand_dims(fn, axis=1), len(reference_points), axis=1)
    norm = np.linalg.norm(reference_points, axis=1)

    distances = np.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    distances = distances[:, :, np.newaxis] * reference_points[np.newaxis, :, :] / norm[np.newaxis, :, np.newaxis]
    distances = np.linalg.norm(distances - fn, axis=2)

    # Retrieve min distance niche index
    niches = np.argmin(distances, axis=1)
    distances = distances[list(range(niches.shape[0])), niches]
    return niches, distances


def niching(individuals, k, niches, distances, niche_counts):
    selected = []
    available = np.ones(len(individuals), dtype=np.bool)
    while len(selected) < k:
        # Maximum number of individuals (niches) to select in that round
        n = k - len(selected)

        # Find the available niches and the minimum niche count in them
        available_niches = np.zeros(len(niche_counts), dtype=np.bool)
        available_niches[np.unique(niches[available])] = True
        min_count = np.min(niche_counts[available_niches])

        # Select at most n niches with the minimum count
        selected_niches = np.flatnonzero(np.logical_and(available_niches, niche_counts == min_count))
        np.random.shuffle(selected_niches)
        selected_niches = selected_niches[:n]

        for niche in selected_niches:
            # Find the individuals associated with this niche
            niche_individuals = np.flatnonzero(niches == niche)
            np.random.shuffle(niche_individuals)

            # If no individual in that niche, select the closest to reference
            # Else select randomly
            if niche_counts[niche] == 0:
                sel_index = niche_individuals[np.argmin(distances[niche_individuals])]
            else:
                sel_index = niche_individuals[0]

            # Update availability, counts and selection
            available[sel_index] = False
            niche_counts[niche] += 1
            selected.append(individuals[sel_index])

    return selected

def get_optimum_pop(pop, ref_points):

    pareto_fronts = sortLogNondominated(pop, len(pop))

    # Extract fitnesses just from first pareto_front
    fitnesses = np.array([ind.fitness.wvalues for ind in pareto_fronts[0]])
    fitnesses *= -1

    best_point = np.min(fitnesses, axis=0)
    worst_point = np.max(fitnesses, axis=0)
    extreme_points = find_extreme_points(fitnesses, best_point)
    front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    niche_available = np.ones(len(ref_points), dtype=np.bool)

    # Add one individual per reference point
    optimum = []
    for i, n in enumerate(niches):
        if niche_available[n]:
            optimum.append(pareto_fronts[0][i])
            niche_available[n] = False

    return optimum