from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize

if __name__ == '__main__':
    problem = get_problem("wfg2", n_var=10, n_obj=3)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=False)

    get_visualization("scatter").add(res.F).show()