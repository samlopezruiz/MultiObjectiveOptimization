import copy
from itertools import combinations
from math import acos, atan2, cos, pi, sin, atan

import numpy as np
from matplotlib import pyplot as plt
from numpy import array, cross, dot, float64, hypot, zeros
from numpy.linalg import norm
from random import gauss, uniform

from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization, associate_to_niches
from pymoo.core.population import Population
from pymoo.factory import get_termination, get_sampling, get_crossover, get_mutation
from pymoo.util.misc import intersect, termination_from_tuple
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from sklearn.cluster import KMeans

from src.models.moo.nsga3 import NSGA3_parallel
from src.optimization.harness.moo import get_algorithm
from src.optimization.harness.repeat import repeat_different_args


def R_2vect(R, vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)

    '''
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    '''
    print('trans')
    trans = np.eye(3) + sin(angle) * np.cross(axis, axis) + (np.eye(3) - cos(angle)) * np.matmul(axis.T, axis)

    # Calculate the rotation matrix elements.
    R[0, 0] = 1.0 + (1.0 - ca) * (x ** 2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y ** 2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z ** 2 - 1.0)


def get_pairs_from_edges(edges, points):
    pairs = []
    for node, neighs in edges.items():
        for n in neighs:
            pairs.append(np.array([points[node, :], points[n, :]]))
    return pairs


def rot_mat_from_2vect(vector_orig, vector_fin):
    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    K = np.cross(axis, np.identity(len(axis)) * -1)
    rot_mat = np.eye(len(axis)) + sin(angle) * K + (1 - cos(angle)) * np.matmul(K, K)

    return rot_mat


def transform_rows(X, transformation_matrix):
    perspective_X = []
    for p in X:
        perspective_X.append((np.matmul(transformation_matrix, p)))
    perspective_X = np.array(perspective_X)
    return perspective_X


def project_in_plane(X, n):
    n = np.array(n).reshape(1, -1)
    n_norm = n / norm(n, axis=1)
    d = X.dot(n_norm.T)
    p = np.multiply(d, n_norm)
    return X - p


def rotate_2d(perspective_ref_dirs):
    delta_x = perspective_ref_dirs[-1, 0] - perspective_ref_dirs[0, 0]
    delta_y = perspective_ref_dirs[-1, 1] - perspective_ref_dirs[0, 1]

    t = -atan(delta_x / delta_y)
    rot_2d = np.array([[cos(t), -sin(t)],
                       [sin(t), cos(t)]])

    plane_ref = perspective_ref_dirs[:, :-1]
    return transform_rows(plane_ref, rot_2d)


def append_to_list_in_dict(node, n, d):
    if node in d:
        if n not in d[node]:
            d[node] += [n]
    else:
        d[node] = [n]


def result_from_tri(tri, points=None, z_score_thold=0, plot_distances=False):
    if points is None:
        points = tri.points

    neighbors = {}
    edges = {}
    for simplice in tri.simplices:
        for x, y in combinations(simplice, 2):
            if x >= 0 and y >= 0:
                if y not in edges or x not in edges[y]:
                    append_to_list_in_dict(x, y, edges)
                append_to_list_in_dict(x, y, neighbors)
                append_to_list_in_dict(y, x, neighbors)

    all_pairs = get_pairs_from_edges(edges, points)

    distances = np.array([euclidean(p[0, Ellipsis], p[1, Ellipsis]) for p in all_pairs])

    discard = distances[zscore(distances) >= z_score_thold]
    threshold = min(discard) if len(discard) > 0 else max(distances)

    if plot_distances:
        plt.plot(distances[distances > threshold], 'o')
        plt.plot(distances[distances <= threshold], 'o')
        plt.show()

    neighbors_thold = {}
    edges_thold = {}
    for node, neighs in neighbors.items():
        for n in neighs:
            if euclidean(points[node, :], points[n, :]) <= threshold:
                if node not in edges_thold or n not in edges_thold[node]:
                    append_to_list_in_dict(node, n, edges_thold)
                append_to_list_in_dict(n, node, neighbors_thold)
                append_to_list_in_dict(node, n, neighbors_thold)

    pairs_thold = get_pairs_from_edges(edges_thold, points)

    return {
        'simplices': tri.points[tri.simplices],
        'neighbors': neighbors,
        'distances': distances,
        'neighbors_thold': neighbors_thold,
        'edges': edges,
        'all_pairs': all_pairs,
        'pairs_thold': pairs_thold
    }


def cluster_reference_directions(ref_dirs,
                                 n_clusters,
                                 z_score_thold=0.5,
                                 plot_distances=False):
    clustering = KMeans(n_clusters=n_clusters).fit(ref_dirs)
    labels = clustering.labels_
    centroids = clustering.cluster_centers_

    count = [len(np.where(labels == k)[0]) for k in np.unique(labels)]
    # print('Cluster size: {} +- {}'.format(round(np.mean(count), 2), round(np.std(count), 2)))

    rotational_matrix = rot_mat_from_2vect([0.5, 0.5, 0.5], [0, 0, 1])
    centroids_transformed = transform_rows(centroids, rotational_matrix)
    plane_centroids = centroids_transformed[:, :-1]

    tri = Delaunay(plane_centroids)
    data_structure = result_from_tri(tri,
                                     points=centroids,
                                     z_score_thold=z_score_thold,
                                     plot_distances=plot_distances)

    return {
        'ref_dirs': ref_dirs,
        'clustering': clustering,
        'labels': labels,
        'centroids': centroids,
        'centroids_transformed': centroids_transformed,
        'data_structure': data_structure
    }


class ClusterNitching:

    def __init__(self, n_obj, ref_dirs):
        self.hyp_norm = HyperplaneNormalization(n_obj)
        self.ref_dirs = ref_dirs

    def get_nitches(self, F):
        self.hyp_norm.update(F)
        ideal, nadir = self.hyp_norm.ideal_point, self.hyp_norm.nadir_point
        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(F, self.ref_dirs, ideal, nadir)
        return niche_of_individuals


def get_optimum(pop, ref_dirs):
    F = pop.get("F")

    # calculate the fronts of the population
    fronts, rank = NonDominatedSorting().do(F, return_rank=True)

    hyp_norm = HyperplaneNormalization(F.shape[1])
    hyp_norm.update(F)
    ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point
    niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(F, ref_dirs, ideal, nadir)
    closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
    opt = pop[intersect(fronts[0], closest)]
    # opt = pop[closest]

    return opt


def ini_algorithm(algorithm, problem, termination, seed=None):

    algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        algorithm.setup(problem, termination=termination, seed=seed, save_history=True, )
    else:
        if termination is not None:
            algorithm.termination = termination_from_tuple(termination)

    termination = algorithm.termination
    termination = copy.deepcopy(termination)
    algorithm.termination = termination

    return algorithm


def get_parallel_algorithms(algo_cfg,
                            cluster_ref_dir,
                            ref_dirs,
                            save_history=False,
                            ref_dirs_with_neighbors=True,
                            ):

    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)

    algorithms = []
    for k in range(algo_cfg['n_clusters']):
        if ref_dirs_with_neighbors:
            ref_dirs_partial = np.concatenate([ref_dirs[cluster_ref_dir['labels'] == k, :]] +
                                              [ref_dirs[cluster_ref_dir['labels'] == n]
                                               for n in cluster_ref_dir['data_structure']['neighbors'][k]
                                               ], axis=0)
        else:
            ref_dirs_partial = ref_dirs[cluster_ref_dir['labels'] == k, :]

        algorithm = NSGA3_parallel(
            ref_dirs=ref_dirs_partial,
            clustering=cluster_ref_dir,
            strategy=algo_cfg['strategy'],
            cluster_id=k,
            pop_size=algo_cfg['parallel_pop_size'],
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            eliminate_duplicates=True,
            save_history=save_history,
            archive_type=algo_cfg['archive_type'],
            epsilon_archive=algo_cfg['epsilon_archive']
        )
        algorithms.append(algorithm)

    return algorithms


def setup_algorithm(algorithm, problem, termination, partial_pop):
    algo = ini_algorithm(algorithm, problem, termination)
    _ = algo.infill()
    algo.pop = partial_pop
    algo.history = []
    algo.n_gen = 0
    algo.evaluator.eval(algo.problem, partial_pop, algo)
    algo.advance(infills=partial_pop)
    return algo


def run_parallel_algos(algorithms, parallel=True, n_jobs=None, verbose=1):
    args = [[algo] for algo in algorithms]
    return repeat_different_args(run_algorithm, args, parallel=parallel, n_jobs=n_jobs, verbose=verbose)


def run_algorithm(algorithm):
    if len(algorithm.pop) > 0:
        res = algorithm.run()
    else:
        res = algorithm.result()

    if hasattr(algorithm, 'archive') and algorithm.archive is not None:
        archive = copy.deepcopy(algorithm.archive.archive)
    else:
        archive = None

    return res, archive


def results_cluster_nitching(pop, cluster_nitching, n_clusters):
    F = pop.get('F')
    niche_of_individuals = cluster_nitching.get_nitches(F)
    partial_pops = [pop[niche_of_individuals == k] for k in range(n_clusters)]
    return partial_pops


def reset_algorithms(partial_pops, algorithms):
    for p, algo in zip(partial_pops, algorithms):
        algo.has_terminated = False
        algo.n_gen = 0
        algo.pop = p
