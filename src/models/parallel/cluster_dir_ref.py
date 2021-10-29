from multiprocessing import cpu_count

import numpy as np
from matplotlib import pyplot as plt
from pymoo.factory import get_problem, get_reference_directions
from scipy.spatial.qhull import Delaunay
from sklearn.cluster import KMeans

from src.models.moo.utils.plot import plot_multiple_pop
from src.models.parallel.utils import transform_rows, rot_mat_from_2vect, result_from_tri

if __name__ == '__main__':
    prob_cfg = {'name': 'dtlz2',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [5] * 3  # used for HV history
                }
    algo_cfg = {'termination': ('n_gen', 100),
                'max_gen': 150,
                'pop_size': 100,
                'hv_ref': [10] * 3  # used only for SMS-EMOA
                }

    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj'])

    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)

    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)

    # %%
    n_clusters = 5 #cpu_count() - 6
    clustering = KMeans(n_clusters=n_clusters).fit(ref_dirs)
    labels = clustering.labels_
    centroids = clustering.cluster_centers_

    count = [len(np.where(labels == k)[0]) for k in np.unique(labels)]
    print('count: {} +- {}'.format(round(np.mean(count), 2), round(np.std(count), 2)))

    ref_dirs_clustered = [ref_dirs[labels == k] for k in range(n_clusters)]



    # %%
    rotational_matrix = rot_mat_from_2vect([0.5, 0.5, 0.5], [0, 0, 1])
    centroids_transformed = transform_rows(centroids, rotational_matrix)
    plane_centroids = centroids_transformed[:, :-1]

    plot_multiple_pop([centroids, centroids_transformed],
                      labels=['centroids', 'plane centroids'])

    tri = Delaunay(plane_centroids)

    data_structures = result_from_tri(tri,
                                      points=centroids,
                                      z_score_thold=1,
                                      plot_distances=True)

    plot_multiple_pop(ref_dirs_clustered+[centroids],
                      labels = ['cluster ' + str(i) for i in range(n_clusters)] + ['centroids'],
                      pair_lines=data_structures['pairs_thold'],
                      plot_nadir=False)

