from multiprocessing import cpu_count

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_problem, get_reference_directions
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src.models.moo.smsemoa import SMSEMOA
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_pop, plot_multiple_pop
from src.optimization.harness.moo import run_moo, create_title
from src.utils.plotly.plot import plot_results_moo
from src.utils.util import get_type_str
from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm


def project_in_plane(X, n):
    n = np.array(n).reshape(1, -1)
    n_norm = n / norm(n, axis=1)
    d = X.dot(n_norm.T)
    p = np.multiply(d, n_norm)
    return X - p


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
    ref_dirs = get_reference_directions("energy", prob_cfg['n_obj'], algo_cfg['pop_size'])
    ref_dirs = get_reference_directions("uniform", prob_cfg['n_obj'], n_partitions=12)
    plot_pop(ref_dirs)
    # algorithm = MOEAD(
    #     sampling=get_sampling("real_random"),
    #     crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
    #     mutation=get_mutation("real_pm", eta=mut_eta),
    #     ref_dirs=ref_dirs,
    #     n_neighbors=15,
    #     prob_neighbor_mating=0.7,
    # )

    # algo_cfg['name'] = get_type_str(algorithm)
    # result = run_moo(problem, algorithm, algo_cfg, verbose=2)

    # plot_pop(ref_dirs, save=False, file_path=None, label_scale=1, size=(1980, 1080),
    #          plot_norm=False, save_png=False, title='', opacity=1)
    #
    # # %%
    # n_clusters = cpu_count() - 1
    # clusters = KMeans(n_clusters=n_clusters)
    #
    # clusters.fit(ref_dirs)
    #
    # labels = clusters.labels_
    # centroids = clusters.cluster_centers_
    #
    # # %%
    # import numpy as np
    #
    # ref_dirs_clustered = []
    # for k in range(n_clusters):
    #     ref_dirs_clustered.append(ref_dirs[labels == k])
    #
    # plot_multiple_pop(ref_dirs_clustered, labels=None, legend_label='clusters', save=False, file_path=None,
    #                   colors_ix=None,
    #                   label_scale=1, size=(1980, 1080), color_default=0, save_png=False, title='', opacities=None,
    #                   opacity=1, prog_markers=False, prog_opacity=False, prog_color=True, plot_border=False,
    #                   use_date=False)
    #
    # # %%
    # ix_sorted = np.argsort(centroids, axis=0)
    #
    # sorted_centroids = []
    # for ix in ix_sorted:
    #     sorted_centroids.append(centroids[ix[0], :].reshape(1, -1))
    #
    # plot_multiple_pop(sorted_centroids)
    #
    # # %%
    # ref_dirs = get_reference_directions("energy", 3, algo_cfg['pop_size'])
    #
    #
    # # %%
    # def calc_proj_matrix(A):
    #     return A * np.linalg.inv(A.T * A) * A.T
    #
    #
    # def calc_proj(b, A):
    #     P = calc_proj_matrix(A)
    #     return P * b.T
    #
    #
    # tx, ty, tz = pi / 4, pi / 4, pi / 4
    # theta_x = np.array([[1, 0, 0],
    #                     [0, cos(tx), sin(tx)],
    #                     [0, -sin(tx), cos(tx)]])
    # theta_y = np.array([[cos(ty), 0, -sin(ty)],
    #                     [0, 1, 0],
    #                     [sin(ty), 0, cos(ty)]])
    # theta_z = np.array([[cos(tz), sin(tz), 0],
    #                     [-sin(tz), cos(tz), 0],
    #                     [0, 0, 1]])
    # trans = theta_z * theta_y * theta_x
    #
    # # trans = np.array([[-0.5, -0.5, 0.5],
    # #               [0.5, -0.5, 0.5],
    # #               [-0.5, -0.5, 0.5]])
    #
    # n = np.array([0, 0.5, 0.5]).reshape(1, -1)
    # u = n / norm(n, axis=1)
    # t = pi/4
    # trans = cos(t) * np.eye(3) + sin(t) * np.cross(u, u) + (1 - cos(t)) * np.matmul(u.T, u)
    #
    # perspective_ref_dirs = []
    # for p in ref_dirs:
    #     # perspective_ref_dirs.append((np.matmul(p, theta_z)))
    #     perspective_ref_dirs.append((np.matmul(trans, p)))
    #     # perspective_ref_dirs.append(np.matmul(theta_y, np.matmul(theta_x, p)))
    #     # perspective_ref_dirs.append(np.matmul(theta_z, np.matmul(theta_y, np.matmul(p, theta_x))))
    # perspective_ref_dirs = np.array(perspective_ref_dirs)
    # plot_pop(perspective_ref_dirs)
    #%%
    from math import acos, atan2, cos, pi, sin
    from numpy import array, cross, dot, float64, hypot, zeros
    from numpy.linalg import norm
    from random import gauss, uniform
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

    rot = np.eye(3)
    R_2vect(rot, [0.5, 0.5, 0.5], [0, 0, 1])

    perspective_ref_dirs = []
    for p in ref_dirs:
        perspective_ref_dirs.append((np.matmul(rot, p)))
    perspective_ref_dirs = np.array(perspective_ref_dirs)

    ix = np.argsort(perspective_ref_dirs[:, 0])
    # plot_pop(ref_dirs)
    plt.plot(perspective_ref_dirs[:, 0], perspective_ref_dirs[:, 1], 'o')
    plt.show()

    delta_x = perspective_ref_dirs[-1, 0] - perspective_ref_dirs[0, 0]
    delta_y = perspective_ref_dirs[-1, 1] - perspective_ref_dirs[0, 1]

    import math
    t = -math.atan(delta_x / delta_y)
    rot_2d = np.array([[cos(t), -sin(t)],
                      [sin(t), cos(t)]])

    plane_ref = perspective_ref_dirs[:, :-1]
    perspective_2d_ref_dirs = []
    for p in plane_ref:
        perspective_2d_ref_dirs.append((np.matmul(rot_2d, p)))
    perspective_2d_ref_dirs = np.array(perspective_2d_ref_dirs)

    plt.plot(perspective_2d_ref_dirs[:, 0], perspective_2d_ref_dirs[:, 1], 'o')
    plt.show()

    #%%
    points = perspective_2d_ref_dirs
    tri = Delaunay(points)  # , qhull_options='QJ')
    #
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()


    #%%
    # # %%
    # ref_dirs = get_reference_directions("energy", 3, algo_cfg['pop_size'])
    # ref_dirs = np.concatenate([ref_dirs, np.ones((ref_dirs.shape[0], 1))], axis=1)
    # # n =
    # # r =
    # # l =
    # # f =
    # # perspective = np.array([[2*n / (r - l), 0, -sin(ty)],
    # #                         [0, 1, 0],
    # #                         [sin(ty), 0, cos(ty)]])
    # perspective[3, 3] = 0
    # perspective[2, 3] = 1
    #
    # perspective_ref_dirs = []
    # for p in ref_dirs:
    #     perspective_ref_dirs.append(np.matmul(perspective, p))
    # perspective_ref_dirs = np.array(perspective_ref_dirs)
    #
    # plot_pop(perspective_ref_dirs)
    #
    # # %%
    # ref_dirs = ref_dirs - np.array([0.5, 0.5, 0.5])
    # plane_ref_dirs = project_in_plane(ref_dirs, [0, 0, 1])[:, :-1]
    # # d2_ref_dirs = plane_ref_dirs[plane_ref_dirs[:, 2] == 0, :]
    # # plot_pop(plane_ref_dirs)
    #
    # plt.plot(plane_ref_dirs[:, 0], plane_ref_dirs[:, 1], 'o')
    # plt.show()
    #
    # # %%
    # ref_dirs = get_reference_directions("energy", 2, algo_cfg['pop_size'])
    # plt.plot(ref_dirs[:, 0], ref_dirs[:, 1], 'o')
    # plt.show()
    # #%%
    # import numpy as np
    # import math as m
    #
    # def Rx(theta):
    #     return np.matrix([[1, 0, 0],
    #                       [0, m.cos(theta), -m.sin(theta)],
    #                       [0, m.sin(theta), m.cos(theta)]])
    #
    #
    # def Ry(theta):
    #     return np.matrix([[m.cos(theta), 0, m.sin(theta)],
    #                       [0, 1, 0],
    #                       [-m.sin(theta), 0, m.cos(theta)]])
    #
    #
    # def Rz(theta):
    #     return np.matrix([[m.cos(theta), -m.sin(theta), 0],
    #                       [m.sin(theta), m.cos(theta), 0],
    #                       [0, 0, 1]])
    # phi = m.pi / 4
    # theta = m.pi / 2
    # psi = 0 #m.pi / 4
    # print("phi =", phi)
    # print("theta  =", theta)
    # print("psi =", psi)
    #
    # Rot = Rz(psi) * Ry(theta) * Rz(phi)
    # print(np.round(Rot, decimals=2))
    #
    # # v1 = np.array([[1], [0], [0]])
    # # v2 = Rot * v1
    #
    # perspective_ref_dirs=[]
    # for p in ref_dirs:
    #     perspective_ref_dirs.append(np.array((Rot * p.reshape(-1, 1)).T))
    # perspective_ref_dirs = np.concatenate(perspective_ref_dirs, axis=0)
    # plot_pop(perspective_ref_dirs)
    #
    # # %%
    # trans_ref_dirs = ref_dirs - np.array([0.5, 0.5, 0.5])
    # r = R.from_euler('xyz', [0, 0, 45], degrees=True)
    # print(r.as_matrix())
    # rot_ref_dirs = np.array([r.apply(c) for c in trans_ref_dirs])
    # plot_multiple_pop([trans_ref_dirs, ref_dirs, rot_ref_dirs])
    #
    # # %%
    #
    # plane_ref_dirs = project_in_plane(ref_dirs, [0, 0, 1])
    # plane_ref_dirs = plane_ref_dirs[:, :-1]
    # import matplotlib.pyplot as plt
    #
    # plt.plot(plane_ref_dirs[:, 0], plane_ref_dirs[:, 1], 'o')
    # plt.show()
    # # plot_pop(plane_ref_dirs)
    # # %%
    # from scipy.spatial import Delaunay
    #
    # points = plane_ref_dirs  # np.array([[2, 0], [0, 1.1], [1, 0], [1, 1]])
    #
    # tri = Delaunay(points)  # , qhull_options='QJ')
    #
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.show()
