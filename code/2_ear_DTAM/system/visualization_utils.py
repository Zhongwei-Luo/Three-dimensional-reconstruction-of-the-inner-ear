import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

import plotly
from plotly.offline import init_notebook_mode
import plotly.plotly as py
import plotly.graph_objs as go
import pathlib

plotly.offline.init_notebook_mode(connected=True)


def quat_to_mat(q):
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz,
    return np.array([
        [1 - 2 * qy2 - 2 * qz2,
         2 * qx * qy - 2 * qz * qw,
         2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw,
         1 - 2 * qx2 - 2 * qz2,
         2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw,
         2 * qy * qz + 2 * qx * qw,
         1 - 2 * qx2 - 2 * qy2]])


def plot_element_wise_error(gt_pose_list, est_pose_list):
    G = np.asarray(gt_pose_list[:, :3])
    P = np.asarray(est_pose_list[:, :3])
    length = G.shape[0]
    pos_err = np.mean(np.abs(G - P), axis=1)
    x = np.arange(length)
    plt.plot(x, pos_err)

    GQuat = np.asarray(gt_pose_list[:, 3:])
    PQuat = np.asarray(est_pose_list[:, 3:])
    GR = np.empty((length, 3, 3))
    PR = np.empty((length, 3, 3))
    for i in range(length):
        GR[i] = quat_to_mat(GQuat[i, :])
        PR[i] = quat_to_mat(PQuat[i, :])
    rot_err = np.mean(np.abs(GR - PR), axis=(1, 2))
    x = np.arange(rot_err.size)
    plt.plot(x, rot_err)
    plt.show()


def plot_trajectory(*args):
    data = []
    for traj in args:
        assert isinstance(traj, np.ndarray)
        assert traj.shape[0] == 3
        X, Y, Z = np.asarray(traj)
        trace = go.Scatter3d(
            x=X, y=Y, z=Z,
            marker=dict(
                size=2,
            ),
            line=dict(
                width=1
            )
        )
        data.append(trace)
    layout = dict(
        width=1000,
        height=700,
        autosize=False,
        title='2d2d',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=-1.7428,
                    y=1.0707,
                    z=0.7100,
                )
            ),
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode='manual'
        ),
    )
    choromap = go.Figure(data=data, layout=layout)
    pathlib.Path('plot/').mkdir(parents=True, exist_ok=True)
    plotly.offline.plot(choromap, filename='plot/trajectory.html')
