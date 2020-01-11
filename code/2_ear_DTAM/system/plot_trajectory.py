import numpy as np
import matplotlib.pyplot as plt
from system.visualization_utils import *


def main():
    gt_pose = np.load("result/trajectory/gt_pose.npy")
    est_pose = np.load("result/trajectory/est_pose.npy")

    gt_position = gt_pose[:, :3].T
    est_position = est_pose[:, :3].T
    print(gt_pose.shape)
    plot_trajectory(gt_position, est_position)


if __name__ == '__main__':
    main()
