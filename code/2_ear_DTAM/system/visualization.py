import numpy as np
import matplotlib.pyplot as plt
import random

# print(random.randint(0,9))
frame_no = 20


def plot_cost(u, v):
    sum_cost = None
    for i in range(1, frame_no + 1):
        cost_volume = np.load("result/cost_volume/" + str(i) + ".npy")
        # print(cost_volume)
        # print(cost_volume)
        if sum_cost is None:
            sum_cost = np.zeros(cost_volume.shape[2])
        cost = cost_volume[u, v, :]
        cost[cost > 1.5] = 1.5
        del cost_volume
        sum_cost += cost
        plt.plot(range(cost.size), cost, linewidth=0.5)
    plt.plot(range(sum_cost.size), sum_cost / frame_no, linewidth=2.0)
    plt.show()


def main():
    # plot_cost(125, 419)
    w = random.randint(0, 640 - 1)
    h = random.randint(0, 480 - 1)
    #plot_cost(w, h)
    #print(w, h)
    plot_cost(206, 115)
    # plot_cost(310, 290)


if __name__ == '__main__':
    main()
