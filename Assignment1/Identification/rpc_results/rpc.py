import matplotlib.pyplot as plt
import numpy as np

hist_types = ["grayvalue", "rg", "rgb", "dxdy"]
dist_types = ["chi2", "intersect", "l2"]
colors = ["r", "g", "b"]

for index, hist_type in enumerate(hist_types):
    plt.subplot(2, 2, index+1)
    for idx, dist_type in enumerate(dist_types):
        path = r"./rpc_{}_{}.csv".format(hist_type, dist_type)
        data = np.genfromtxt(path, delimiter=";", names=["recall", "precision"])

        plt.plot(data["precision"], data["recall"], color=colors[idx])

    plt.title(hist_type + " histogram")
    plt.axis([0, 1, 0, 1]);
    plt.xlabel('1 - precision');
    plt.ylabel('recall');
    plt.legend(dist_types, loc='best')

plt.show()

for index, hist_type in enumerate(["rg", "rgb"]):
    for idx, dist_type in enumerate(dist_types):
        path = r"./rpc_{}_{}.csv".format(hist_type, dist_type)
        data = np.genfromtxt(path, delimiter=";", names=["recall", "precision"])
        plt.plot(data["precision"], data["recall"], label="{} - {}".format(hist_type, dist_type))
    
plt.title("RG histogram vs RGB histogram")
plt.axis([0, 1, 0, 1]);
plt.xlabel('1 - precision');
plt.ylabel('recall');
plt.legend()
plt.show()
